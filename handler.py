from __future__ import annotations

import base64
import binascii
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import requests
import runpod


COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/comfyui/input"))
OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/comfyui/output"))
ALLOWED_INPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".wav", ".mp3", ".flac"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
MAX_FILE_BYTES = 8_000_000
MAX_TOTAL_FILE_BYTES = 9_000_000
MAX_INLINE_OUTPUT_BYTES = 7_000_000


def safe_input_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each input file needs a non-empty name")
    candidate = Path(value)
    if candidate.name != value or value in {".", ".."}:
        raise ValueError(f"Input filename must be a basename: {value!r}")
    if candidate.suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported input extension: {candidate.suffix}")
    return value


def decode_base64_file(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("Input file data must be base64 text")
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Input file data is not valid base64") from exc
    if len(decoded) > MAX_FILE_BYTES:
        raise ValueError(f"Input file exceeds {MAX_FILE_BYTES} bytes")
    return decoded


def write_input_files(files: list[dict[str, Any]]) -> list[Path]:
    if not isinstance(files, list) or not files:
        raise ValueError("input.files must contain at least one file")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each input.files item must be an object")
        name = safe_input_name(item.get("name"))
        payload = decode_base64_file(item.get("data"))
        total += len(payload)
        if total > MAX_TOTAL_FILE_BYTES:
            raise ValueError(f"Combined input files exceed {MAX_TOTAL_FILE_BYTES} bytes")
        destination = INPUT_DIR / name
        temporary = destination.with_suffix(destination.suffix + ".upload")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        written.append(destination)
    return written


def snapshot_videos() -> set[Path]:
    if not OUTPUT_DIR.exists():
        return set()
    return {
        path.resolve()
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }


def _walk_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            yield value
        for child in value.values():
            yield from _walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_records(child)


def history_video_paths(history: dict[str, Any]) -> list[Path]:
    output_root = OUTPUT_DIR.resolve()
    found: list[Path] = []
    for record in _walk_records(history.get("outputs", {})):
        filename = Path(record["filename"]).name
        if Path(filename).suffix.lower() not in VIDEO_SUFFIXES:
            continue
        subfolder = record.get("subfolder", "")
        if not isinstance(subfolder, str):
            continue
        candidate = (OUTPUT_DIR / subfolder / filename).resolve()
        if candidate == output_root or output_root not in candidate.parents:
            continue
        if candidate.is_file():
            found.append(candidate)
    return list(dict.fromkeys(found))


def wait_for_comfy(session: requests.Session, timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = session.get(f"{COMFY_URL}/system_stats", timeout=5)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"ComfyUI did not become ready: {last_error}")


def queue_workflow(session: requests.Session, workflow: dict[str, Any]) -> str:
    response = session.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"ComfyUI rejected workflow ({response.status_code}): {response.text[:2000]}")
    payload = response.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI response had no prompt_id: {payload}")
    return str(prompt_id)


def wait_for_history(
    session: requests.Session,
    prompt_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = session.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        payload = response.json().get(prompt_id)
        if payload:
            status = payload.get("status", {})
            if status.get("completed"):
                messages = status.get("messages", [])
                errors = [message for message in messages if message and message[0] == "execution_error"]
                if errors:
                    raise RuntimeError(f"ComfyUI execution error: {errors[-1][1]}")
                return payload
        time.sleep(2)
    try:
        session.post(f"{COMFY_URL}/interrupt", timeout=10)
    except requests.RequestException:
        pass
    raise TimeoutError(f"ComfyUI workflow {prompt_id} exceeded {timeout_seconds} seconds")


def upload_or_inline(path: Path, upload_url: str | None, download_url: str | None) -> dict[str, Any]:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if upload_url:
        if not download_url:
            raise ValueError("output_download_url is required with output_upload_url")
        with path.open("rb") as source:
            response = requests.put(
                upload_url,
                data=source,
                headers={"Content-Type": content_type},
                timeout=(30, 900),
            )
        response.raise_for_status()
        return {
            "filename": path.name,
            "type": "url",
            "data": download_url,
            "bytes": path.stat().st_size,
        }

    size = path.stat().st_size
    if size > MAX_INLINE_OUTPUT_BYTES:
        raise ValueError("Output is too large to inline; provide presigned output upload/download URLs")
    return {
        "filename": path.name,
        "type": "base64",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "bytes": size,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise ValueError("Job input must be an object")
    workflow = payload.get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("input.workflow must be a non-empty ComfyUI API workflow")

    written: list[Path] = []
    started = time.monotonic()
    before = snapshot_videos()
    try:
        runpod.serverless.progress_update(job, "Uploading reference files")
        written = write_input_files(payload.get("files"))
        with requests.Session() as session:
            runpod.serverless.progress_update(job, "Waiting for ComfyUI")
            wait_for_comfy(session)
            runpod.serverless.progress_update(job, "Running MiniMax H3 Ref2VA")
            prompt_id = queue_workflow(session, workflow)
            history = wait_for_history(
                session,
                prompt_id,
                int(payload.get("workflow_timeout_seconds", 3_600)),
            )

        videos = history_video_paths(history)
        if not videos:
            videos = sorted(snapshot_videos() - before, key=lambda item: item.stat().st_mtime_ns)
        if not videos:
            raise RuntimeError("Workflow completed but produced no video file")

        result = upload_or_inline(
            videos[-1],
            payload.get("output_upload_url"),
            payload.get("output_download_url"),
        )
        return {
            "prompt_id": prompt_id,
            "videos": [result],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
