from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests


REPOSITORY = "Comfy-Org/MiniMax-H3"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/main"
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ModelFile:
    relative_path: str
    size: int


MODEL_FILES = (
    ModelFile("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 20_970_379_616),
    ModelFile("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15_687_142_551),
    ModelFile("vae/minimax_h3_video_vae_fp16.safetensors", 5_207_808_496),
    ModelFile("vae/minimax_h3_audio_vae_fp32.safetensors", 605_254_808),
    ModelFile("loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1_956_193_000),
)


def _download_one(session: requests.Session, root: Path, model: ModelFile) -> None:
    destination = root / model.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == model.size:
        print(f"h3-models: ready {model.relative_path}", flush=True)
        return

    if destination.exists():
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > model.size:
        partial.unlink()
        offset = 0

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(
        f"{BASE_URL}/{model.relative_path}",
        headers=headers,
        stream=True,
        timeout=(30, 300),
    )
    response.raise_for_status()

    append = offset > 0 and response.status_code == 206
    mode = "ab" if append else "wb"
    if not append:
        offset = 0

    print(
        f"h3-models: downloading {model.relative_path} from {offset / 1_000_000_000:.2f} GB",
        flush=True,
    )
    written = offset
    next_report = written + 1_000_000_000
    with partial.open(mode) as output:
        for chunk in response.iter_content(CHUNK_BYTES):
            if not chunk:
                continue
            output.write(chunk)
            written += len(chunk)
            if written >= next_report:
                print(
                    f"h3-models: {model.relative_path} {written / model.size:.0%}",
                    flush=True,
                )
                next_report += 1_000_000_000

    actual_size = partial.stat().st_size
    if actual_size != model.size:
        raise RuntimeError(
            f"Size mismatch for {model.relative_path}: expected {model.size}, got {actual_size}"
        )
    os.replace(partial, destination)
    print(f"h3-models: completed {model.relative_path}", flush=True)


def main() -> None:
    if os.getenv("MINIMAX_H3_LICENSE_ACCEPTED") != "1":
        raise RuntimeError("MiniMax H3 authorization acknowledgement is required")

    root = Path(os.getenv("H3_MODEL_ROOT", "/runpod-volume/models"))
    root.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = "BHuman-H3-RunPod/1.0"
        for model in MODEL_FILES:
            _download_one(session, root, model)


if __name__ == "__main__":
    main()
