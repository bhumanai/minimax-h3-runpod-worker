# MiniMax H3 Ref2VA RunPod worker

An isolated RunPod Serverless worker for native MiniMax H3 reference-to-video-with-audio generation. The image pins ComfyUI commit `3216c62e9962c3babd28a4dfea6e5aef50b8fe16`, because the current RunPod base release still ships ComfyUI 0.34 without the native H3 nodes.

The worker accepts a ComfyUI API graph plus small base64 reference files. It writes the generated MP4 to a caller-provided presigned upload URL, avoiding RunPod's response-size limit. Model files are downloaded once onto the attached `/runpod-volume` network volume and reused across scale-to-zero cold starts.

## Runtime contract

- `MINIMAX_H3_LICENSE_ACCEPTED=1` is required.
- Attach at least a 60 GB network volume at `/runpod-volume`.
- Use a Blackwell GPU for the bundled NVFP4 text encoder; the intended endpoint is one B200.
- The production-safe default is `workersMin=0`, `workersMax=1`.

Request shape:

```json
{
  "input": {
    "workflow": {"1": {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}},
    "files": [
      {"name": "identity.png", "data": "BASE64"},
      {"name": "driver.mp4", "data": "BASE64"},
      {"name": "new-words.wav", "data": "BASE64"}
    ],
    "output_upload_url": "PRESIGNED_PUT_URL",
    "output_download_url": "PRESIGNED_GET_URL"
  }
}
```

`h3_workflow.py` builds the full official-node Ref2VA graph used by the smoke test.
Pass `audio_name="new-words.wav"` to use standalone speech as `<Audio 1>` while
retaining the driver video as `<Video 1>`. When `audio_name` is omitted, the
driver video's own soundtrack remains `<Audio 1>` for backward compatibility.
Pass `video_name=None` together with standalone audio to run image-plus-audio
generation without a motion reference; in that mode the prompt only needs
`<Picture 1>` and `<Audio 1>`.
Its prompt must refer to every supplied medium with the literal H3 tags
`<Picture 1>`, `<Audio 1>`, and, when present, `<Video 1>`; untagged references
are rejected before a paid job can be submitted.

## Verification

```bash
python -m unittest discover -s tests -v
```
