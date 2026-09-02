from __future__ import annotations

from typing import Any


REQUIRED_REFERENCE_TAGS = ("<Picture 1>", "<Video 1>", "<Audio 1>")


def _validate_reference_prompt(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("MiniMax H3 Ref2VA requires a non-empty prompt")
    missing = [tag for tag in REQUIRED_REFERENCE_TAGS if tag not in prompt]
    if missing:
        raise ValueError(
            "MiniMax H3 Ref2VA prompt must explicitly bind every supplied reference with "
            + ", ".join(missing)
        )


def build_h3_ref2va_workflow(
    image_name: str,
    video_name: str,
    *,
    audio_name: str | None = None,
    prompt: str,
    seed: int = 20260901,
    width: int = 768,
    height: int = 1024,
    length: int = 124,
    steps: int = 20,
    turbo: bool = False,
) -> dict[str, dict[str, Any]]:
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32")
    if length < 5 or length % 17 != 5:
        raise ValueError("length must be at least 5 frames and congruent to 5 modulo 17")
    if turbo and steps != 4:
        raise ValueError("The bundled H3 Turbo LoRA is a 4-step adapter")
    _validate_reference_prompt(prompt)
    if audio_name is not None and (not isinstance(audio_name, str) or not audio_name.strip()):
        raise ValueError("audio_name must be a non-empty filename when provided")

    model_node = "8" if turbo else "6"
    workflow: dict[str, dict[str, Any]] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "6": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "7": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "9": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["7", 0],
                "vae": ["4", 0],
                "audio_vae": ["5", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
                "ref_image_size": "match",
                "ref_images.ref_image_0": ["1", 0],
                "ref_videos.ref_video_0": ["3", 0],
            },
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": [model_node, 0], "conditioning": ["9", 0]},
        },
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "13": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [model_node, 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["10", 0],
                "sampler": ["12", 0],
                "sigmas": ["13", 0],
                "latent_image": ["9", 1],
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["14", 0], "vae": ["4", 0]},
        },
        "16": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["14", 0], "vae": ["5", 0]},
        },
        "17": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["15", 0],
                "audio": ["16", 0],
                "fps": 24.0,
                "bit_depth": 8,
                "color_space": "sRGB",
            },
        },
        "18": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["17", 0],
                "filename_prefix": "video/BHuman_H3_Ref2VA",
                "format": "auto",
            },
        },
    }
    if audio_name is None:
        workflow["9"]["inputs"]["ref_video_audios.ref_video_audio_0"] = ["3", 1]
    else:
        workflow["19"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
        workflow["9"]["inputs"]["ref_audios.ref_audio_0"] = ["19", 0]
    if turbo:
        workflow["8"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["6", 0],
                "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "strength_model": 1.0,
            },
        }
    return workflow
