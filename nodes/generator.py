"""
Generator node for Alice T2V pipeline.
"""
import logging

import torch

from ..utils.video_utils import alice_video_to_comfy

log = logging.getLogger(__name__)

# Preset resolutions from Alice's SIZE_CONFIGS.
# README also mentions 960*960, 1024*576, 576*1024 but these are not in the
# official SIZE_CONFIGS - they may work but are untested by Mirage.
_RESOLUTION_PRESETS = [
    "1280*720",   # 16:9 landscape  (recommended)
    "720*1280",   # 9:16 portrait
    "832*480",    # 16:9 landscape  (lower VRAM)
    "480*832",    # 9:16 portrait   (lower VRAM)
    "960*960",    # 1:1 square      (README only, untested)
    "1024*576",   # 16:9 landscape  (README only, untested)
    "576*1024",   # 9:16 portrait   (README only, untested)
    "custom",     # use custom_width / custom_height below
]


def _parse_resolution(resolution: str, custom_width: int, custom_height: int):
    """
    Return (width, height) from a resolution string or custom values.
    Width/height must each be divisible by 16 (VAE spatial stride requirement).
    """
    if resolution == "custom":
        w, h = custom_width, custom_height
    else:
        w_str, h_str = resolution.split("*")
        w, h = int(w_str), int(h_str)

    # Snap to nearest multiple of 16
    w_snapped = round(w / 16) * 16
    h_snapped = round(h / 16) * 16
    if w_snapped != w or h_snapped != h:
        log.warning(
            f"[Eric_AliceT2V] Resolution {w}×{h} not divisible by 16. "
            f"Snapped to {w_snapped}×{h_snapped}."
        )
    return w_snapped, h_snapped


class Eric_AliceT2V:
    """
    Generates a video from a text prompt using the Alice T2V 14B MoE model.

    Alice uses a two-expert MoE DiT: the high-noise expert handles early
    denoising (structure/composition) and the low-noise expert handles late
    denoising (detail refinement). The switchover is determined automatically
    by the SNR boundary baked into the model config (boundary=0.875, i.e.
    the high-noise expert runs for timesteps > 875 and low-noise for ≤ 875).

    At the default 40 steps this gives approximately 35 high-noise steps and
    5 low-noise steps. The model was distilled to work at as few as 4 total
    steps, but 40 is the tuned default and produces the best quality.

    Output: IMAGE tensor [T, H, W, C] float32 in [0, 1].
    Compatible with VHS_VideoCombine, SaveAnimatedWEBP, and all standard
    ComfyUI video nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ALICE_PIPELINE",),
                "prompt": ("STRING", {
                    "default": (
                        "A cinematic shot of a majestic waterfall in a lush tropical forest, "
                        "golden hour lighting, mist in the air, 4K, photorealistic"
                    ),
                    "multiline": True,
                    "tooltip": (
                        "Describe the video in natural language (not tag lists). "
                        "Include subject, action, setting, lighting, and camera style. "
                        "Max 512 tokens - longer prompts are truncated."
                    ),
                }),
                "resolution": (_RESOLUTION_PRESETS, {
                    "default": "1280*720",
                    "tooltip": (
                        "Output resolution (width*height). "
                        "The four official presets are recommended. "
                        "960*960, 1024*576, 576*1024 are from the README but untested. "
                        "Choose 'custom' to enter arbitrary dimensions (snapped to multiples of 16)."
                    ),
                }),
                "custom_width": ("INT", {
                    "default": 1280,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Used only when resolution = 'custom'. Must be a multiple of 16.",
                }),
                "custom_height": ("INT", {
                    "default": 720,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Used only when resolution = 'custom'. Must be a multiple of 16.",
                }),
                "frame_num": ("INT", {
                    "default": 81,
                    "min": 9,
                    "max": 201,
                    "step": 4,
                    "tooltip": (
                        "Number of frames. Must satisfy (frame_num - 1) % 4 == 0. "
                        "81 = 5 seconds at 16 fps. Node auto-corrects invalid values."
                    ),
                }),
                "steps": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "Denoising steps. 40 is the model's tuned default. "
                        "With boundary=0.875, ~35 steps go to the high-noise expert "
                        "(structure) and ~5 to the low-noise expert (detail). "
                        "Fewer than ~20 steps noticeably degrades detail quality."
                    ),
                }),
                "cfg_high_noise": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for the high-noise expert (structure phase). Default: 4.0",
                }),
                "cfg_low_noise": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for the low-noise expert (detail phase). Default: 3.0",
                }),
                "solver": (["unipc", "dpm++"], {
                    "default": "unipc",
                    "tooltip": (
                        "ODE solver. UniPC is recommended for ≤40 steps. "
                        "DPM++ may perform better at 50+ steps. Only two solvers are implemented."
                    ),
                }),
                "shift": ("FLOAT", {
                    "default": 12.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Timestep shift. Model config default: 12.0. "
                        "Higher = more steps at high-noise (structure) timesteps. "
                        "Lower = more evenly spread, giving the detail expert more work."
                    ),
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1,
                    "tooltip": "-1 = random seed each run.",
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Negative prompt. Leave empty to use Alice's built-in Chinese "
                        "negative prompt (strongly recommended - it was tuned with the model)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "Eric/Alice"

    def generate(
        self,
        pipeline,
        prompt: str,
        resolution: str,
        custom_width: int,
        custom_height: int,
        frame_num: int,
        steps: int,
        cfg_high_noise: float,
        cfg_low_noise: float,
        solver: str,
        shift: float,
        seed: int,
        negative_prompt: str = "",
    ):
        # Enforce Alice's frame_num constraint: (frame_num - 1) % 4 == 0
        if (frame_num - 1) % 4 != 0:
            corrected = round((frame_num - 1) / 4) * 4 + 1
            log.warning(
                f"[Eric_AliceT2V] frame_num={frame_num} is invalid "
                f"(must satisfy (n-1)%4==0). Auto-corrected to {corrected}."
            )
            frame_num = corrected

        width, height = _parse_resolution(resolution, custom_width, custom_height)
        size = (width, height)

        # guide_scale tuple is (low_noise_cfg, high_noise_cfg)
        guide_scale = (cfg_low_noise, cfg_high_noise)

        offload_model = getattr(pipeline, "_eric_offload_model", True)

        print(
            f"[Eric_AliceT2V] Generating {width}×{height} | {frame_num} frames | "
            f"{steps} steps | CFG ({cfg_low_noise}/{cfg_high_noise}) | "
            f"shift={shift} | solver={solver} | seed={seed}"
        )

        with torch.inference_mode():
            video_tensor = pipeline.generate(
                input_prompt=prompt,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=solver,
                sampling_steps=steps,
                guide_scale=guide_scale,
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=offload_model,
            )

        if video_tensor is None:
            raise RuntimeError(
                "[Eric_AliceT2V] Pipeline returned None. "
                "This can happen if running in a distributed context where rank != 0."
            )

        # Alice output: [C, T, H, W] float, range [-1, 1]
        # ComfyUI IMAGE: [T, H, W, C] float32, range [0, 1]
        frames = alice_video_to_comfy(video_tensor)

        print(f"[Eric_AliceT2V] Done. Output: {frames.shape} {frames.dtype}")
        return (frames,)
