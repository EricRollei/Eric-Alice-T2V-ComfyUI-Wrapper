"""
Video tensor conversion utilities for Eric_Alice_t2v.
"""
import torch


def alice_video_to_comfy(video: torch.Tensor) -> torch.Tensor:
    """
    Convert Alice pipeline output to ComfyUI IMAGE format.

    Alice generate() returns:  [C, T, H, W]  float,   range [-1.0, 1.0]
    ComfyUI IMAGE expects:     [T, H, W, C]  float32, range [ 0.0, 1.0]

    Args:
        video: Raw output tensor from AliceTextToVideo.generate()

    Returns:
        Frame batch tensor ready for ComfyUI video/image nodes.
    """
    video = video.float()               # ensure float32 (was bfloat16 from model)
    video = video.clamp(-1.0, 1.0)     # safety clamp before rescale
    video = (video + 1.0) / 2.0        # [-1, 1] → [0, 1]
    video = video.permute(1, 2, 3, 0)  # [C, T, H, W] → [T, H, W, C]
    return video.cpu().contiguous()
