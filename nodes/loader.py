"""
Loader node for Alice T2V pipeline.
"""
import json
import logging
import os
from copy import deepcopy

import torch

log = logging.getLogger(__name__)

# Module-level pipeline cache.
# Key: (ckpt_dir, device_id, offload_model, t5_cpu, convert_model_dtype)
# Value: AliceTextToVideo instance
_pipeline_cache: dict = {}


def _ensure_tokenizer(ckpt_dir: str) -> None:
    """
    Ensure the umt5-xxl tokenizer files exist inside ckpt_dir/google/umt5-xxl/.
    Downloads from HuggingFace if missing (~2 MB, config/vocab files only).
    """
    tokenizer_dir = os.path.join(ckpt_dir, 'google', 'umt5-xxl')
    if os.path.isdir(tokenizer_dir):
        return

    print(
        f"[Eric_AliceLoader] Tokenizer not found at:\n  {tokenizer_dir}\n"
        "  Downloading google/umt5-xxl tokenizer from HuggingFace (~2 MB)..."
    )
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('google/umt5-xxl')
        tok.save_pretrained(tokenizer_dir)
        print(f"[Eric_AliceLoader] Tokenizer saved to: {tokenizer_dir}")
    except Exception as e:
        raise RuntimeError(
            f"[Eric_AliceLoader] Failed to download tokenizer: {e}\n"
            "Check your internet connection, or manually place the umt5-xxl "
            f"tokenizer files in:\n  {tokenizer_dir}"
        ) from e


def _ensure_shard_index(model_dir: str) -> None:
    """
    Generate diffusion_pytorch_model.safetensors.index.json if it is missing.

    The Alice HF upload includes sharded safetensors files but omits the index
    JSON that diffusers' from_pretrained() needs to map parameter names to
    shard files. We generate it by reading each shard's metadata header
    (key names only - no tensor data is loaded into memory).
    """
    index_path = os.path.join(
        model_dir, "diffusion_pytorch_model.safetensors.index.json"
    )
    if os.path.isfile(index_path):
        return

    import glob
    shard_pattern = os.path.join(
        model_dir, "diffusion_pytorch_model-*-of-*.safetensors"
    )
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        # Single-file or non-standard layout - let from_pretrained handle it
        return

    print(
        f"[Eric_AliceLoader] Generating shard index for: {model_dir}\n"
        f"  Found {len(shard_files)} shard(s) - reading metadata headers..."
    )

    try:
        import struct
        weight_map: dict[str, str] = {}

        for shard_path in shard_files:
            shard_name = os.path.basename(shard_path)
            # safetensors format: first 8 bytes = uint64 header length,
            # followed by header_length bytes of UTF-8 JSON.
            with open(shard_path, "rb") as f:
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    raise IOError(f"Truncated file: {shard_name}")
                header_len = struct.unpack("<Q", header_len_bytes)[0]
                header_json = f.read(header_len).decode("utf-8")

            header = json.loads(header_json)
            for key in header.keys():
                if key == "__metadata__":
                    continue
                weight_map[key] = shard_name

        # Build the index structure diffusers expects
        total_size = sum(
            os.path.getsize(p) for p in shard_files
        )
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(
            f"[Eric_AliceLoader] Shard index written ({len(weight_map)} keys): "
            f"{index_path}"
        )

    except Exception as e:
        raise RuntimeError(
            f"[Eric_AliceLoader] Failed to generate shard index for {model_dir!r}: {e}\n"
            "Ensure all shard files are fully downloaded and not corrupted."
        ) from e


def _build_config(ckpt_dir: str):
    """
    Return a deep copy of the t2v-14b config with the tokenizer path fixed
    for the current OS (replaces forward-slash separator with os.sep).
    """
    from alice.configs import ALICE_CONFIGS
    cfg = deepcopy(ALICE_CONFIGS["t2v-14b"])
    # Replace 'google/umt5-xxl' with an OS-correct relative path so that
    # os.path.join(ckpt_dir, cfg.t5_tokenizer) resolves cleanly on Windows.
    cfg.t5_tokenizer = os.path.join('google', 'umt5-xxl')
    return cfg


class Eric_AliceLoader:
    """
    Loads and caches the Alice T2V 14B MoE pipeline.

    The pipeline (T5 encoder + two DiT experts + VAE) is held in memory between
    runs. Changing any parameter forces an unload and reload.

    Vendor setup required: run setup_vendor.py once before first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Path to the Alice model checkpoint directory "
                        "(downloaded from gomirageai/Alice-T2V-14B-MoE on HuggingFace)."
                    ),
                }),
                "device_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 7,
                    "step": 1,
                    "tooltip": "CUDA device index (0 = first GPU).",
                }),
                "offload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Swap inactive DiT expert (high/low noise) to CPU during generation. "
                        "Saves ~14B params of VRAM per step at the cost of transfer overhead. "
                        "Disable if you have 96GB+ VRAM for maximum speed."
                    ),
                }),
                "t5_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep the T5 text encoder (~6B params) on CPU throughout. "
                        "Recommended - encoding is fast and frees VRAM for the DiTs."
                    ),
                }),
                "convert_model_dtype": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Cast model weights to bfloat16. Halves VRAM usage for the DiTs "
                        "at a minor quality cost. Only needed on lower-VRAM GPUs."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("ALICE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "Eric/Alice"

    def load(
        self,
        ckpt_dir: str,
        device_id: int,
        offload_model: bool,
        t5_cpu: bool,
        convert_model_dtype: bool,
    ):
        global _pipeline_cache

        ckpt_dir = ckpt_dir.strip()
        if not ckpt_dir:
            raise ValueError("[Eric_AliceLoader] ckpt_dir cannot be empty.")
        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"[Eric_AliceLoader] Directory not found: {ckpt_dir!r}")

        cache_key = (ckpt_dir, device_id, offload_model, t5_cpu, convert_model_dtype)

        if cache_key in _pipeline_cache:
            log.info("[Eric_AliceLoader] Returning cached pipeline (no reload needed).")
            return (_pipeline_cache[cache_key],)

        # Parameters changed - clear old pipeline and free VRAM before loading new one
        if _pipeline_cache:
            log.info("[Eric_AliceLoader] Parameters changed - evicting old pipeline from cache.")
            _pipeline_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info(f"[Eric_AliceLoader] Loading Alice T2V pipeline from: {ckpt_dir!r}")
        print(f"[Eric_AliceLoader] Loading Alice T2V pipeline from: {ckpt_dir}")
        print("[Eric_AliceLoader] This will take a moment (loading T5, two DiT experts, and VAE)...")

        try:
            from alice.pipeline import AliceTextToVideo
        except ImportError as e:
            raise ImportError(
                f"[Eric_AliceLoader] Cannot import alice package: {e}\n"
                "Run setup_vendor.py from the Eric_Alice_t2v node directory first."
            ) from e

        # Fix tokenizer path and download if missing
        _ensure_tokenizer(ckpt_dir)

        # Generate shard index JSON files if missing (diffusers requirement)
        for subfolder in ("low_noise_model", "high_noise_model"):
            model_dir = os.path.join(ckpt_dir, subfolder)
            if os.path.isdir(model_dir):
                _ensure_shard_index(model_dir)
            else:
                raise FileNotFoundError(
                    f"[Eric_AliceLoader] Expected subfolder not found: {model_dir}\n"
                    "Ensure the full checkpoint is downloaded from "
                    "gomirageai/Alice-T2V-14B-MoE on HuggingFace."
                )

        cfg = _build_config(ckpt_dir)

        pipeline = AliceTextToVideo(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=t5_cpu,
            init_on_cpu=True,       # safe default: load to CPU first, move to GPU on demand
            convert_model_dtype=convert_model_dtype,
        )

        # Stash the offload flag so the generator node can read it
        pipeline._eric_offload_model = offload_model

        _pipeline_cache[cache_key] = pipeline
        log.info("[Eric_AliceLoader] Pipeline loaded and cached successfully.")
        print("[Eric_AliceLoader] Pipeline ready.")
        return (pipeline,)
