"""
Eric_Alice_t2v - ComfyUI nodes for Alice T2V 14B MoE
Author: Eric Rollei
https://github.com/mirage-video/Alice
"""
import sys
import os

# Add vendor directory to Python path so alice can be imported without installing it
_vendor_dir = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_vendor_dir) and _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

try:
    from .nodes.loader import Eric_AliceLoader
    from .nodes.generator import Eric_AliceT2V

    NODE_CLASS_MAPPINGS = {
        "Eric_AliceLoader": Eric_AliceLoader,
        "Eric_AliceT2V": Eric_AliceT2V,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "Eric_AliceLoader": "Alice T2V Loader (Eric)",
        "Eric_AliceT2V": "Alice T2V Generator (Eric)",
    }

    print("[Eric_Alice_t2v] Nodes loaded successfully.")

except ImportError as e:
    print(f"[Eric_Alice_t2v] WARNING: Could not import nodes: {e}")
    print("[Eric_Alice_t2v] Run setup_vendor.py to install the alice package into vendor/.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
