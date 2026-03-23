"""
setup_vendor.py - One-time setup: vendors the alice package into Eric_Alice_t2v/vendor/

This copies only the alice Python source (not model weights) into the vendor/
subdirectory. No packages are installed into your ComfyUI environment, so there
are zero dependency conflicts with transformers, numpy, etc.

Usage
-----
Option A: Point at a local clone of https://github.com/mirage-video/Alice
    (fastest - no network required)

    python setup_vendor.py --alice-src "C:/path/to/Alice"

Option B: Auto-clone from GitHub
    (requires git on PATH)

    python setup_vendor.py

Verification
------------
After running, you should see:
    Copying ...alice/ -> ...vendor/alice
    ✓ Vendor install verified - alice imports OK.

Re-run any time to update the vendored source from a newer clone.
"""
import argparse
import os
import shutil
import sys

# vendor/ lives alongside this script
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
ALICE_PKG = "alice"  # the Python package subfolder inside the GitHub repo root


def _copy_alice(alice_repo_root: str) -> None:
    """Copy alice/ from a local repo root into vendor/alice/."""
    src = os.path.join(alice_repo_root, ALICE_PKG)
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Expected to find an '{ALICE_PKG}' subfolder inside:\n  {alice_repo_root}\n"
            "Make sure --alice-src points at the repository root "
            "(the folder that contains alice/, scripts/, configs/, etc.)"
        )
    dst = os.path.join(VENDOR_DIR, ALICE_PKG)
    if os.path.exists(dst):
        print(f"Removing existing vendor/{ALICE_PKG}/ ...")
        shutil.rmtree(dst)
    print(f"Copying:\n  {src}\n  → {dst}")
    shutil.copytree(src, dst)
    print("Copy complete.")


def _clone_and_copy() -> None:
    """Git-clone the Alice repo into a temp dir and copy alice/ from it."""
    import subprocess
    import tempfile

    repo_url = "https://github.com/mirage-video/Alice.git"
    print(f"Cloning {repo_url} (shallow) ...")
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo_url, tmp],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed:\n{result.stderr}\n"
                "Make sure git is installed and you have internet access, "
                "or use --alice-src to point at a local clone."
            )
        _copy_alice(tmp)


def _verify() -> None:
    """Quick smoke-test: confirm alice can be imported from vendor/."""
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    try:
        from alice.pipeline import AliceTextToVideo   # noqa: F401
        from alice.configs import ALICE_CONFIGS       # noqa: F401
        print("✓ Vendor install verified - alice imports OK.")
    except ImportError as exc:
        print(f"✗ Import check FAILED: {exc}")
        print("  Check that the vendor/alice/ directory contains __init__.py and pipeline/.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vendor the alice package into Eric_Alice_t2v/vendor/"
    )
    parser.add_argument(
        "--alice-src",
        metavar="PATH",
        default=None,
        help=(
            "Path to a local clone of https://github.com/mirage-video/Alice. "
            "If omitted, the repo is cloned automatically (requires git)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(VENDOR_DIR, exist_ok=True)
    print(f"Vendor directory: {VENDOR_DIR}")

    if args.alice_src:
        _copy_alice(os.path.abspath(args.alice_src))
    else:
        _clone_and_copy()

    _verify()
    print("\nSetup complete. You can now load Eric_Alice_t2v nodes in ComfyUI.")


if __name__ == "__main__":
    main()
