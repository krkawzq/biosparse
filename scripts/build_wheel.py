#!/usr/bin/env python3
"""Build platform-specific wheel with pre-compiled Rust library.

This script builds a wheel with the native library already in place.
The library must be downloaded/built before running this script.

Usage:
    python scripts/build_wheel.py --plat win_amd64
    python scripts/build_wheel.py --plat manylinux_2_17_x86_64
    python scripts/build_wheel.py --plat macosx_11_0_arm64
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Supported platforms and their library names
PLATFORM_LIBS = {
    "win_amd64": "biosparse.dll",
    "win32": "biosparse.dll",
    "macosx_10_9_x86_64": "libbiosparse.dylib",
    "macosx_11_0_arm64": "libbiosparse.dylib",
    "macosx_10_9_universal2": "libbiosparse.dylib",
    "manylinux_2_17_x86_64": "libbiosparse.so",
    "manylinux_2_17_aarch64": "libbiosparse.so",
    "linux_x86_64": "libbiosparse.so",
    "linux_aarch64": "libbiosparse.so",
}


def get_version():
    """Extract version from pyproject.toml."""
    pyproject = ROOT / "pyproject.toml"
    content = pyproject.read_text()
    for line in content.splitlines():
        if line.strip().startswith("version"):
            # version = "0.1.0"
            return line.split("=")[1].strip().strip('"').strip("'")
    raise ValueError("Version not found in pyproject.toml")


def verify_library(plat: str) -> Path:
    """Verify that the native library exists for the platform."""
    lib_name = PLATFORM_LIBS.get(plat)
    if not lib_name:
        raise ValueError(f"Unknown platform: {plat}. Supported: {list(PLATFORM_LIBS.keys())}")
    
    lib_path = ROOT / "src" / "biosparse" / "_binding" / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Native library not found at {lib_path}\n"
            f"Please build or download the library first:\n"
            f"  cargo build --release"
        )
    
    print(f"[OK] Found native library: {lib_path}")
    return lib_path


def build_wheel(plat: str):
    """Build wheel for the specified platform."""
    version = get_version()
    lib_path = verify_library(plat)
    
    print(f"\n[BUILD] Building wheel for {plat}")
    print(f"   Version: {version}")
    print(f"   Library: {lib_path}")
    
    # Create dist directory
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(exist_ok=True)
    
    # Build wheel using pip wheel
    # We use a specific approach: build with bdist_wheel and rename
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Run bdist_wheel
        subprocess.run(
            [
                sys.executable, "-m", "build",
                "--wheel",
                "--outdir", str(tmpdir),
            ],
            cwd=ROOT,
            check=True,
        )
        
        # Find the built wheel
        wheels = list(tmpdir.glob("*.whl"))
        if not wheels:
            raise RuntimeError("No wheel was built")
        
        src_wheel = wheels[0]
        
        # Parse wheel name: biosparse-0.1.0-py3-none-any.whl
        # We want: biosparse-0.1.0-py3-none-{plat}.whl
        parts = src_wheel.name.split("-")
        # ['biosparse', '0.1.0', 'py3', 'none', 'any.whl']
        
        # For binary wheels with native code, we should use:
        # - py3 (Python 3 only)
        # - abi3 or none (ABI tag)
        # - platform tag
        
        # Replace the platform tag
        parts[-1] = f"{plat}.whl"
        new_name = "-".join(parts)
        
        dst_wheel = dist_dir / new_name
        
        # Copy and rename
        shutil.copy2(src_wheel, dst_wheel)
        
        print(f"\n[OK] Built wheel: {dst_wheel}")
        print(f"  Size: {dst_wheel.stat().st_size / 1024:.1f} KB")
    
    return dst_wheel


def main():
    parser = argparse.ArgumentParser(description="Build platform-specific wheel")
    parser.add_argument(
        "--plat",
        required=True,
        choices=list(PLATFORM_LIBS.keys()),
        help="Platform tag for the wheel",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify library exists, don't build",
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            verify_library(args.plat)
        else:
            build_wheel(args.plat)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
