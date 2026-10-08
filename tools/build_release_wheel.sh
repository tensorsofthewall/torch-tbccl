#!/usr/bin/env bash
# Build a publishable torch-tbccl wheel against an installed TBCCL prefix.
#   TBCCL_ROOT=<prefix> tools/build_release_wheel.sh <out-dir>
# Linux: run inside a manylinux_2_28 environment with the CUDA 13 toolkit headers and PyTorch 2.13 installed; the wheel is built, then repaired with auditwheel
# (PyTorch's own libraries and the CUDA runtime that PyTorch brings are excluded, never bundled), and the result must be a genuine manylinux wheel.
# macOS: build on arm64 with MACOSX_DEPLOYMENT_TARGET set (the wheel tag follows it); delocate lists the dependencies and only system libraries and PyTorch may appear.
# Either way the wheel is then inspected (tag, licence, version, no private paths, no development files).
set -euo pipefail
OUT=${1:?output directory}
: "${TBCCL_ROOT:?set TBCCL_ROOT to an installed TBCCL prefix}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
RAW=$(mktemp -d "${TMPDIR:-/tmp}/torch-tbccl-wheel.XXXXXX")
trap 'rm -rf "$RAW"' EXIT
python -m build --wheel --no-isolation -o "$RAW" "$HERE"
case "$(uname -s)" in
    Linux)
        auditwheel show "$RAW"/*.whl
        auditwheel repair --plat manylinux_2_28_x86_64 \
            --exclude 'libtorch*.so' --exclude 'libc10*.so' --exclude 'libcudart.so.13' \
            -w "$OUT" "$RAW"/*.whl ;;
    Darwin)
        : "${MACOSX_DEPLOYMENT_TARGET:?set MACOSX_DEPLOYMENT_TARGET (for example 14.0) so the wheel tag is deliberate}"
        delocate-listdeps --all "$RAW"/*.whl
        cp "$RAW"/*.whl "$OUT"/ ;;
    *) echo "unsupported platform" >&2; exit 1 ;;
esac
python "$HERE/tools/package_inspect.py" "$OUT"/*.whl > "$OUT/inspect.json"
echo "release wheel(s) in $OUT:"; ls "$OUT"/*.whl
