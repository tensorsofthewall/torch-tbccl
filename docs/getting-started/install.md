# Installing torch-tbccl

torch-tbccl is built from source against an **installed** [TBCCL](https://github.com/tensorsofthewall/tbccl) prefix and distributed as a platform wheel. libtbccl is a build-time input and is linked statically into the extension, so the wheel needs no TBCCL at run time. There is no published binary release yet.

## Validated tuple

| | |
|---|---|
| torch-tbccl | 0.2.0.dev0 (development) |
| PyTorch | 2.13.x (validated with 2.13.0) |
| CPython | 3.13 |
| TBCCL | C ABI 1, wire protocol 4 |
| Platforms | Linux x86-64 (CPU and CUDA 13) and macOS arm64 (CPU and MPS) |

Anything else is untested. See [Compatibility](../reference/compatibility.md).

## Build and install a wheel

```sh
# 1. an environment with the matching torch (the build imports it)
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install torch==2.13.0 setuptools wheel build
# Linux with CUDA: add --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# 2. build the wheel against the TBCCL prefix and install it
TBCCL_ROOT=/path/to/tbccl-install python -m build --wheel --no-isolation -o dist .
uv pip install dist/torch_tbccl-*.whl
```

| Build | Needs |
|---|---|
| Linux x86-64 with CUDA | a TBCCL prefix that includes `libtbccl_cuda.a` and a CUDA-enabled torch (its pip `nvidia/` runtime is used) |
| Linux host-only | a prefix without the CUDA component, or a CPU-only torch: CPU tensors only |
| macOS arm64 | TBCCL's host-only prefix and a torch built with MPS (the PyTorch macOS wheel): CPU and MPS tensors; needs the Xcode command-line tools |

The TBCCL static libraries must be built position-independent (TBCCL does this by default). The build refuses a prefix with an unsupported C ABI.

```{important}
Never use `--reinstall` on an environment that holds torch: it rewrites torch. The compiled extension is tied to torch's minor series, and `import torch_tbccl` refuses another series with a rebuild hint (`TORCH_TBCCL_ALLOW_TORCH_MISMATCH=1` overrides the check at your own risk).
```

For development, install editable with `TBCCL_ROOT=... uv pip install --no-build-isolation --no-deps -e .`.

## Check the installation

```sh
python -m torch_tbccl.info
```

prints the package, torch, libtbccl, C ABI and wire protocol versions and the devices this build supports.

## After upgrading TBCCL

TBCCL's wire protocol must match across ranks and across everything linked against it. After installing a TBCCL with a different wire protocol version, rebuild torch-tbccl against the new prefix ([TBCCL versioning](https://tbccl.tensorsofthewall.com/en/stable/reference/versioning.html)).
