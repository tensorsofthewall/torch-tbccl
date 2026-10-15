# Installing torch-tbccl

torch-tbccl is distributed as a platform wheel that contains its own statically linked TBCCL runtime. Install the matching PyTorch first, then the wheel. You do not build or install TBCCL yourself to use the wheel; [building from source](#build-from-source) is for developers.

## Requirements

| | Linux x86-64 | macOS arm64 (Apple silicon) |
|---|---|---|
| Python | **3.13** (the wheel is `cp313`; other versions are not supported) | **3.13** |
| PyTorch | **2.13.x, the CUDA 13 build** (`+cu130`) | **2.13.x** from PyPI (the macOS wheel, built with MPS) |
| Devices | CPU and CUDA | CPU and MPS |
| OS | glibc 2.27 or newer (`manylinux_2_27`/`manylinux_2_28` wheel) | macOS 14 or newer (`macosx_14_0_arm64` wheel) |

```{warning}
**On Linux the wheel needs the CUDA build of PyTorch**, because the extension is linked against `libtorch_cuda.so` and `libc10_cuda.so`. A CPU-only PyTorch for Linux does not provide them and `import torch_tbccl` fails with an `ImportError`. You do not need a GPU in the machine to import it, but the PyTorch build must be the CUDA 13 one. Validated: PyTorch 2.13.0 (`+cu130`). Mixing in another CUDA series of PyTorch is not validated.
```

The compiled extension is tied to the torch minor series it was built with: `import torch_tbccl` refuses another series with a rebuild hint (`TORCH_TBCCL_ALLOW_TORCH_MISMATCH=1` overrides the check at your own risk).

## Before you start: uv and Python

The commands below use [uv](https://docs.astral.sh/uv/) to create the environment. If `uv --version` fails, install it (macOS and Linux): `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal. You do not need to install Python 3.13 yourself: `uv venv --python 3.13` downloads a suitable interpreter if the machine has none. Run the commands in the directory where you want the environment (`.venv` is created there); use the same layout on both machines of a two-machine job.

## Install the wheel

The 0.2.0 wheels are attached to the GitHub release [`v0.2.0`](https://github.com/tensorsofthewall/torch-tbccl/releases/tag/v0.2.0). They are not published on PyPI or TestPyPI. Verify the download against `SHA256SUMS` from the same release if you like.

Every command names the environment (`.venv/bin/python`), so nothing needs to be activated, and several environments can coexist on one machine without ambiguity.

**Linux x86-64:**

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "torch==2.13.0" --torch-backend cu130
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/torch-tbccl/releases/download/v0.2.0/torch_tbccl-0.2.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

`--torch-backend cu130` makes `uv pip` take PyTorch from the CUDA 13 index (it resolves `torch==2.13.0+cu130`). It is a preview feature of uv; if your uv rejects it, use the explicit index, which is equivalent:

```sh
uv pip install --python .venv/bin/python "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

**macOS arm64:**

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "torch==2.13.0"
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/torch-tbccl/releases/download/v0.2.0/torch_tbccl-0.2.0-cp313-cp313-macosx_14_0_arm64.whl
```

`--no-deps` keeps the package manager from replacing the PyTorch you just chose, so install PyTorch first. Plain `pip` works the same way (`.venv/bin/python -m pip install --no-deps <wheel URL>`). Never use `--reinstall` on an environment that holds torch: it rewrites torch.

## Verify the download (optional)

`SHA256SUMS` in the release lists the checksum of each wheel. To check a wheel before installing it, download both files into one directory and compare:

```sh
curl -fsSLO https://github.com/tensorsofthewall/torch-tbccl/releases/download/v0.2.0/SHA256SUMS
curl -fsSLO <the wheel URL for your machine from above>
sha256sum -c --ignore-missing SHA256SUMS      # Linux; on macOS: shasum -a 256 -c --ignore-missing SHA256SUMS
uv pip install --python .venv/bin/python --no-deps ./<the wheel file name>
```

`<wheel>: OK` means the file matches the release. (The `--ignore-missing` flag skips the entries for the other platform's wheel.)

## Check the installation

```sh
.venv/bin/python -m torch_tbccl.info
```

prints the package, torch, libtbccl, C ABI and wire protocol versions and the devices this build supports, for example:

```text
torch_tbccl        0.2.0
torch_running      2.13.0+cu130
libtbccl           0.6.0
c_abi              1
wire_protocol      4
devices            ['cpu', 'cuda']
backend            tbccl
registered         True
```

`registered True` means `"tbccl"` is a known `torch.distributed` backend in this process. Every machine of a job must report the same wire protocol.

## Build from source

Building is only needed on a platform or PyTorch version without a wheel, or to change the code. It is separate from using the wheel: libtbccl is then a build-time input, an **installed** [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) prefix, linked statically into the extension.

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "torch==2.13.0" setuptools wheel build        # Linux CUDA: add --torch-backend cu130
TBCCL_ROOT=/path/to/tbccl-install .venv/bin/python -m build --wheel --no-isolation -o dist .
uv pip install --python .venv/bin/python --no-deps dist/torch_tbccl-*.whl
```

| Build | Needs |
|---|---|
| Linux x86-64 with CUDA | a TBCCL prefix that includes `libtbccl_cuda.a` (the CUDA archive) and a CUDA-enabled torch (its pip `nvidia/` runtime is used) |
| Linux host-only | a prefix without the CUDA component, or a CPU-only torch: CPU tensors only |
| macOS arm64 | TBCCL's macOS prefix and the PyTorch macOS wheel (built with MPS); the Xcode command-line tools |

The TBCCL static libraries must be position-independent (TBCCL's default). The build refuses a prefix with an unsupported C ABI. For development install editable with `TBCCL_ROOT=... uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .`.

## After upgrading TBCCL

TBCCL's wire protocol must match across ranks. Because the wheel bundles TBCCL, upgrading means installing a newer torch-tbccl wheel on **every** machine of the job. Ranks built against different wire protocols cannot connect ([TBCCL versioning](https://tbccl.tensorsofthewall.com/en/latest/reference/versioning.html)).
