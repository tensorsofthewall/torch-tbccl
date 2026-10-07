# torch-tbccl

torch-tbccl is an out-of-tree PyTorch `torch.distributed` backend named `"tbccl"`. It adapts PyTorch's c10d process groups onto an installed [TBCCL](https://github.com/tensorsofthewall/tbccl) runtime, so PyTorch programs can communicate between heterogeneous machines (for example a Linux host with an NVIDIA GPU and a Mac) over TCP, including a direct Thunderbolt 4 link. It is an adapter: all transport, algorithm and staging logic lives in TBCCL. It does not aim for NCCL feature parity.

> **Status:** development version 0.2.0.dev0, experimental, no release published. Validated tuple: **PyTorch 2.13.x**, **CPython 3.13**, TBCCL **C ABI 1 / wire protocol 4**, on Linux x86-64 (CPU and CUDA 13) and macOS arm64 (CPU and MPS). Anything else is untested.

## What you can use it for

- `send`/`recv`, `broadcast`, `all_gather`, `barrier` and `SUM` `all_reduce` over CPU, CUDA and (on macOS) MPS tensors, at world sizes 1 to 4.
- Real `DistributedDataParallel` training (world sizes 2 to 4).
- Point-to-point and collective calls in flight together on one process group.
- Heterogeneous groups, such as CUDA on Linux with MPS on a Mac.

The complete list, and what is rejected, is in [supported operations](docs/reference/supported-operations.md).

## Install

libtbccl is a build-time input (an installed TBCCL prefix) linked statically into the extension:

```sh
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install torch==2.13.0 setuptools wheel build
TBCCL_ROOT=/path/to/tbccl-install python -m build --wheel --no-isolation -o dist .
uv pip install dist/torch_tbccl-*.whl
```

Details for CUDA and macOS builds: [installing](docs/getting-started/install.md). Never use `--reinstall` on an environment that holds torch.

## Minimal example

```python
import torch
import torch.distributed as dist
import torch_tbccl                      # registers the "tbccl" backend

dist.init_process_group("tbccl")
x = torch.tensor([dist.get_rank() + 1.0])
dist.all_reduce(x)                      # SUM
print(x)                                # tensor([3.]) on both ranks of a 2-rank group
dist.destroy_process_group()
```

Each rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>` (use `<host>:0` for free ports), separate from PyTorch's rendezvous:

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --standalone --nproc-per-node 2 your_script.py
```

## Supported configurations

| | Validated |
|---|---|
| PyTorch | 2.13.x |
| Python | 3.13 |
| TBCCL | C ABI 1, wire protocol 4 |
| Platforms | Linux x86-64 (CPU, CUDA 13), macOS arm64 (CPU, MPS) |

See [compatibility](docs/reference/compatibility.md) and the draft [validation](docs/validation/0.2.0.md).

## Documentation

The documentation is in `docs/` and builds with `make docs`: [getting started](docs/getting-started/index.md), [guides](docs/guides/index.md), [concepts](docs/concepts/index.md), [reference](docs/reference/index.md). Contributing: `CONTRIBUTING.md` and `AGENTS.md`.

## License

torch-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.

Security: see `SECURITY.md` and the security model in the documentation. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.
