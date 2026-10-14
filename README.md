# torch-tbccl

[![docs](https://github.com/tensorsofthewall/torch-tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/torch-tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/torch-tbccl?include_prereleases&label=pre-release)](https://github.com/tensorsofthewall/torch-tbccl/releases)

torch-tbccl is a PyTorch `torch.distributed` backend named `"tbccl"` that lets PyTorch programs communicate between heterogeneous machines, for example a Linux host with an NVIDIA GPU and a Mac, over TCP, including a direct Thunderbolt 4 link. PyTorch itself is not modified: you `import torch_tbccl` and use the normal `torch.distributed` API.

> **Status:** **0.2.0** (first release; experimental). Validated tuple: **PyTorch 2.13.x**, **CPython 3.13**, TBCCL **C ABI 1 / wire protocol 4**, on Linux x86-64 (CPU and CUDA 13) and macOS arm64 (CPU and MPS). Anything else is untested.

It is one of three independent adapters over the [TBCCL](https://github.com/tensorsofthewall/tbccl) runtime (the others are [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) and [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl)). It is an adapter only: transport, collectives and staging live in TBCCL. It does not aim for NCCL feature parity.

## What is supported

- `send`/`recv`, `isend`/`irecv`, `broadcast`, `all_gather`, `barrier` and `SUM` `all_reduce` over CPU, CUDA and (on macOS) MPS tensors, at world sizes 1 to 4.
- Real `DistributedDataParallel` training (world sizes 2 to 4).
- Point-to-point and collective calls in flight together on one process group.
- Heterogeneous groups, such as CUDA on Linux with MPS on a Mac.

The complete list, with dtype, world-size and device limits, and what is rejected, is in [supported operations](docs/reference/supported-operations.md).

## Quick start

Install the CUDA 13 build of PyTorch 2.13 first, then the release-candidate wheel (it contains TBCCL; nothing else to install):

```sh
uv venv --python 3.13 .venv && . .venv/bin/activate
# Linux (x86-64, CUDA 13 build of PyTorch; a CPU-only PyTorch does not work with the Linux wheel):
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --no-deps https://github.com/tensorsofthewall/torch-tbccl/releases/download/v0.2.0/torch_tbccl-0.2.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# macOS (Apple silicon): uv pip install torch==2.13.0, then the macosx_14_0_arm64 wheel from the same release page
python -m torch_tbccl.info
```

Details, other platforms and building from source: [installing](docs/getting-started/install.md).

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

Two hosts, the PyTorch/TBCCL relationship and DDP: [quickstart](docs/getting-started/quickstart.md). To confirm your job really runs on TBCCL, see [how do I know TBCCL is being used?](docs/guides/verify-active.md).

## Supported configurations

| | Validated |
|---|---|
| PyTorch | 2.13.x (CUDA 13 build on Linux, the macOS wheel on a Mac) |
| Python | 3.13 |
| TBCCL | C ABI 1, wire protocol 4 (bundled in the wheel) |
| Platforms | Linux x86-64 (CPU, CUDA 13), macOS arm64 (CPU, MPS) |

See the [compatibility page](docs/reference/compatibility.md), the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html) and the [validated configurations](docs/project/validated-configurations.md).

## Documentation

Hosted documentation: https://torch-tbccl.tensorsofthewall.com/ (`latest` is development documentation built from `main`; release-candidate versions appear as numbered versions; `stable` does not exist until the final release). Build it locally with `make docs`. Pages to start with: [installing](docs/getting-started/install.md), [quickstart](docs/getting-started/quickstart.md), [troubleshooting](docs/guides/troubleshooting.md), [known limitations](docs/project/limitations.md).

## Known limitations

Reductions are `SUM` only; float16/bfloat16 reductions only at world size 2; `gather` only for two-rank groups; no `reduce`, `scatter`, `reduce_scatter`, `all_to_all`; no FSDP validation; no recovery after an abort; one submitting thread per MPS rank. See [known limitations](docs/project/limitations.md).

## Development

`CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base; the [development documentation](docs/development/index.md) covers building, testing and where a change belongs. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

torch-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.
