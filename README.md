# torch-tbccl

[![docs](https://github.com/tensorsofthewall/torch-tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/torch-tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/torch-tbccl?label=release)](https://github.com/tensorsofthewall/torch-tbccl/releases)

torch-tbccl is a PyTorch `torch.distributed` backend named `"tbccl"` that lets PyTorch programs communicate between heterogeneous machines, for example a Linux host with an NVIDIA GPU and a Mac, over TCP, including a direct Thunderbolt 4 link. PyTorch itself is not modified: you `import torch_tbccl` and use the normal `torch.distributed` API.

> **Status:** **0.2.0** (first release; experimental). Validated tuple: **PyTorch 2.13.x**, **CPython 3.13**, TBCCL **C ABI 1 / wire protocol 4**, on Linux x86-64 (CPU and CUDA 13) and macOS arm64 (CPU and MPS). Anything else is untested.

It is one of three independent adapters over the [TBCCL](https://github.com/tensorsofthewall/tbccl) runtime (the others are [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) and [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl)). It is an adapter only: transport, collectives and staging live in TBCCL. It does not aim for NCCL feature parity. You do not need to install TBCCL first: the wheel contains it.

## Supported configurations

| | Validated |
|---|---|
| PyTorch | 2.13.x (CUDA 13 build on Linux, the macOS wheel on a Mac) |
| Python | 3.13 (the wheels are `cp313`) |
| TBCCL | C ABI 1, wire protocol 4 (bundled in the wheel) |
| Platforms | Linux x86-64 (CPU, CUDA 13), macOS arm64 (CPU, MPS) |

What it covers, by use:

- **One machine:** several ranks of one group on CPU, CUDA or MPS tensors (the quickstart below).
- **Several hosts:** the same calls between machines over TCP, including a direct Thunderbolt 4 link ([quickstart](docs/getting-started/quickstart.md)).
- **CUDA with MPS:** a group may mix a CUDA rank on Linux with an MPS rank on a Mac ([MPS guide](docs/guides/mps.md)).
- **DDP training:** real `DistributedDataParallel` at world sizes 2 to 4, with the validated example `examples/ddp_mlp.py` ([DDP guide](docs/guides/ddp.md)).
- **Operations:** `send`/`recv`, `isend`/`irecv`, `broadcast`, `all_gather`, `barrier` and `SUM` `all_reduce` at world sizes 1 to 4, with point-to-point and collective calls in flight together. FSDP and everything not listed in [supported operations](docs/reference/supported-operations.md) are not validated.

## Installation

Prebuilt wheels: no compiler, no TBCCL install. Commands name the environment explicitly, so nothing needs to be activated.

**Linux x86-64** (needs the CUDA 13 build of PyTorch even on a machine without a GPU; a CPU-only PyTorch does not work with the Linux wheel):

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "torch==2.13.0" --torch-backend cu130
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/torch-tbccl/releases/download/v0.2.0/torch_tbccl-0.2.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

**macOS arm64** (Apple silicon, macOS 14 or newer):

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "torch==2.13.0"
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/torch-tbccl/releases/download/v0.2.0/torch_tbccl-0.2.0-cp313-cp313-macosx_14_0_arm64.whl
```

`--no-deps` keeps the installer from replacing the PyTorch you just installed. `--torch-backend` is a preview feature of `uv pip`; if your uv lacks it, [installing](docs/getting-started/install.md) shows the equivalent explicit PyTorch index. Install the same version on every machine of a job. Details, other platforms, troubleshooting and building from source (developers only): [installing](docs/getting-started/install.md).

## Verify the installation

```sh
.venv/bin/python -m torch_tbccl.info
```

It prints the package, torch, libtbccl, C ABI and wire protocol versions and the supported devices (`torch_tbccl 0.2.0`, `torch_running 2.13.0+cu130` on Linux, `libtbccl 0.6.0`, `c_abi 1`, `wire_protocol 4`).

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
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 .venv/bin/torchrun --standalone --nproc-per-node 2 your_script.py
```

Run it with `.venv/bin/torchrun` (no activation needed). Two hosts, the PyTorch/TBCCL relationship and DDP: [quickstart](docs/getting-started/quickstart.md). To confirm your job really runs on TBCCL, see [how do I know TBCCL is being used?](docs/guides/verify-active.md).

The authoritative version table is the [current ecosystem status](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html) (a living page) and the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html); this repository's [compatibility page](docs/reference/compatibility.md) and [validated configurations](docs/project/validated-configurations.md) record what was run for this release.

## Documentation

Hosted documentation: https://torch-tbccl.tensorsofthewall.com/ (`stable` is the latest final release, numbered versions are earlier releases, and `latest` is development documentation built from `main`). Build it locally with `make docs`. Pages to start with: [installing](docs/getting-started/install.md), [quickstart](docs/getting-started/quickstart.md), [troubleshooting](docs/guides/troubleshooting.md).

## Limitations

Reductions are `SUM` only; float16/bfloat16 reductions only at world size 2; `gather` only for two-rank groups; no `reduce`, `scatter`, `reduce_scatter`, `all_to_all`; no FSDP validation; no recovery after an abort; one submitting thread per MPS rank. See [known limitations](docs/project/limitations.md).

## Development

[MAINTENANCE.md](MAINTENANCE.md) records the maintenance policy and what needs physical requalification. `CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base; the [development documentation](docs/development/index.md) covers building, testing and where a change belongs. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

torch-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.
