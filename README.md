# torch-tbccl

> **Experimental. Phase 42. N=2, Float32 SUM AllReduce only.**
> Not NCCL-feature-parity.

An out-of-tree PyTorch distributed backend (`"tbccl"`) that adapts
`torch.distributed` onto an **installed** [TBCCL](../tbccl) runtime. It is
an adapter: all transport and collective logic stays in libtbccl.

```
PyTorch -> torch-tbccl -> installed libtbccl -> TBCCL transport/collectives
```

## Status
Package skeleton builds and links against installed TBCCL; the backend
registration and AllReduce are being added (see `docs/`).

Target usage:

```python
import torch_tbccl
import torch.distributed as dist

dist.init_process_group("tbccl", ...)
work = dist.all_reduce(tensor, async_op=True)
work.wait()
```

| | Linux | macOS |
|---|---|---|
| Devices | cpu, cuda | cpu |

## Install (development)
Requires an installed TBCCL prefix (built with
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`) and PyTorch in the environment.

```sh
uv venv .venv && source .venv/bin/activate
uv pip install torch pytest numpy setuptools wheel
export TBCCL_ROOT=/path/to/tbccl/install
uv pip install -e . --no-build-isolation
python -c "import torch_tbccl; print(torch_tbccl.__version__, torch_tbccl.runtime_version())"
pytest
```

Each rank will need `TBCCL_LOCAL_ENDPOINT=<ip>:<port>` for its TBCCL data
endpoint (separate from PyTorch's rendezvous store).

## Known limitations
World size 2 only; Float32 SUM AllReduce only; no DDP/FSDP; no MPS.
See `docs/architecture.md` and `docs/pytorch_api_audit.md`.
