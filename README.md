# torch-tbccl

> **Experimental. N=2, Float32 SUM AllReduce only.**
> Not NCCL-feature-parity.

An out-of-tree PyTorch distributed backend (`"tbccl"`) that adapts
`torch.distributed` onto an **installed** [TBCCL](../tbccl) runtime. It is
an adapter: all transport and collective logic stays in libtbccl.

```
PyTorch -> torch-tbccl -> installed libtbccl -> TBCCL transport/collectives
```

## Status
- Done: package builds against an installed TBCCL; `import torch_tbccl`
  registers the `"tbccl"` backend (idempotent); `init_process_group("tbccl")`
  rendezvouses through the c10d Store and creates a TBCCL `Communicator`.
- Done: blocking CPU Float32 SUM `all_reduce` (one dense contiguous tensor,
  in place, no copies). `async_op=True` is accepted but currently also
  blocks; true asynchronous `Work` is the next step.
- Not yet: asynchronous Work, CUDA tensors, other collectives.
- Rejected with a clear error: other dtypes/ops, non-contiguous, sparse,
  multiple tensors. Zero-element tensors are a no-op on both ranks.

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

Each rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>` for its TBCCL data
endpoint (separate from PyTorch's rendezvous store; give each rank a
distinct endpoint, including on one machine). Endpoints are exchanged
through the Store under `torch_tbccl/v1/endpoint/<rank>`; the PyTorch
timeout bounds the exchange and TBCCL connection setup.

## Known limitations
World size 2 only; Float32 SUM AllReduce only; no DDP/FSDP; no MPS.
See `docs/architecture.md` and `docs/pytorch_api_audit.md`.
