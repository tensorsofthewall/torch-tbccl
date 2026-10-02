# torch-tbccl

> **Experimental. Phase 44 complete (persistent CUDA staging in TBCCL); Phase 43: world_size=2; Float32 SUM AllReduce, byte-generic Broadcast/AllGather; experimental DDP.**
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
- Done: CPU Float32 SUM `all_reduce` (one dense contiguous tensor, in place,
  no copies) with real asynchronous `Work` (`async_op=True` returns after
  submission): `is_completed()`, `wait()` (repeatable; `wait(timeout)` is a
  bounded adapter-side wait that does not cancel the TBCCL operation),
  `get_future()` (completed by one persistent completion thread per group),
  tensors retained until the operation finishes.
- Done: CUDA tensors (Linux build with TBCCL's `tbccl_cuda` component):
  a CUDA tensor maps to a `Cuda` BufferView and PyTorch's *current* stream at
  submission is passed to TBCCL as the producer stream, so no explicit
  synchronization is needed between producing a tensor and `all_reduce`.
  CUDA and CPU ranks can be mixed. The first CUDA collective in a process
  pays one-time TBCCL setup.
- Done (Phase 43): `broadcast` (any dense contiguous dtype, one tensor) and `allgather` (one input, `world_size`
  outputs of equal size) on CPU and CUDA tensors, same `Work`/`Future` semantics as `all_reduce`.
- Done (Phase 43, **experimental DDP**): `torch.nn.parallel.DistributedDataParallel` works with world_size=2,
  Float32 gradient AllReduce, CPU<->CPU and Linux CUDA<->Mac CPU (both rank orders), including buffer sync and
  gradient bucketing. Not supported: `find_unused_parameters=True`, other dtypes/ops for reductions, MPS, N>2,
  fault recovery (no abort/cancel: a silent peer can block). This is not NCCL feature parity.
- Phase 44 (runtime-side, no adapter change): TBCCL >= `c7cac28` keeps CUDA pinned staging persistent per communicator, so CUDA collective submission
  no longer allocates payload-sized pinned memory (warm submit ~10-20 us at any size). Retained pinned memory = the largest transfer the communicator has seen.
- Opt-in diagnostics: `TORCH_TBCCL_TRACE=1` (per-collective timeline via `torch_tbccl.trace_events()`).
- Not yet: other collectives (`barrier`, `reduce`, `all_gather_into_tensor`, ...), MPS.
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
| Devices | cpu, cuda (needs `tbccl_cuda` in the TBCCL prefix and a CUDA torch) | cpu |

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
World size 2 only; reductions are Float32 SUM only; experimental DDP only (no FSDP); no MPS; no fault recovery.
See `docs/architecture.md` and `docs/pytorch_api_audit.md`.

Phase 42/43 reports and results: `docs/phase42_report.md`, `docs/phase42_results.md`, `docs/phase43_report.md`,
`docs/phase43_results.md`, `docs/phase43_ddp_api_audit.md`, `docs/phase44_report.md`, `docs/phase44_results.md`.
Two-host examples: `examples/cross_host_allreduce.py`, `cross_host_collectives.py`, `real_link_overlap.py`, `ddp_train.py`.
