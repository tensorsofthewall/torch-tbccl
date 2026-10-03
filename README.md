# torch-tbccl

> **Experimental. Phase 50: world_size 1-4 (validated; N>2 uses TBCCL's reference collectives); SUM AllReduce (float16/bfloat16 at world_size 2 only), byte-generic Broadcast/AllGather/send/recv, barrier; experimental DDP.**
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
- Phase 45: `ProcessGroup` abort (`torch.distributed.distributed_c10d._abort_process_group()`) maps to TBCCL's communicator-wide abort: outstanding Works/Futures fail, later collectives raise,
  and `destroy_process_group()` no longer hangs on a silent peer. `Work.wait(timeout)` stays non-destructive. No recovery/reconnect.
- Phase 46 (vLLM PP=2 consumer, see ../vllm-tbccl): `send`/`recv` (byte-generic, one tensor, FIFO matched), `gather` and `barrier` on 2-rank groups, one-rank groups, int32/int64/float64 SUM,
  and `TBCCL_LOCAL_ENDPOINT=<host>:0` (a free port pair per communicator, so many groups can coexist in one process).
- Opt-in diagnostics: `TORCH_TBCCL_TRACE=1` (per-collective timeline via `torch_tbccl.trace_events()`).
- Not yet: other collectives (`barrier`, `reduce`, `all_gather_into_tensor`, ...), MPS.
- Phase 49 (needs TBCCL with the Phase 49 datatypes): `all_reduce` SUM also for float16, bfloat16, int8 and uint8 (16-bit floats: widen to float32, add, round once to nearest even; int8/uint8 wrap modulo 256), on CPU and CUDA tensors. `send`/`recv`, `broadcast` and `all_gather` stay byte-generic for any dense dtype: FP8 (`float8_e4m3fn`, `float8_e5m2`, ...) and packed 4-bit payloads cross bit-exactly (`tests/test_byte_transport.py`); they have no reduction, so `all_reduce` of such dtypes and non-SUM ops are rejected before any communication, with a message naming the dtype and op.
- Rejected with a clear error: dtypes without reduction arithmetic, non-SUM ops, non-contiguous, sparse,
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

Each rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>`: `<host>` is where this rank listens for TBCCL connections (separate from PyTorch's rendezvous
store). `<port>` is its control port and its data port is `<port> + 1000`; `<host>:0` asks the kernel for two free ports (recommended: many groups can coexist in one
process, and one machine can host any number of ranks). Whatever was bound, each rank publishes its ACTUAL control and data endpoint through the group's Store under
`torch_tbccl/v2/endpoint/<rank>`, rank 0 publishes one shared communicator id under `torch_tbccl/v2/communicator_id`, and TBCCL itself never sees the Store. Only ranks
that accept connections (every rank but the last) bind listeners. The PyTorch timeout bounds the exchange and TBCCL connection setup.

**torch version.** The compiled extension links against torch's C++ ABI, which is stable only within one minor series. `import torch_tbccl` therefore compares the torch it was
built against (`torch_tbccl.built_with_torch()`) with the running torch and fails with a rebuild hint on a minor-series mismatch (override: `TORCH_TBCCL_ALLOW_TORCH_MISMATCH=1`);
`pyproject.toml` bounds the dependency to the tested series (`torch_tbccl.TESTED_TORCH_SERIES`). Rebuild per environment with `uv pip install --no-build-isolation --no-deps -e .`
(never `--reinstall`, which rewrites torch itself).

## Known limitations
World size 1-4 validated (full mesh, unoptimized reference collectives for N>2; float16/bfloat16 reductions only at world size 2; `gather` is 2-rank only); reductions are SUM only (float16/bfloat16/float32/float64/int8/uint8/int32/int64); experimental DDP only (no FSDP); no MPS; no fault recovery (abort only).
See `docs/architecture.md` and `docs/pytorch_api_audit.md`.

Phase 42/43 reports and results: `docs/phase42_report.md`, `docs/phase42_results.md`, `docs/phase43_report.md`,
`docs/phase43_results.md`, `docs/phase43_ddp_api_audit.md`, `docs/phase44_report.md`, `docs/phase44_results.md`.
Two-host examples: `examples/cross_host_allreduce.py`, `cross_host_collectives.py`, `real_link_overlap.py`, `ddp_train.py`.
