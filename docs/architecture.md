# Architecture

```
PyTorch (torch.distributed, c10d)
   |
   v
torch-tbccl            adapter only
   |
   v
installed libtbccl     transport / collectives / device providers
```

Dependencies flow one way. libtbccl never depends on PyTorch.

torch-tbccl translates:

| PyTorch                     | TBCCL                          |
|-----------------------------|--------------------------------|
| `c10d::Store`               | `tbccl::CommunicatorOptions`   |
| rank / world_size           | `tbccl::Communicator`          |
| `at::Tensor`                | `tbccl::BufferView`            |
| torch dtype                 | `tbccl::DataType`              |
| c10d `ReduceOp`             | `tbccl::ReduceOp`              |
| current CUDA stream         | `tbccl::ExecutionContext`      |
| `tbccl::Work`               | `c10d::Work`                   |

No collective algorithm, TCP code, chunk scheduling, CUDA staging or local
reduction lives here. If one seems necessary, identify the missing libtbccl
API instead.

## Build rule
The extension links only against an **installed** TBCCL (`TBCCL_ROOT`):
`include/tbccl/*.hpp` plus the installed library. No TBCCL source-tree
include paths, no `add_subdirectory`, no TBCCL `.cpp` files compiled in.

The installed static library must be built position-independent
(`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`), since it is linked into a
shared Python extension.

## Collectives and the DDP path (Phase 43)
| c10d call | role in DDP | TBCCL call |
|---|---|---|
| `allgather` (1 input, `world_size` outputs) | constructor: parameter-count verification (int64[1]) | `Communicator::all_gather` |
| `broadcast` (1 tensor, `rootTensor == 0`) | constructor: shape metadata and `_sync_module_states` (coalesced params + buffers); per-forward buffer sync; bucket-index sync after iteration 0 | `Communicator::broadcast` |
| `allreduce` (SUM, Float32, 1 tensor) | gradient buckets (one call per bucket, Work Future consumed by the Reducer) | `Communicator::all_reduce` |

All three share one FIFO executor in libtbccl, so cross-collective ordering is the issue order on each rank. The
adapter's Work/Future layer (`WorkTBCCL` + one `CompletionWorker` per group) is identical for the three ops; it retains
tensors until the TBCCL operation completes. Broadcast/AllGather are byte-generic (validation of layout/device only).
The exact call path and payload sizes are in `docs/phase43_ddp_api_audit.md`.

## Threads, the GIL and Work lifetime (Phase 71)
Each group has one `CompletionWorker` thread that waits for submitted operations in order and completes their Futures; it never touches the Python API. It also never *destroys* a finished `WorkState`: the tensors it retains may
be the last reference to a live Python object, and dropping it takes the GIL. Finished states are retired to a list that the submitting thread (`reap()`) or the group destructor destroys; the destructor also releases the GIL
while it joins the worker. (Before Phase 71 a process that exited with a live group hung in interpreter finalization.)

## Collectives and point-to-point (Phase 71)
libtbccl runs collectives and P2P as independent ordering domains over one connection per peer. They must not be in flight at the same time on one group (overlap corrupts data silently), so `ProcessGroupTBCCL` tracks the in-flight Works of each
domain under its submission mutex and refuses a submission that would overlap the other domain (`NotImplementedError`). Collectives overlap freely with collectives, P2P with P2P.

## Rendezvous keys (Phase 71)
`torch_tbccl/v3/joined` (atomic join counter) and `torch_tbccl/v3/g<generation>/{endpoint/<rank>,communicator_id}`, `generation = (joins-1)/world_size`: successive groups created over the same persistent store never see each other's records.

## Diagnostics
`TORCH_TBCCL_TRACE=1` records per collective (in memory): sequence, op, bytes, device, entry, before-submit,
TBCCL-return, completion, and wait entry/exit stamps (this process's monotonic clock; never compare across machines).
`TORCH_TBCCL_FORCE_SYNC_ALLREDUCE=1` makes allreduce wait for completion before returning (diagnostic baseline only).

## Deferred
- TBCCL's C ABI (not needed by a C++ extension; useful for Rust/Swift/MLX later).
- other collectives (reduce, scatter, reduce_scatter, all_gather_into_tensor, all_to_all, non-SUM reductions), FSDP, MPS, recovery after an abort. (N>2, barrier, `find_unused_parameters` and abort are done: `docs/phase71_capability_audit.md`.)
- Pooled pinned staging in libtbccl's CUDA provider (per-collective `cudaMallocHost` is the size-proportional submit cost).
