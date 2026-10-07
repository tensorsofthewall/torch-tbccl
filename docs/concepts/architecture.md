# Architecture

```
PyTorch (torch.distributed, c10d)
   |
torch-tbccl            adapter only
   |
installed libtbccl     transport, collectives, device providers
```

Dependencies flow one way: libtbccl never depends on PyTorch. torch-tbccl is an adapter and nothing more; no communication algorithm, TCP code, chunk scheduling, CUDA staging or local reduction lives here. If one seems necessary, the missing generic API belongs in TBCCL ({doc}`../adr/0001-adapter-only`).

## What is translated

| PyTorch | TBCCL |
|---|---|
| `c10d::Store` | the endpoint exchange that builds `tbccl::CommunicatorOptions` ({doc}`rendezvous`) |
| rank and world size | `tbccl::Communicator` |
| `at::Tensor` | `tbccl::BufferView` |
| torch dtype | `tbccl::DataType` |
| c10d `ReduceOp` | `tbccl::ReduceOp` |
| current CUDA stream | `tbccl::ExecutionContext` |
| `tbccl::Work` | `c10d::Work` |

## Build rule

The extension links only against an **installed** TBCCL (`TBCCL_ROOT`): the installed headers and static libraries. It never includes TBCCL source-tree paths or compiles TBCCL sources, and it is never patched around: a generic TBCCL defect is fixed in TBCCL.

## Collectives and the DDP path

DDP needs `allgather`, `broadcast` and `allreduce`; all three share one ordered executor in libtbccl, so cross-collective ordering is the issue order on each rank. The adapter's `Work` and `Future` layer (`WorkTBCCL` plus one completion worker per group) is identical for all operations and retains tensors until the TBCCL operation completes. Broadcast and all_gather are byte-generic: only layout and device are validated.

## Diagnostics

`TORCH_TBCCL_TRACE=1` records, per collective and in memory, the sequence, operation, bytes, device and the entry, submit, TBCCL-return, completion and wait timestamps of this process (its monotonic clock; never compare across machines); read them with `torch_tbccl.trace_events()`. `TORCH_TBCCL_FORCE_SYNC_ALLREDUCE=1` makes all_reduce wait for completion before returning and is a diagnostic baseline only.

## Not provided

Reductions other than `SUM`, `reduce`, `scatter`, `reduce_scatter`, `all_gather_into_tensor`, `all_to_all`, FSDP, and recovery after an abort. See the [supported operations](../reference/supported-operations.md).
