# ADR 0002: MPS support through out-of-tree c10d kernel registration

- Status: accepted
- Date: 2026-10-06

## Context

PyTorch 2.13 registers no c10d kernels for the MPS dispatch key, so `dist.all_reduce` on an MPS tensor fails in the dispatcher before reaching any backend (`'c10d::allreduce_' is not currently implemented for the MPS device`). PyTorch must not be patched.

## Decision

`import torch_tbccl` registers the seven supported operators (`send`, `recv_`, `broadcast_`, `allreduce_`, `allgather_`, `gather_`, `barrier`) for the MPS key out of tree. The registration is gated on the torch version and verified against the operator schemas, and it is skipped if an upstream kernel already exists. An MPS tensor's `data_ptr()` is a Metal object handle plus a byte offset, not a CPU address; the adapter converts it (shared `MTLBuffer` contents plus storage offset, after a device-wide MPS synchronize) into a TBCCL `MetalShared` buffer, or `Host` for reduction types `MetalShared` lacks. libtbccl is unchanged.

## Consequences

- No PyTorch source change and no TBCCL change are needed for MPS.
- Operators outside the seven fail with PyTorch's own dispatcher message.
- TBCCL operations on MPS tensors must be submitted from one thread at a time, because PyTorch's own `torch.mps.synchronize()` is not safe to call concurrently with MPS compute from another thread.
