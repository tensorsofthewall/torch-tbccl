# Using Apple MPS tensors

On macOS, torch-tbccl lets PyTorch tensors on the `mps` device take part in `torch.distributed` operations over the `tbccl` backend.

## What works

Send/recv, isend/irecv, broadcast, all_gather, two-rank gather, barrier and `SUM` all_reduce on MPS tensors; DDP with an MPS rank next to a CPU or CUDA rank. For reductions, float32, float16, bfloat16, int32, int64, int8 and uint8 are supported; any dtype PyTorch can create on MPS is moved as bytes by send, recv, broadcast and all_gather. PyTorch cannot create float64 or float8 tensors on MPS at all.

## How it works

PyTorch 2.13 registers no c10d kernels for the MPS dispatch key, so `import torch_tbccl` registers the supported operators for MPS out of tree. The registration is gated on the torch version and the operator schemas and never duplicates an upstream registration. An MPS tensor's `data_ptr()` is a Metal object handle plus a byte offset, not a CPU address; the adapter takes the shared `MTLBuffer`'s CPU-visible contents pointer plus the storage offset and hands it to TBCCL as a `MetalShared` buffer (or as `Host` for reduction types that `MetalShared` does not reduce). Before TBCCL touches the bytes, the adapter runs a device-wide MPS synchronize.

## Limits

- Only shared-storage MPS buffers are supported; operators outside the list above fail with PyTorch's own dispatcher message, not a `torch-tbccl:` one.
- Every MPS operation first runs a device-wide synchronize; no overlap with MPS compute is claimed.
- **Single submitting thread.** PyTorch's `torch.mps.synchronize()` is not safe to call from two threads while one of them computes on MPS (a Metal assertion fires; this reproduces without TBCCL). Submit TBCCL operations on MPS tensors from one thread at a time. A background thread doing point-to-point side traffic alongside an MPS rank should use CPU tensors.
- World sizes above 2 with MPS, and MPS-to-MPS between two processes on one Mac, have not been validated.
- This applies to torch 2.13 only and to macOS.
