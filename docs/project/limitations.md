# Known limitations

The same categories are used in the other TBCCL projects' limitation pages. Core transport limits (TCP only, no direct CUDA-to-Metal DMA, no RDMA, no encryption) are in the [TBCCL limitations](https://tbccl.tensorsofthewall.com/en/latest/project/limitations.html).

## Unsupported

| Limitation | Detail |
|---|---|
| Reductions | `SUM` only. No `PRODUCT`, `MIN`, `MAX`, `AVG` |
| Operations | no `reduce`, `scatter`, `reduce_scatter`, `all_gather_into_tensor`, `all_to_all`, `scatter_object_list`; `gather` only for two-rank groups |
| Float16 / BFloat16 reductions | world size 2 only |
| Non-contiguous or sparse tensors | rejected; the adapter never makes a hidden copy |
| Recovery | none. After an abort or a peer failure the process group must be destroyed and recreated; retrying over the same store is not supported |
| FSDP | not validated |
| Python and PyTorch | CPython 3.13 and PyTorch 2.13.x only |
| Linux with a CPU-only PyTorch | the Linux wheel needs the CUDA 13 build of PyTorch |
| Platforms | Linux x86-64 and macOS arm64 only |

## Supported but not validated

| Item | State |
|---|---|
| World sizes 5 to 8 | accepted, not validated (1 to 4 are) |
| MPS with world sizes above 2, and MPS-to-MPS between two processes on one Mac | not validated |
| CUDA series other than 13 | not validated |

## Known upstream issue

| Issue | Detail |
|---|---|
| PyTorch MPS thread safety | PyTorch's `torch.mps.synchronize()` is not safe to call from two threads while one computes on MPS (a Metal assertion fires; it reproduces without TBCCL). Submit TBCCL operations on MPS tensors from one thread, and use CPU tensors for background side traffic |
| No c10d kernels for MPS in PyTorch 2.13 | `import torch_tbccl` registers the supported operators for MPS itself, gated on the torch version and operator schemas |

## Performance

- Every MPS operation first runs a device-wide synchronize; no overlap with MPS compute is claimed.
- CUDA tensors are staged through pinned host memory and travel over TCP; there is no direct GPU-to-GPU path.
- Timings include the completion-thread handoff; `TORCH_TBCCL_TRACE=1` shows where time goes per collective.

## Operational workarounds

| Situation | Workaround |
|---|---|
| a Mac rank hangs at start | answer the macOS firewall / Local Network prompt from a local session and keep the rank in a live terminal or SSH session ([troubleshooting](../guides/troubleshooting.md)) |
| installing the wheel replaced my PyTorch | install with `--no-deps` and never use `--reinstall` |
| several ranks on one machine are slow | set `OMP_NUM_THREADS` (for example 2) |
