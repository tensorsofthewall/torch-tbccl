# Supported operations

This is a tested contract: the matrix is produced by `tools/capability_matrix.py` and enforced by `tests/test_capability_matrix.py`. Anything outside it is rejected with a clear `NotImplementedError` or `ValueError` before anything is communicated.

| Operation | Support |
|---|---|
| `init_process_group`, `new_group`, `destroy_process_group`, `torchrun` | world sizes 1 to 4 validated (5 to 8 accepted, untested); every group has its own communicator and ports |
| `send`, `recv`, `isend`, `irecv`, `batch_isend_irecv` | any dense contiguous dtype (as bytes: bool, int8 to int64, fp16, bf16, fp32, fp64, complex64, fp8), world sizes 2 to 4; matched FIFO per peer, tags ignored |
| `broadcast`, `all_gather` | any dense contiguous dtype, world size 2 and above, one tensor, equal sizes |
| `all_reduce` | **`SUM` only**: int8, uint8, int32, int64, float32, float64; float16 and bfloat16 **at world size 2 only**; one tensor; async supported |
| `gather` | two-rank groups only (so `gather_object` too); `all_gather_object` and `broadcast_object_list` work at world size 2 and above |
| `barrier` | world sizes 1 to 4 |
| abort and failure | `_abort_process_group()` aborts the communicator; a peer exit fails the survivors within milliseconds; destroy with pending work aborts it; no recovery |
| DDP | world sizes 2 to 4 on CPU, with CUDA and CPU ranks, with `find_unused_parameters`, any bucket size; **not at world size 1** |

## Devices

| Device | Support |
|---|---|
| CPU | yes |
| CUDA | yes (Linux); uses PyTorch's current stream at submission; mixed CUDA and CPU ranks work |
| MPS (macOS) | send/recv, isend/irecv, broadcast, all_gather, two-rank gather, barrier, `SUM` all_reduce; see [MPS](../guides/mps.md) |

## Rejected

`PRODUCT`, `MIN`, `MAX`, `AVG` and other reductions; dtypes without reduction arithmetic (int16, bool, complex, fp8); `reduce`, `scatter`, `reduce_scatter(_tensor)`, `all_gather_into_tensor`, `all_to_all(_single)`, `scatter_object_list`; non-contiguous tensors (no hidden copy is ever made); sparse tensors; collectives on a one-rank group.

## Not supported

FSDP (not validated) and fault recovery.
