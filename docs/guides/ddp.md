# Distributed data parallel

`torch.nn.parallel.DistributedDataParallel` works over the `tbccl` backend at world sizes 2 to 4, on CPU, with CUDA and CPU ranks together, with `find_unused_parameters`, and with any bucket size. It does **not** work at world size 1, because collectives on a one-rank group are rejected.

DDP uses three c10d operations, all supported:

| c10d call | Role in DDP |
|---|---|
| `allgather` (one input, `world_size` outputs) | parameter-count verification at construction |
| `broadcast` (one tensor, root 0) | shape metadata and initial parameter and buffer synchronization; per-forward buffer sync |
| `allreduce` (`SUM`, Float32, one tensor) | one call per gradient bucket; the returned future is consumed by DDP's reducer |

All three go through one ordered executor in TBCCL, so the order of collectives is the issue order on each rank.

## Mixing point-to-point traffic with DDP

An application may exchange point-to-point messages on the same process group while DDP runs, from another thread, in any relative order on each rank. The only constraints are the usual ones: every rank issues collectives in the same order and P2P messages match in FIFO order per peer. `examples/ddp_mlp.py --side-p2p N` demonstrates this. On an MPS rank, use CPU tensors for such side traffic (see [MPS](mps.md)).

## Example

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --standalone --nproc-per-node 2 examples/ddp_mlp.py
```
