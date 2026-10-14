# Quickstart

A two-rank all_reduce on one machine, then the same across two hosts. Install first ([installing](install.md)).

## Two ranks on one machine

`allreduce.py`:

```python
import torch
import torch.distributed as dist
import torch_tbccl                      # registers the "tbccl" backend (required once per process)

dist.init_process_group("tbccl")

x = torch.tensor([dist.get_rank() + 1.0])
dist.all_reduce(x)                      # SUM
print(dist.get_rank(), dist.get_backend(), x)

dist.destroy_process_group()
```

```sh
OMP_NUM_THREADS=2 TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 .venv/bin/torchrun --standalone --nproc-per-node 2 allreduce.py
```

Each rank prints its rank, the backend name `tbccl` and `tensor([3.])`; the two lines may interleave. Set `OMP_NUM_THREADS` (for example to 2) when several ranks share one machine, or they starve each other.

Every rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>`: the address where this rank listens for TBCCL connections. It is **separate** from PyTorch's rendezvous store. `<host>:0` picks free ports, which is what you normally want; many groups can coexist in one process. If it is unset in a group of two or more ranks, `init_process_group` raises `ValueError: torch-tbccl: invalid argument: TBCCL_LOCAL_ENDPOINT is not set`. If the machine's hostname does not resolve, add `--local-addr 127.0.0.1` to `torchrun` for single-machine runs.

## Across two hosts

Use PyTorch's normal rendezvous and give each host its own reachable address in `TBCCL_LOCAL_ENDPOINT` (for example the address of a direct Thunderbolt link). Install the same torch-tbccl version on both hosts:

```sh
# host A (rank 0)
TBCCL_LOCAL_ENDPOINT=<A address>:0 .venv/bin/torchrun --nnodes 2 --nproc-per-node 1 --node-rank 0 --master-addr <A address> --master-port 29500 allreduce.py
# host B (rank 1)
TBCCL_LOCAL_ENDPOINT=<B address>:0 .venv/bin/torchrun --nnodes 2 --nproc-per-node 1 --node-rank 1 --master-addr <A address> --master-port 29500 allreduce.py
```

Rank 0's host is `--master-addr`. On a Mac, run the rank in a live terminal or SSH session and allow the Python process through the macOS firewall prompt the first time; a background job without a session cannot reach the network ([troubleshooting](../guides/troubleshooting.md)).

## Devices

Put tensors on `cpu`, `cuda` (Linux) or `mps` (macOS) as usual. A group may mix them, for example a CUDA rank with an MPS rank. See [supported operations](../reference/supported-operations.md) and the [MPS guide](../guides/mps.md).

## DDP

`examples/ddp_mlp.py` is a small distributed-data-parallel training run on synthetic data, checked against a single-process reference:

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 .venv/bin/torchrun --standalone --nproc-per-node 2 examples/ddp_mlp.py
```

See the [DDP guide](../guides/ddp.md). To confirm a job really uses TBCCL see [how do I know TBCCL is being used?](../guides/verify-active.md).
