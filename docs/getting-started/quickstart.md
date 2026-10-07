# Quickstart

```python
import torch
import torch.distributed as dist
import torch_tbccl                      # registers the "tbccl" backend (required once per process)

dist.init_process_group("tbccl")

x = torch.tensor([dist.get_rank() + 1.0])
dist.all_reduce(x)                      # SUM
print(x)                                # tensor([3.]) on both ranks of a 2-rank group

dist.destroy_process_group()
```

Every rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>`: the address where this rank listens for TBCCL connections. It is **separate** from PyTorch's rendezvous store. `<host>:0` picks free ports, which is what you normally want; many groups can coexist in one process. With `torchrun` on one machine:

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --standalone --nproc-per-node 2 your_script.py
```

If the machine's hostname does not resolve, add `--local-addr 127.0.0.1` for single-machine runs.

Across two hosts use PyTorch's normal rendezvous and give each host its own reachable address in `TBCCL_LOCAL_ENDPOINT` (for example the address of a direct Thunderbolt link):

```sh
# host A (rank 0)
TBCCL_LOCAL_ENDPOINT=<A address>:0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank 0 --master-addr <A address> --master-port 29500 train.py
# host B (rank 1)
TBCCL_LOCAL_ENDPOINT=<B address>:0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank 1 --master-addr <A address> --master-port 29500 train.py
```

`examples/ddp_mlp.py` is a small distributed-data-parallel training run on synthetic data, checked against a single-process reference. See the [supported operations](../reference/supported-operations.md) for what a ProcessGroup can do.
