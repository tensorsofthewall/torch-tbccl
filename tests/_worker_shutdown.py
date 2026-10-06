"""init, one collective and one point-to-point operation, then return from the main module: interpreter shutdown with a live process group, an un-waited
completed Work and (DEVICE=cuda) CUDA tensors. Nothing is torn down explicitly."""
import os

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl")
rank = dist.get_rank()
dev = torch.device("cuda" if os.environ.get("DEVICE") == "cuda" and rank == 0 else "cpu")
x = torch.ones(1024, device=dev)
dist.all_reduce(x)
assert x[0].item() == 2.0
pending = dist.all_reduce(torch.ones(16, device=dev), async_op=True)  # never waited on explicitly; the barrier below completes after it
dist.barrier()
if rank == 0:
    dist.send(torch.arange(8.0, device=dev), dst=1)
else:
    y = torch.zeros(8)
    dist.recv(y, src=0)
    assert y[7].item() == 7.0
print(f"rank {rank} exiting normally", flush=True)
