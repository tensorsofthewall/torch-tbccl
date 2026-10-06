"""init_process_group -> collective -> destroy_process_group, repeated CYCLES times in ONE process over ONE persistent store (the torchrun situation: the
launcher owns the store, so every cycle reuses the same key prefix). A later cycle must never read an earlier cycle's rendezvous records."""
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), world, is_master=rank == 0, timeout=timedelta(seconds=60))
for cycle in range(int(os.environ.get("CYCLES", "5"))):
    dist.init_process_group("tbccl", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=30))
    x = torch.full((4,), float(rank + 1 + cycle))
    dist.all_reduce(x)
    assert x[0].item() == sum(r + 1 + cycle for r in range(world)), (cycle, x)
    dist.destroy_process_group()
print(f"rank {rank} ok")
sys.stdout.flush()
os._exit(0)  # rank 0 owns the store server; leave only after everyone is done is the caller's business
