"""One rank: init_process_group("tbccl"), check basic properties, tear down."""
import sys
from datetime import timedelta

import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=30))
assert dist.get_backend() == "tbccl", dist.get_backend()
assert dist.get_world_size() == 2
rank = dist.get_rank()
dist.destroy_process_group()
print(f"rank {rank} ok")
sys.exit(0)
