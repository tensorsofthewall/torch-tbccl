"""One rank of an N-rank group: init, barrier, tear down."""
import sys
from datetime import timedelta

import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=30))
dist.barrier()
rank = dist.get_rank()
dist.destroy_process_group()
print(f"rank {rank} ok")
sys.exit(0)
