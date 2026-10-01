"""Blocking Float32 SUM all_reduce on CPU. Run one process per rank, e.g. on one machine:

  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 TBCCL_LOCAL_ENDPOINT=127.0.0.1:29600 python examples/allreduce.py &
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 TBCCL_LOCAL_ENDPOINT=127.0.0.1:29601 python examples/allreduce.py
"""
import torch
import torch.distributed as dist

import torch_tbccl

dist.init_process_group("tbccl")
rank = dist.get_rank()
x = torch.tensor([1.0, 2.0, 3.0]) if rank == 0 else torch.tensor([4.0, 5.0, 6.0])
dist.all_reduce(x)
print(f"rank {rank} (tbccl runtime {torch_tbccl.runtime_version()}): {x.tolist()}")  # [5.0, 7.0, 9.0]
dist.destroy_process_group()
