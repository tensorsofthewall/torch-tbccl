"""new_group over a subset of the world (W3/W4): every rank must call new_group for every group (c10d rule); members run collectives and P2P on it, non-members
are not involved, and the default (world) group keeps working alongside."""
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()
subsets = [[0, world - 1], [1, 2], list(range(1, world))]
groups = [(ranks, dist.new_group(ranks)) for ranks in subsets]
for ranks, g in groups:
    if rank not in ranks:
        continue
    k = len(ranks)
    x = torch.full((64,), float(rank + 1))
    if k == 2 or k == 3:
        dist.all_reduce(x, group=g)
        assert x[0].item() == sum(r + 1 for r in ranks), (ranks, x[0])
    root = ranks[-1]
    b = torch.full((9,), 7.0 if rank == root else 0.0)
    dist.broadcast(b, src=root, group=g)
    assert b[0].item() == 7.0
    gl = [torch.zeros(3) for _ in ranks]
    dist.all_gather(gl, torch.full((3,), float(rank)), group=g)
    assert [t[0].item() for t in gl] == [float(r) for r in ranks]
    if k == 2:  # P2P between group members (peer addressed by GLOBAL rank, as c10d does)
        a, bb = ranks
        if rank == a:
            dist.send(torch.arange(5.0), dst=bb, group=g)
        else:
            y = torch.zeros(5)
            dist.recv(y, src=a, group=g)
            assert torch.equal(y, torch.arange(5.0))
    dist.barrier(group=g)
y = torch.full((4,), float(rank + 1))
dist.all_reduce(y)  # the world group is still fine
assert y[0].item() == sum(r + 1 for r in range(world))
dist.barrier()
dist.destroy_process_group()
print(f"rank {rank} ok")
sys.exit(0)
