"""One rank of an N-rank (3 or 4) world: barrier, broadcast from every root, all_gather, all_reduce, a send/recv ring, several process groups,
and the N>2 low-precision rejection. DEVICE_RANK=<r> puts that one rank's tensors on CUDA, the others stay on CPU (the single GPU is never shared)."""
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()
cuda_rank = os.environ.get("DEVICE_RANK")
dev = torch.device("cuda" if cuda_rank is not None and int(cuda_rank) == rank else "cpu")


def to(t):
    return t.to(dev)


# barrier (several, back to back)
for _ in range(3):
    dist.barrier()

# broadcast from every root
for root in range(world):
    t = to(torch.full((1000,), float(root * 10 + 1)) if rank == root else torch.zeros(1000))
    dist.broadcast(t, src=root)
    assert torch.equal(t.cpu(), torch.full((1000,), float(root * 10 + 1))), (root, t[:3])

# all_gather, rank order
mine = to(torch.arange(5, dtype=torch.float32) + 100 * rank)
outs = [to(torch.zeros(5)) for _ in range(world)]
dist.all_gather(outs, mine)
for r in range(world):
    assert torch.equal(outs[r].cpu(), torch.arange(5, dtype=torch.float32) + 100 * r), (r, outs[r])

# all_reduce, exact values, several dtypes and sizes
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    for n in (1, 777, 100000):
        x = to((torch.arange(n) % 13 + rank + 1).to(dtype))
        want = sum((torch.arange(n) % 13 + r + 1).to(dtype) for r in range(world))
        dist.all_reduce(x)
        assert torch.equal(x.cpu(), want), (dtype, n)

# send/recv ring (asynchronous: every rank sends to the next while receiving from the previous)
nxt, prv = (rank + 1) % world, (rank - 1) % world
out = to(torch.full((1 << 20,), float(rank + 1)))
inn = to(torch.zeros(1 << 20))
reqs = [dist.isend(out, nxt), dist.irecv(inn, prv)]
for r in reqs:
    r.wait()
assert torch.equal(inn.cpu(), torch.full((1 << 20,), float(prv + 1)))

# several process groups at once (each has its own Store namespace, ports and communicator id)
g1 = dist.new_group(list(range(world)))
g2 = dist.new_group(list(range(world)))
a = to(torch.ones(8) * (rank + 1))
b = to(torch.ones(8) * 10 * (rank + 1))
dist.all_reduce(a, group=g1)
dist.all_reduce(b, group=g2)
total = world * (world + 1) // 2
assert a.cpu()[0] == total and b.cpu()[0] == 10 * total
dist.all_reduce(torch.ones(2, device=dev))  # the default group still works alongside

# N>2 low precision is rejected explicitly (nothing is communicated, the group stays usable)
for dt in (torch.float16, torch.bfloat16):
    try:
        dist.all_reduce(to(torch.ones(8, dtype=dt)))
    except (NotImplementedError, RuntimeError) as e:
        assert "N>2 reduction semantics are not defined" in str(e), str(e)
    else:
        raise AssertionError(f"{dt} all_reduce across {world} ranks must be rejected")
dist.barrier()

dist.destroy_process_group()
print(f"rank {rank} ok")
sys.exit(0)
