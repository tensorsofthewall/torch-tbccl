"""One rank of an N-rank (3 or 4) torch-tbccl group spanning two hosts: a SMALL correctness probe, not a benchmark.

The N-rank runtime work limits real-link traffic to a few tiny exchanges (payloads <= 64 KiB, a handful of operations) because sustained TB4 traffic has raised the
root port's correctable Replay Timer Timeout counter. Every check is exact (integer-valued Float32 sums). Launch one process per rank:

  RANK=<r> WORLD_SIZE=<n> MASTER_ADDR=<rank-0 host> MASTER_PORT=<p> TBCCL_LOCAL_ENDPOINT=<this host's TB4 ip>:0 python examples/nrank_link.py
"""
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()
log = lambda *a: print(f"rank {rank}:", *a, flush=True)  # noqa: E731

dist.barrier()
for root in range(world):
    t = torch.full((1024,), float(root + 1)) if rank == root else torch.zeros(1024)  # 4 KiB
    dist.broadcast(t, src=root)
    assert torch.equal(t, torch.full((1024,), float(root + 1))), f"broadcast root {root}"
log("barrier + broadcast from every root ok")

outs = [torch.zeros(1024) for _ in range(world)]
dist.all_gather(outs, torch.full((1024,), float(10 * rank + 1)))
for r in range(world):
    assert torch.equal(outs[r], torch.full((1024,), float(10 * r + 1))), f"all_gather slot {r}"
log("all_gather ok")

for n in (256, 16384):  # 1 KiB and 64 KiB of Float32
    for _ in range(3):
        x = torch.full((n,), float(rank + 1))
        dist.all_reduce(x)
        assert torch.equal(x, torch.full((n,), float(world * (world + 1) // 2))), f"all_reduce n={n}"
log("all_reduce Float32 1 KiB and 64 KiB x3 ok")

nxt, prv = (rank + 1) % world, (rank - 1) % world
out, inn = torch.full((1024,), float(rank + 1)), torch.zeros(1024)  # 4 KiB ring
for r in (dist.isend(out, nxt), dist.irecv(inn, prv)):
    r.wait()
assert torch.equal(inn, torch.full((1024,), float(prv + 1))), "ring"
log("send/recv ring ok")

dist.barrier()
dist.destroy_process_group()
log("done")
sys.exit(0)
