"""One rank of a tiny two-host N-rank (3 or 4) probe of TBCCL's optimized collective algorithms: a CORRECTNESS check, not a benchmark.

Run it once per forced algorithm combination (the SAME TBCCL_*_ALGORITHM variables on every rank; differing overrides are a protocol mismatch),
e.g. TBCCL_ALLREDUCE_ALGORITHM=ring. Payloads are at most 1 MiB and about a dozen operations per rank, because sustained TB4 traffic has raised the root port's
correctable Replay Timer Timeout counter before. Every check is exact (integer-valued Float32 sums). Launch one process per rank, like examples/nrank_link.py.
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
dist.barrier()
log("barrier x2 ok")

for root, n in ((0, 262144), (world - 1, 1024)):  # 1 MiB from rank 0, 4 KiB from the last rank
    t = torch.full((n,), float(root + 1)) if rank == root else torch.zeros(n)
    dist.broadcast(t, src=root)
    assert torch.equal(t, torch.full((n,), float(root + 1))), f"broadcast root {root}"
log("broadcast 1 MiB (root 0) and 4 KiB (last rank) ok")

n = 16384  # 64 KiB per rank
outs = [torch.zeros(n) for _ in range(world)]
dist.all_gather(outs, torch.full((n,), float(10 * rank + 1)))
for r in range(world):
    assert torch.equal(outs[r], torch.full((n,), float(10 * r + 1))), f"all_gather slot {r}"
log("all_gather 64 KiB per rank ok")

want = float(world * (world + 1) // 2)
for n in (1024, 262144):  # 4 KiB and 1 MiB of Float32
    for _ in range(2):
        x = torch.full((n,), float(rank + 1))
        dist.all_reduce(x)
        assert torch.equal(x, torch.full((n,), want)), f"all_reduce n={n}"
log("all_reduce Float32 4 KiB and 1 MiB x2 ok")

dist.barrier()
dist.destroy_process_group()
log("done")
sys.exit(0)
