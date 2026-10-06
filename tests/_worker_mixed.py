"""Phase 73: collectives and point-to-point operations in flight TOGETHER on one process group, in the same and in OPPOSITE relative order on the ranks (libtbccl 0.5.1 wire 4
treats them as independent ordering domains; this replaces the Phase 71 overlap guard test). Every payload is exact. Rank 0 may hold CUDA or MPS tensors (DEVICE_RANK0).

  W2: all_reduce / broadcast / all_gather / barrier each overlapped with an isend + irecv pair, 4 KiB / 1 MiB / 64 MiB, relative order same, opposite, mirrored
  W3: a P2P ring (isend next, irecv previous) overlapped with an all_reduce, order alternating by rank parity
"""
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=120))
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(os.environ["DEVICE_RANK0"] if os.environ.get("DEVICE_RANK0") in ("cuda", "mps") and rank == 0 else "cpu")
SIZES = [1 << 10, 1 << 18, 1 << 24]  # float32 elements: 4 KiB, 1 MiB, 64 MiB


def pat(sender, n, k=0):
    return ((torch.arange(n, dtype=torch.int64) * 7 + sender * 13 + k * 3) % 101).to(torch.float32)


def wait_all(works):
    for w in works:
        assert w.wait() is True and w.is_success()


def family_op(family, n, k):
    """Returns (post, check): post() starts the collective, check() verifies it after wait."""
    if family == "all_reduce":
        x = (pat(0, n, k) % 5 + rank).to(dev)
        return (lambda: dist.all_reduce(x, async_op=True)), (lambda: torch.equal(x.cpu(), (pat(0, n, k) % 5) * world + sum(range(world))))
    if family == "broadcast":
        b = pat(0, n, k).to(dev) if rank == 0 else torch.zeros(n, device=dev)
        return (lambda: dist.broadcast(b, src=0, async_op=True)), (lambda: torch.equal(b.cpu(), pat(0, n, k)))
    if family == "all_gather":
        outs = [torch.zeros(n, device=dev) for _ in range(world)]
        mine = pat(rank, n, k).to(dev)
        return (lambda: dist.all_gather(outs, mine, async_op=True)), (lambda: all(torch.equal(outs[r].cpu(), pat(r, n, k)) for r in range(world)))
    return (lambda: dist.barrier(async_op=True)), (lambda: True)


def w2_case(family, n, order, k):
    peer = 1 - rank
    mine, theirs = pat(rank, n, k + 9).to(dev), torch.zeros(n, device=dev)
    post_c, check_c = family_op(family, n, k)
    # order: "same" = collective first on both ranks; "opposite" = rank 0 collective first, rank 1 P2P first; "mirror" = the reverse
    coll_first = {"same": True, "opposite": rank == 0, "mirror": rank == 1}[order]
    works = []
    p2p = lambda: works.extend([dist.irecv(theirs, src=peer), dist.isend(mine, dst=peer)])  # noqa: E731
    if coll_first:
        works.append(post_c())
        p2p()
    else:
        p2p()
        works.append(post_c())
    wait_all(works)
    assert check_c(), f"{family} result wrong ({order}, {n} elements)"
    assert theirs.device.type == dev.type
    assert torch.equal(theirs.cpu(), pat(peer, n, k + 9)), f"P2P payload wrong next to {family} ({order}, {n} elements)"


if world == 2:
    k = 0
    for family in ("all_reduce", "broadcast", "all_gather", "barrier"):
        for n in SIZES:
            if family == "barrier" and n != SIZES[0]:
                continue
            for order in ("same", "opposite", "mirror"):
                w2_case(family, n, order, k)
                k += 1
    # a burst: many P2P messages and collectives queued together, all waited at the end
    works, checks, bufs = [], [], []
    for i in range(8):
        post_c, check_c = family_op("all_reduce", 4096 + i, 100 + i)
        mine, theirs = pat(rank, 3000 + i, i).to(dev), torch.zeros(3000 + i, device=dev)
        bufs.append(theirs)
        if (rank + i) % 2 == 0:
            works.append(post_c())
            works.extend([dist.isend(mine, dst=1 - rank), dist.irecv(theirs, src=1 - rank)])
        else:
            works.extend([dist.isend(mine, dst=1 - rank), dist.irecv(theirs, src=1 - rank)])
            works.append(post_c())
        checks.append(check_c)
    wait_all(works)
    assert all(c() for c in checks)
    for i, b in enumerate(bufs):
        assert torch.equal(b.cpu(), pat(1 - rank, 3000 + i, i))
else:
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    for n in SIZES[:2]:
        for flip in (False, True):
            post_c, check_c = family_op("all_reduce", n, 5)
            mine, theirs = pat(rank, n, 3).to(dev), torch.zeros(n, device=dev)
            works = []
            ring = lambda: works.extend([dist.irecv(theirs, src=prv), dist.isend(mine, dst=nxt)])  # noqa: E731
            if ((rank % 2 == 0) == flip):
                works.append(post_c())
                ring()
            else:
                ring()
                works.append(post_c())
            wait_all(works)
            assert check_c() and torch.equal(theirs.cpu(), pat(prv, n, 3))

dist.barrier()
t0 = time.monotonic()
dist.destroy_process_group()
assert time.monotonic() - t0 < 10, "shutdown was slow"
print(f"rank {rank} ok")
sys.exit(0)
