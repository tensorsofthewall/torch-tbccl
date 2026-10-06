"""The packaging and capability-audit work asynchronous Work semantics, one scenario per TEST_MODE (2 ranks, optionally CUDA on rank 0 via DEVICE_RANK0=cuda).

  p2p_outstanding N        N isend/irecv in flight in both directions at once, waited in reverse order, every payload checked (FIFO association)
  allreduce_outstanding N  N small async all_reduces in flight, waited in a scrambled order, every result checked
  async_collectives        async_op=True for broadcast / all_gather / all_reduce / isend+irecv: returns before the (late) peer joins, wait() then correct
  growth                   repeated batches of outstanding Works: fds, threads and RSS stay flat
"""
import gc
import os
import random
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
N = int(os.environ.get("N", "8"))
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
peer = 1 - rank
dev = torch.device("cuda" if os.environ.get("DEVICE_RANK0") == "cuda" and rank == 0 else "cpu")


def payload(sender, k, n):
    return (((torch.arange(n, dtype=torch.int64) * 3 + sender * 1000 + k * 17) % 251)).to(torch.float32)


def sizes(k):
    return 4 + 37 * k + (k % 5) * 1024  # distinct, mostly small, a few KiB


def nfds():
    return len(os.listdir("/proc/self/fd"))


def nthreads():
    return len(os.listdir("/proc/self/task"))


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def p2p_batch(n):
    sends, recvs, bufs = [], [], []
    for k in range(n):
        sends.append(dist.isend(payload(rank, k, sizes(k)).to(dev), dst=peer))
    for k in range(n):
        b = torch.zeros(sizes(k), device=dev)
        bufs.append(b)
        recvs.append(dist.irecv(b, src=peer))
    for w in reversed(recvs + sends):
        w.wait()
    for k, b in enumerate(bufs):
        assert torch.equal(b.cpu(), payload(peer, k, sizes(k))), f"recv {k} holds the wrong message"


def allreduce_batch(n):
    xs = [payload(rank, k, 64 + k).to(dev) for k in range(n)]
    works = [dist.all_reduce(x, async_op=True) for x in xs]
    order = list(range(n))
    random.Random(5 + rank).shuffle(order)  # the two ranks wait in different orders
    for k in order:
        works[k].wait()
    for k, x in enumerate(xs):
        want = payload(0, k, 64 + k) + payload(1, k, 64 + k)
        assert torch.equal(x.cpu(), want), f"all_reduce {k} holds the wrong result"


if mode == "p2p_outstanding":
    p2p_batch(N)

elif mode == "allreduce_outstanding":
    allreduce_batch(N)

elif mode == "async_collectives":
    # Collectives and P2P are independent ordering domains over one peer lane (TBCCL's contract): they must not be in flight at the same time on one group,
    # so the three collectives are pending together first, then (after they complete) the point-to-point pair.
    LATE = 1.0
    n = 1 << 20
    if rank == 1:
        time.sleep(LATE)  # rank 0's calls below are in flight before rank 1 participates
    t0 = time.monotonic()
    x = payload(rank, 0, n).to(dev)
    w_ar = dist.all_reduce(x, async_op=True)
    b = (payload(0, 1, n) if rank == 0 else torch.zeros(n)).to(dev)
    w_bc = dist.broadcast(b, src=0, async_op=True)
    outs = [torch.zeros(n, device=dev) for _ in range(2)]
    w_ag = dist.all_gather(outs, payload(rank, 2, n).to(dev), async_op=True)
    submit = time.monotonic() - t0
    works = {"all_reduce": w_ar, "broadcast": w_bc, "all_gather": w_ag}
    if rank == 0:
        assert submit < 0.6, f"submitting three async collectives blocked for {submit:.2f}s"
        for name, w in works.items():
            assert not w.is_completed(), f"{name} completed before the peer joined"
    for name, w in works.items():
        assert w.wait() is True, name
        assert w.is_completed() and w.is_success(), name
    assert torch.equal(x.cpu(), payload(0, 0, n) + payload(1, 0, n))
    assert torch.equal(b.cpu(), payload(0, 1, n))
    assert all(torch.equal(outs[r].cpu(), payload(r, 2, n)) for r in range(2))

    if rank == 1:
        time.sleep(LATE)
    big = payload(rank, 3, 1 << 24).to(dev)  # 64 MiB: cannot complete into a peer that has not posted its receive
    rb = torch.zeros(1 << 24, device=dev)
    t1 = time.monotonic()
    w_s, w_r = dist.isend(big, dst=peer), dist.irecv(rb, src=peer)
    submit_p2p = time.monotonic() - t1
    if rank == 0:
        assert submit_p2p < 0.6, f"submitting isend/irecv blocked for {submit_p2p:.2f}s"
        assert not w_s.is_completed() and not w_r.is_completed(), "p2p completed before the peer posted"
    for w in (w_s, w_r):
        assert w.wait() is True and w.is_success()
    assert torch.equal(rb.cpu(), payload(peer, 3, 1 << 24))
    if rank == 0:
        print(f"async submit: 3 collectives in {submit * 1e3:.1f} ms, isend+irecv in {submit_p2p * 1e3:.1f} ms, all pending", flush=True)

elif mode == "growth":
    for _ in range(3):  # warm up allocators, the completion thread and TBCCL's staging
        p2p_batch(32)
        allreduce_batch(32)
    gc.collect()
    f0, t0, r0 = nfds(), nthreads(), rss_mb()
    for _ in range(25):
        p2p_batch(32)
        allreduce_batch(32)
    gc.collect()
    f1, t1, r1 = nfds(), nthreads(), rss_mb()
    print(f"rank {rank} growth: fds {f0}->{f1} threads {t0}->{t1} rss {r0:.1f}->{r1:.1f} MB", flush=True)
    assert f1 == f0 and t1 == t0, (f0, f1, t0, t1)
    assert r1 - r0 < 40, f"RSS grew by {r1 - r0:.1f} MB over 25 batches of 64 operations"

else:
    raise SystemExit(f"unknown mode {mode}")

dist.barrier()
t0 = time.monotonic()
dist.destroy_process_group()
assert time.monotonic() - t0 < 10, "shutdown was slow"
print(f"rank {rank} ok")
sys.exit(0)
