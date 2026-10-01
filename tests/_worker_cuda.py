"""One rank of a CUDA scenario. CUDA_RANKS lists ranks whose tensors live on the GPU; the
others use CPU, so "0" is the Linux-CUDA <-> Mac-CPU shape and "0,1" puts both on one GPU."""
import gc
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
cuda_ranks = {int(r) for r in os.environ["CUDA_RANKS"].split(",")}
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
on_gpu = rank in cuda_ranks
dev = torch.device("cuda", 0) if on_gpu else torch.device("cpu")
CYCLES = 300_000_000  # ~0.2 s of GPU time


def vals(r, n, k=0):
    return (torch.arange(n, dtype=torch.float32) % 100) + r * 1000 + k


def expected(n, k=0):
    return vals(0, n, k) + vals(1, n, k)


def cpu(t):
    return t.cpu()


def warmup():
    # TBCCL's first CUDA collective pays lazy setup (staging allocation, stream creation); without
    # this the producer delay below could elapse before TBCCL even starts, hiding a missing dependency.
    w = torch.ones(1 << 18, device=dev)
    dist.all_reduce(w)
    dist.all_reduce(w)
    if on_gpu:
        torch.cuda.synchronize()


if mode == "basic":
    for n in (1, 1001, 1 << 10, 1 << 18, 4 << 20):
        for async_op in (False, True):
            x = vals(rank, n).to(dev)
            work = dist.all_reduce(x, async_op=async_op)
            if async_op:
                work.wait()
            assert torch.equal(cpu(x), expected(n)), (n, async_op)
    z = torch.empty(0, device=dev)
    dist.all_reduce(z)

elif mode == "stream":
    warmup()
    # Producer is delayed on a non-default stream; all_reduce is submitted inside that stream
    # with NO synchronization. TBCCL must wait on the stream via its event dependency.
    n = 1 << 18
    s = torch.cuda.Stream() if on_gpu else None
    x = torch.full((n,), -1.0, device=dev)
    if on_gpu:
        src = vals(rank, n).to(dev)  # produced on-device so the stream op below is a pure kernel
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            torch.cuda._sleep(CYCLES)
            x.copy_(src)
            work = dist.all_reduce(x, async_op=True)
    else:
        x = vals(rank, n)
        work = dist.all_reduce(x, async_op=True)
    work.wait()
    assert torch.equal(cpu(x), expected(n)), "stale data: stream dependency broken"

elif mode == "stream_negative_control":
    warmup()
    # Same, but submitted on a different idle stream: the producer's stream is not
    # communicated, so stale data must be observed. Proves the test above can fail.
    # (The producer must be a pure device kernel: a pageable host->device copy would add
    # implicit ordering and hide the missing dependency.)
    n = 1 << 18
    if on_gpu:
        s = torch.cuda.Stream()
        other = torch.cuda.Stream()
        x = torch.full((n,), -1.0, device=dev)
        src = vals(rank, n).to(dev)
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            torch.cuda._sleep(CYCLES)
            x.copy_(src)
        with torch.cuda.stream(other):
            work = dist.all_reduce(x, async_op=True)  # submitted on `other`, not the producer s
        work.wait()
        torch.cuda.synchronize()
        stale = not torch.equal(cpu(x), expected(n))
        print(f"rank {rank} control_stale={stale}")
        assert stale, "negative control did not observe stale data (test is not sensitive)"
    else:
        x = vals(rank, n)
        dist.all_reduce(x, async_op=True).wait()

elif mode == "consumer":
    # After wait(), consume on other streams without any explicit device synchronization.
    n = 1 << 20
    for round_ in range(5):
        x = vals(rank, n, round_).to(dev)
        work = dist.all_reduce(x, async_op=True)
        work.wait()
        if on_gpu:
            s2 = torch.cuda.Stream()
            with torch.cuda.stream(s2):
                y = x * 2.0 + 1.0
            z = x * 3.0  # default stream
            torch.cuda.synchronize()
            assert torch.equal(y.cpu(), expected(n, round_) * 2.0 + 1.0)
            assert torch.equal(z.cpu(), expected(n, round_) * 3.0)
        else:
            assert torch.equal(x * 2.0 + 1.0, expected(n, round_) * 2.0 + 1.0)

elif mode == "lifetime":
    n = 1 << 20
    x = vals(rank, n).to(dev)
    if rank == 1:
        time.sleep(1.0)
    ptr = x.data_ptr()
    work = dist.all_reduce(x, async_op=True)
    del x
    gc.collect()
    for _ in range(50):  # allocator pressure while in flight
        y = torch.full((n,), -1.0, device=dev)
        if rank == 0:
            assert y.data_ptr() != ptr, "storage recycled while Work in flight"
        del y
    work.wait()
    assert torch.equal(cpu(work.result()[0]), expected(n))

elif mode == "overlap":
    # Real GPU compute on a separate tensor, overlapped with a 64 MiB all_reduce.
    n = 16 << 20
    a = torch.randn(2048, 2048, device=dev)

    def compute():
        b = a
        for _ in range(20):
            b = b @ a
            b = b / b.abs().max()
        return b

    def timed(fn):
        if on_gpu:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        fn()
        if on_gpu:
            torch.cuda.synchronize()
        return time.monotonic() - t0

    x = vals(rank, n).to(dev)
    dist.all_reduce(x)  # warmup
    compute()
    comm = timed(lambda: dist.all_reduce(x))
    comp = timed(compute)

    def serial():
        dist.all_reduce(x)
        compute()

    submit = {}

    def overlap():
        t0 = time.monotonic()
        w = dist.all_reduce(x, async_op=True)
        submit["t"] = time.monotonic() - t0
        compute()
        w.wait()

    t_serial = timed(serial)
    t_overlap = timed(overlap)
    print(f"rank {rank} comm={comm:.4f} compute={comp:.4f} serial={t_serial:.4f} "
          f"overlap={t_overlap:.4f} submit={submit['t']*1e3:.3f}ms")
    assert submit["t"] < 0.2 * comm, f"submission blocked: {submit['t']:.4f}s vs comm {comm:.4f}s"

else:
    raise SystemExit(f"unknown mode {mode}")

dist.destroy_process_group()
print(f"rank {rank} ok")
sys.exit(0)
