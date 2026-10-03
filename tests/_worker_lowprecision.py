"""One rank of a low-precision reduction scenario (Phase 49): float16 / bfloat16 / int8 / uint8 SUM all_reduce through torch.distributed.

CUDA_RANKS (default none) lists the ranks whose tensors live on the GPU; the others use CPU. Expected results are computed with torch on CPU from
both ranks' (regenerated) contributions: 16-bit floats = widen to float32, add, round once (the TBCCL semantics); integers wrap modulo 2^N.
"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
cuda_ranks = {int(r) for r in os.environ.get("CUDA_RANKS", "").split(",") if r}
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
on_gpu = rank in cuda_ranks
dev = torch.device("cuda", 0) if on_gpu else torch.device("cpu")
CYCLES = 300_000_000  # ~0.2 s of GPU time

FLOATS = (torch.float16, torch.bfloat16)
INTS = (torch.int8, torch.uint8)
SHAPES = [(1,), (17,), (4096,), (1, 1024), (4, 1024), (12, 1024), (128, 1024), (512, 1024), (1 << 20,)]


def contribution(r, shape, dtype, seed):
    g = torch.Generator().manual_seed(1000 * seed + r)
    n = 1
    for d in shape:
        n *= d
    if dtype in FLOATS:
        # arbitrary 16-bit patterns: subnormals, infinities, NaNs and extremes included
        bits = torch.randint(-32768, 32768, (n,), dtype=torch.int16, generator=g)
        return bits.view(dtype).reshape(shape)
    return torch.randint(-128, 128, (n,), dtype=torch.int16, generator=g).to(dtype).reshape(shape)


def expected_sum(dtype, a, b):
    if dtype in FLOATS:
        return (a.float() + b.float()).to(dtype)
    return a + b  # torch integer add wraps modulo 2^N


def same(dtype, got, want):
    if dtype in FLOATS:
        nan = torch.isnan(want)
        if not torch.equal(torch.isnan(got), nan):
            return False
        return torch.equal(got.view(torch.int16)[~nan], want.view(torch.int16)[~nan])
    return torch.equal(got, want)


def check(dtype, shape, seed, async_op):
    mine = contribution(rank, shape, dtype, seed)
    x = mine.to(dev)
    work = dist.all_reduce(x, async_op=async_op)
    if async_op:
        work.wait()
    want = expected_sum(dtype, contribution(0, shape, dtype, seed), contribution(1, shape, dtype, seed))
    assert same(dtype, x.cpu(), want), f"{dtype} shape {shape} async={async_op}: mismatch"


def warmup(dtype):
    # TBCCL's first CUDA collective pays lazy setup (staging allocation, stream creation); without this the producer delay in "stream" mode could
    # elapse before TBCCL even starts, hiding a missing dependency.
    w = torch.ones(1 << 18, dtype=dtype, device=dev)
    dist.all_reduce(w)
    dist.all_reduce(w)
    if on_gpu:
        torch.cuda.synchronize()


if mode == "allreduce":
    for dtype in FLOATS + INTS:
        for i, shape in enumerate(SHAPES):
            check(dtype, shape, i, async_op=False)
        for r in range(6):  # repeated rounds, async, distinct data
            check(dtype, (4099,), 100 + r, async_op=(r % 2 == 1))
        z = torch.empty(0, dtype=dtype, device=dev)
        dist.all_reduce(z)
        assert z.numel() == 0
    # in-place into a view: neighbors untouched
    for dtype in FLOATS + INTS:
        big = torch.zeros(64, dtype=dtype)
        big[8:40] = contribution(rank, (32,), dtype, 7)
        t = big.to(dev)
        dist.all_reduce(t[8:40])
        got = t.cpu()
        assert same(dtype, got[8:40], expected_sum(dtype, contribution(0, (32,), dtype, 7), contribution(1, (32,), dtype, 7)))
        assert (got[:8] == 0).all() and (got[40:] == 0).all(), "neighbors of the reduced view were modified"

elif mode == "stream":
    for dtype in FLOATS + INTS:
        warmup(dtype)
        n = 1 << 18
        s = torch.cuda.Stream() if on_gpu else None
        mine = contribution(rank, (n,), dtype, 31)
        if on_gpu:
            src = mine.to(dev)  # produced on-device so the stream op below is a pure kernel
            x = torch.zeros(n, dtype=dtype, device=dev)
            torch.cuda.synchronize()
            with torch.cuda.stream(s):
                torch.cuda._sleep(CYCLES)  # delayed producer on a non-default stream; all_reduce submitted with NO synchronization
                x.copy_(src)
                work = dist.all_reduce(x, async_op=True)
        else:
            x = mine.clone()
            work = dist.all_reduce(x, async_op=True)
        work.wait()
        want = expected_sum(dtype, contribution(0, (n,), dtype, 31), contribution(1, (n,), dtype, 31))
        assert same(dtype, x.cpu(), want), f"{dtype}: stale data, the producer-stream dependency is broken"

elif mode == "stream_negative_control":
    # The producer delays on stream `s` but all_reduce is submitted on a different idle stream, so the dependency is NOT communicated and stale data
    # must be observed. Proves the "stream" mode above can fail for the 16-bit types too (the producer is a pure device kernel on purpose: a pageable
    # host->device copy would add implicit ordering and hide the missing dependency).
    for dtype in FLOATS:
        warmup(dtype)
        n = 1 << 18
        if on_gpu:
            s, other = torch.cuda.Stream(), torch.cuda.Stream()
            src = contribution(rank, (n,), dtype, 31).to(dev)
            x = torch.zeros(n, dtype=dtype, device=dev)
            torch.cuda.synchronize()
            with torch.cuda.stream(s):
                torch.cuda._sleep(CYCLES)
                x.copy_(src)
            with torch.cuda.stream(other):
                work = dist.all_reduce(x, async_op=True)
            work.wait()
            torch.cuda.synchronize()
            want = expected_sum(dtype, contribution(0, (n,), dtype, 31), contribution(1, (n,), dtype, 31))
            stale = not same(dtype, x.cpu(), want)
            assert stale, f"{dtype}: negative control did not observe stale data (the stream test is not sensitive)"
        else:
            dist.all_reduce(contribution(rank, (n,), dtype, 31), async_op=True).wait()

elif mode == "errors":
    def promptly(exc, fragment, fn):
        t0 = time.monotonic()
        try:
            fn()
        except exc as e:
            assert fragment in str(e), f"wrong message: {e}"
        else:
            raise AssertionError(f"no error for {fragment!r}")
        assert time.monotonic() - t0 < 5, "error was not prompt"

    for dtype in FLOATS + INTS:
        for op, name in ((dist.ReduceOp.PRODUCT, "PRODUCT"), (dist.ReduceOp.MIN, "MIN"), (dist.ReduceOp.MAX, "MAX")):
            label = str(dtype).replace("torch.", "")
            promptly(NotImplementedError, f"dtype={label} op={name} (supported ops: SUM)", lambda: dist.all_reduce(torch.ones(8, dtype=dtype, device=dev), op=op))
    # the group is still usable after the rejected calls
    x = torch.full((16,), float(rank + 1), dtype=torch.float16, device=dev)
    dist.all_reduce(x)
    assert torch.equal(x.cpu(), torch.full((16,), 3.0, dtype=torch.float16))

else:
    raise SystemExit(f"unknown TEST_MODE {mode}")

dist.destroy_process_group()
print(f"rank {rank} ok")
