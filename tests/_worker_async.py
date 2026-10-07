"""One rank of an async-Work scenario selected by TEST_MODE. Rank 1 joins collectives late
where the scenario needs rank 0's Work provably in flight."""
import gc
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
LATE = 1.0  # seconds rank 1 waits before joining
dev = torch.device(os.environ["DEVICE_RANK0"] if os.environ.get("DEVICE_RANK0") in ("cuda", "mps") and rank == 0 else "cpu")  # Rank 0 may hold MPS tensors


def vals(r, n, k=0):
    return (torch.arange(n, dtype=torch.float32) % 100) + r * 1000 + k


def mine(n, k=0):
    return vals(rank, n, k).to(dev)


def expected(n, k=0):
    return vals(0, n, k) + vals(1, n, k)


def late():
    if rank == 1:
        time.sleep(LATE)


if mode == "inflight":
    n = 1 << 20
    x = mine(n)
    late()
    t0 = time.monotonic()
    work = dist.all_reduce(x, async_op=True)
    submit = time.monotonic() - t0
    if rank == 0:
        assert submit < 0.5, f"submit blocked {submit:.3f}s"
        assert not work.is_completed(), "work completed before peer joined"
        s = sum(i * i for i in range(200000))  # unrelated CPU work
    fut = work.get_future()
    assert work.wait() is True
    assert work.is_completed() and work.is_success()
    assert work.wait() is True  # repeated wait
    assert torch.equal(x.cpu(), expected(n))
    fut.wait()
    assert torch.equal(fut.value()[0].cpu(), expected(n))

elif mode == "future_independent":
    # The Future must complete without anyone calling work.wait().
    n = 4096
    x = mine(n)
    work = dist.all_reduce(x, async_op=True)
    fut = work.get_future()
    deadline = time.monotonic() + 20
    while not fut.done():
        assert time.monotonic() < deadline, "future never completed"
        time.sleep(0.001)
    assert torch.equal(x.cpu(), expected(n))

elif mode == "lifetime":
    n = 1 << 18
    x = mine(n)
    ptr = x.data_ptr()
    late()
    work = dist.all_reduce(x, async_op=True)
    del x
    gc.collect()
    seen = []
    for i in range(200):  # allocator pressure while the op is (on rank 0) in flight
        y = torch.full((n,), -1.0, device=dev)
        seen.append(y.data_ptr())
        if rank == 0:
            assert y.data_ptr() != ptr, "storage was recycled while Work in flight"
        del y
    work.wait()
    out = work.result()[0]
    assert torch.equal(out.cpu(), expected(n))

elif mode == "sequential":
    n = 5000
    xs = [mine(n, k) for k in range(4)]
    works = [dist.all_reduce(x, async_op=True) for x in xs]  # queued back to back
    for k in reversed(range(4)):
        works[k].wait()
    for k, x in enumerate(xs):
        assert torch.equal(x.cpu(), expected(n, k)), k

elif mode == "timeout":
    n = 1024
    x = mine(n)
    late()
    work = dist.all_reduce(x, async_op=True)
    if rank == 0:
        t0 = time.monotonic()
        try:
            work.wait(timedelta(milliseconds=200))
        except RuntimeError as e:
            assert "timeout" in str(e), e
            assert 0.15 < time.monotonic() - t0 < 5
        else:
            raise AssertionError("bounded wait did not time out")
        assert not work.is_completed()
    work.wait()  # the op was not cancelled; it completes once the peer joins
    assert torch.equal(x.cpu(), expected(n))

elif mode == "threads":
    def threads():
        if os.path.isdir("/proc/self/task"):
            return len(os.listdir("/proc/self/task"))
        import subprocess  # macOS: one `ps -M` row per thread, after the header

        return len(subprocess.check_output(["ps", "-M", "-p", str(os.getpid())], text=True).splitlines()) - 1
    x = mine(128)
    dist.all_reduce(x)
    before = threads()
    for k in range(50):
        y = mine(128, k)
        dist.all_reduce(y, async_op=True).wait()
    assert threads() == before, (before, threads())

elif mode == "peer_dies":
    n = 1 << 16
    x = mine(n)
    if rank == 1:
        time.sleep(0.3)
        os._exit(7)  # abrupt death with rank 0 mid-collective
    work = dist.all_reduce(x, async_op=True)
    fut = work.get_future()
    try:
        work.wait()
    except RuntimeError as e:
        msg = str(e)
        assert msg.startswith("torch-tbccl:"), msg
    else:
        raise AssertionError("wait() did not raise after peer death")
    assert work.is_completed() and not work.is_success()
    try:
        work.wait()
    except RuntimeError:
        pass
    else:
        raise AssertionError("second wait() did not raise")
    deadline = time.monotonic() + 10
    while not fut.done():
        assert time.monotonic() < deadline
        time.sleep(0.001)
    try:
        fut.wait()
    except Exception:
        pass
    else:
        raise AssertionError("future did not fail")
    # subsequent submissions fail promptly, not hang
    try:
        dist.all_reduce(mine(8), async_op=True).wait()
    except Exception:
        pass
    else:
        raise AssertionError("allreduce after peer death succeeded")

else:
    raise SystemExit(f"unknown mode {mode}")

t0 = time.monotonic()
dist.destroy_process_group()
assert time.monotonic() - t0 < 10, "shutdown was slow"
print(f"rank {rank} ok")
sys.exit(0)
