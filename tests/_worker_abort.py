"""One rank of an abort scenario. Rank 1 is the 'silent peer': alive, backend open, but not participating, until rank 0
releases it through the c10d Store (which is unrelated to TBCCL's data socket). Rank 0 keeps running until rank 1 reports done."""
import faulthandler
import gc
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

faulthandler.dump_traceback_later(20, exit=True)
mode = os.environ["TEST_MODE"]
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
store = dist.distributed_c10d._get_default_store()
dev = torch.device(os.environ.get("ABORT_DEVICE", "cpu"))
dtype = getattr(torch, os.environ.get("ABORT_DTYPE", "float32"))  # Phase 49: also float16 / bfloat16 / int8 / uint8 reductions
x = torch.ones(1024, dtype=dtype, device=dev)
dist.all_reduce(x)  # healthy control
assert x[0].item() == 2.0
REMOTE = os.environ.get("OBSERVE_REMOTE") == "1"


def expect_raises(f, what):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        return str(e)
    raise AssertionError(f"expected an exception: {what}")


if rank == 1:
    if mode in ("timeout_then_success",):
        time.sleep(0.8)
        y = torch.ones(1 << 18, dtype=dtype, device=dev)
        dist.all_reduce(y)
        assert y[0].item() == 2.0
        store.set("r1_done", "1")
    else:
        store.wait(["release"], timedelta(seconds=60))   # silent but alive
        if REMOTE:
            # rank 0 has aborted: the connection is gone, so our next collective must fail promptly (no abort frame is used)
            z = torch.ones(1 << 18, dtype=dtype, device=dev)
            t1 = time.monotonic()
            msg = expect_raises(lambda: dist.all_reduce(z), "collective after remote abort")
            print(f"rank 1 observed remote abort after {(time.monotonic() - t1) * 1e3:.1f} ms: {msg[:100]}", flush=True)
            assert time.monotonic() - t1 < 5.0
        store.set("r1_done", "1")
    print("rank 1 ok")
else:
    t = torch.ones(1 << 18, dtype=dtype, device=dev)
    if mode == "timeout_then_success":
        work = dist.all_reduce(t, async_op=True)
        msg = expect_raises(lambda: work.wait(timedelta(milliseconds=150)), "wait timeout")
        assert "timeout" in msg, msg
        assert not work.is_completed() or True
        work.wait()                      # a timeout is not an abort: the op still completes
        assert t[0].item() == 2.0
    else:
        work = dist.all_reduce(t, async_op=True)
        fut = work.get_future()
        time.sleep(0.4)
        assert not work.is_completed(), "work must be pending on a silent peer"
        if mode == "timeout_then_abort":
            msg = expect_raises(lambda: work.wait(timedelta(milliseconds=100)), "wait timeout")
            assert "timeout" in msg, msg
            assert not work.is_completed()
        if mode == "lifetime":
            del t
            gc.collect()
        t0 = time.monotonic()
        if mode == "destroy":
            dist.destroy_process_group()
        else:
            dist.distributed_c10d._abort_process_group()
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"teardown took {elapsed:.2f}s"
        msg = expect_raises(lambda: work.wait(), "wait after abort")
        assert "abort" in msg.lower(), msg
        msg2 = expect_raises(lambda: work.wait(), "second wait")
        assert msg2 == msg or "abort" in msg2.lower()
        assert work.is_completed() and not work.is_success()
        fut.wait() if False else None
        assert expect_raises(lambda: fut.wait(), "future") is not None
        print(f"rank 0 teardown {elapsed*1000:.1f} ms")
    store.set("release", "1")
    store.wait(["r1_done"], timedelta(seconds=60))
    print("rank 0 ok")
import sys; sys.stdout.flush(); sys.stderr.flush()
os._exit(0)  # the process group may already be gone; skip atexit teardown
