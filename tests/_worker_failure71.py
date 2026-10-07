"""The packaging and capability-audit work peer-failure, abort and teardown semantics, one scenario per TEST_MODE. Loopback only: the dying peer is a local process, never a physical host.

Peer exits (rank 1 dies abruptly with os._exit(9); the survivors must fail within BOUND seconds with a structured torch-tbccl error, never hang):
  recv_peer_exit[_blocking]   rank 0 waits in irecv / a blocking recv
  send_peer_exit              rank 0's 256 MiB isend can never complete
  allreduce_peer_exit_before  the peer is gone before the collective starts
  allreduce_peer_exit_during  the peer dies in the middle of a 128 MiB all_reduce
  allreduce_w3                world 3: rank 2 dies, ranks 0 and 1 are in an all_reduce
  ddp_peer_exit               rank 1 dies between DDP steps; rank 0's next backward must raise
Abort / teardown with a silent but alive peer (released through the c10d store, unrelated to TBCCL's socket):
  abort_recv, abort_send, abort_collective     _abort_process_group() with that operation pending: every Work and Future goes terminal, bounded
  destroy_outstanding                          destroy_process_group() with a pending irecv + a pending large isend: bounded, Works fail (abort), no hang
"""
import faulthandler
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_tbccl  # noqa: F401

faulthandler.dump_traceback_later(float(os.environ.get("HANG_DUMP", "120")), exit=True)  # a hang becomes a stack dump + nonzero exit
mode = os.environ["TEST_MODE"]
BOUND = float(os.environ.get("BOUND", "20"))
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()
store = dist.distributed_c10d._get_default_store()
dev = torch.device(os.environ["DEVICE_RANK0"] if os.environ.get("DEVICE_RANK0") in ("cuda", "mps") and rank == 0 else "cpu")
MB = 1 << 20


def die(code=9):
    sys.stdout.flush()
    os._exit(code)


def expect_failure(fn, what):
    t0 = time.monotonic()
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        dt = time.monotonic() - t0
        msg = str(e)
        assert dt < BOUND, f"{what}: failure took {dt:.1f}s"
        assert "torch-tbccl" in msg, f"{what}: not a structured torch-tbccl error: {msg[:200]}"
        print(f"rank {rank} {what}: {type(e).__name__} after {dt * 1e3:.0f} ms: {msg[:160]}", flush=True)
        return dt
    raise AssertionError(f"{what}: completed although the peer is gone")


def finish():
    t0 = time.monotonic()
    dist.destroy_process_group()
    assert time.monotonic() - t0 < 10, "destroy_process_group was slow after the failure"
    print(f"rank {rank} ok", flush=True)
    sys.stdout.flush()
    os._exit(0)  # the peer is gone; skip any atexit coordination


dist.all_reduce(torch.ones(8, device=dev))  # a healthy first collective on every rank
dist.barrier()

if mode in ("recv_peer_exit", "recv_peer_exit_blocking"):
    if rank == 1:
        time.sleep(0.5)
        die()
    buf = torch.zeros(1 << 16, device=dev)
    if mode == "recv_peer_exit":
        w = dist.irecv(buf, src=1)
        expect_failure(w.wait, "irecv wait")
        assert w.is_completed() and not w.is_success()
    else:
        expect_failure(lambda: dist.recv(buf, src=1), "blocking recv")
    finish()

elif mode == "send_peer_exit":
    if rank == 1:
        time.sleep(0.5)
        die()
    w = dist.isend(torch.ones(64 * MB, device=dev), dst=1)  # 256 MiB of float32: cannot be absorbed by socket buffers
    expect_failure(w.wait, "isend wait")
    assert w.is_completed() and not w.is_success()
    finish()

elif mode == "allreduce_peer_exit_before":
    if rank == 1:
        die()
    time.sleep(0.5)
    # the communicator has already seen the peer's connection close, so the submission itself may raise; either way it must fail, not hang
    expect_failure(lambda: dist.all_reduce(torch.ones(MB, device=dev), async_op=True).wait(), "all_reduce (peer already gone)")
    finish()

elif mode == "allreduce_peer_exit_during":
    x = torch.ones(32 * MB, device=dev)  # 128 MiB
    w = dist.all_reduce(x, async_op=True)
    if rank == 1:
        time.sleep(0.03)
        die()
    expect_failure(w.wait, "all_reduce wait (peer died mid-collective)")
    # the group stays failed: a later submission fails promptly instead of hanging
    expect_failure(lambda: dist.all_reduce(torch.ones(8, device=dev)), "all_reduce after the failure")
    finish()

elif mode == "allreduce_w3":
    assert world == 3
    if rank == 2:
        time.sleep(0.3)
        die()
    w = dist.all_reduce(torch.ones(8 * MB, device=dev), async_op=True)
    expect_failure(w.wait, f"rank {rank} all_reduce wait (rank 2 died)")
    finish()

elif mode == "ddp_peer_exit":
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 8)).to(dev)
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
    for step in range(2):
        opt.zero_grad()
        ddp(torch.ones(4, 256, device=dev)).sum().backward()
        opt.step()
    if rank == 1:
        time.sleep(0.3)
        die()
    time.sleep(1.0)  # rank 1 is gone before rank 0's next backward starts the gradient reduction

    def step():
        opt.zero_grad()
        ddp(torch.ones(4, 256, device=dev)).sum().backward()

    expect_failure(step, "DDP backward (peer gone)")
    finish()

elif mode in ("abort_recv", "abort_send", "abort_collective", "destroy_outstanding"):
    if rank == 1:
        store.wait(["release"], timedelta(seconds=60))  # silent but alive
        store.set("r1_done", "1")
        print("rank 1 ok", flush=True)
        os._exit(0)
    if mode == "abort_recv":
        works = [dist.irecv(torch.zeros(1 << 16, device=dev), src=1)]
    elif mode == "abort_send":
        works = [dist.isend(torch.ones(64 * MB, device=dev), dst=1)]
    elif mode == "abort_collective":
        works = [dist.all_reduce(torch.ones(MB, device=dev), async_op=True)]
    else:
        works = [dist.irecv(torch.zeros(1 << 16, device=dev), src=1), dist.isend(torch.ones(64 * MB, device=dev), dst=1)]
    futs = [w.get_future() for w in works]
    time.sleep(0.4)
    assert not any(w.is_completed() for w in works), "work must be pending on a silent peer"
    t0 = time.monotonic()
    if mode == "destroy_outstanding":
        dist.destroy_process_group()
    else:
        dist.distributed_c10d._abort_process_group()
    dt = time.monotonic() - t0
    assert dt < 5.0, f"teardown took {dt:.2f}s"
    for w, f in zip(works, futs):
        try:
            w.wait()
        except Exception as e:  # noqa: BLE001
            assert "abort" in str(e).lower(), str(e)
        else:
            raise AssertionError("a pending Work survived the abort")
        assert w.is_completed() and not w.is_success()
        try:
            f.wait()
        except Exception:  # noqa: BLE001
            pass
        else:
            raise AssertionError("a pending Future survived the abort")
    print(f"rank 0 {mode}: {len(works)} pending Work(s) terminated, teardown {dt * 1e3:.1f} ms", flush=True)
    store.set("release", "1")
    store.wait(["r1_done"], timedelta(seconds=60))
    print("rank 0 ok", flush=True)
    sys.stdout.flush()
    os._exit(0)

else:
    raise SystemExit(f"unknown mode {mode}")
