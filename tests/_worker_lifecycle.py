"""Repeated lifecycle with resource accounting (one rank). TEST_MODE:
  cycles   CYCLES x (init_process_group over ONE persistent store -> all_reduce + send/recv + barrier -> destroy_process_group)
  groups   one init, then GROUPS x (new_group over both ranks -> destroy_process_group(group)), no communication on the groups
After a warm-up the file-descriptor count and the Python/OS thread counts must be flat; RSS (and CUDA memory when DEVICE=cuda) must stay bounded.
The per-checkpoint numbers are printed as one JSON line for the evidence files."""
import gc
import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401
from _procinfo import fd_count, os_threads, rss_mb

mode = os.environ["TEST_MODE"]
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
dev = torch.device("cuda" if os.environ.get("DEVICE") == "cuda" else "cpu")
store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), world, is_master=rank == 0, timeout=timedelta(seconds=60))


def snapshot():
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    import threading

    snap = {"fds": fd_count(), "os_threads": os_threads(), "py_threads": threading.active_count(), "rss_mb": round(rss_mb(), 1)}
    if dev.type == "cuda":
        snap["cuda_alloc_mb"] = round(torch.cuda.memory_allocated() / 2**20, 2)
        snap["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 2)
    return snap


def one_cycle(k):
    dist.init_process_group("tbccl", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=30))
    x = torch.full((1024,), float(rank + 1 + k), device=dev)
    dist.all_reduce(x)
    assert x[0].item() == sum(r + 1 + k for r in range(world))
    if rank == 0:
        for dst in range(1, world):
            dist.send(torch.full((64,), float(k), device=dev), dst=dst)
    else:
        y = torch.zeros(64, device=dev)
        dist.recv(y, src=0)
        assert y[0].item() == float(k)
    dist.barrier()
    dist.destroy_process_group()


points = {}
t_start = time.monotonic()
if mode == "cycles":
    n, warm = int(os.environ["CYCLES"]), 2
    for k in range(n):
        one_cycle(k)
        if k + 1 == warm:
            points["after_warmup"] = snapshot()
    points["end"] = snapshot()
else:
    n, warm = int(os.environ["GROUPS"]), 5
    dist.init_process_group("tbccl", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=30))
    dist.all_reduce(torch.ones(8, device=dev))
    for k in range(n):
        g = dist.new_group(list(range(world)))
        dist.destroy_process_group(g)
        if k + 1 == warm:
            points["after_warmup"] = snapshot()
    points["end"] = snapshot()
    dist.all_reduce(torch.ones(8, device=dev))  # the default group still works
    dist.destroy_process_group()

a, b = points["after_warmup"], points["end"]
print("LIFECYCLE " + json.dumps({"rank": rank, "mode": mode, "n": n, "wall_s": round(time.monotonic() - t_start, 1), "after_warmup": a, "end": b}), flush=True)
assert b["fds"] == a["fds"], f"file descriptors grew {a['fds']} -> {b['fds']}"
assert b["os_threads"] == a["os_threads"] and b["py_threads"] == a["py_threads"], (a, b)
assert b["rss_mb"] - a["rss_mb"] < 60, f"RSS grew {b['rss_mb'] - a['rss_mb']:.1f} MB"
if dev.type == "cuda":
    assert b["cuda_reserved_mb"] - a["cuda_reserved_mb"] < 64, "CUDA memory grew"
print(f"rank {rank} ok", flush=True)
sys.stdout.flush()
os._exit(0)
