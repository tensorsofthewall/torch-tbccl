"""DDP over torch-tbccl with a silent (alive, not participating) peer, released by an explicit abort.

Rank 1 completes one healthy DDP step, then stops participating but stays alive. Rank 0 enters its next backward (blocked on
the gradient all-reduce) and a control thread aborts the process group after --abort-after seconds. Rank 0 must see backward raise,
tear down in bounded time and exit normally. Launch like ddp_train.py. A c10d Store key (not TBCCL's socket) sequences the ranks.
"""
import argparse
import datetime
import os
import sys
import threading
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_tbccl  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--abort-after", type=float, default=1.0)
p.add_argument("--width", type=int, default=256)
args = p.parse_args()

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=60))
rank = dist.get_rank()
store = dist.distributed_c10d._get_default_store()
dev = torch.device(args.device)
torch.manual_seed(7 + rank)
model = nn.Sequential(nn.Linear(args.width, args.width), nn.ReLU(), nn.Linear(args.width, 8)).to(dev)
ddp = DDP(model, device_ids=[0] if dev.type == "cuda" else None)
x = torch.randn(16, args.width, device=dev)

ddp(x).sum().backward()  # one healthy step
if dev.type == "cuda":
    torch.cuda.synchronize()
print(f"rank {rank}: healthy step done", flush=True)

if rank == 1:
    store.wait(["release"], datetime.timedelta(seconds=60))  # alive and silent
    store.set("r1_done", "1")
    print("rank 1 ok", flush=True)
else:
    def control():
        time.sleep(args.abort_after)
        t = time.monotonic()
        dist.distributed_c10d._abort_process_group()
        print(f"rank 0: abort returned in {(time.monotonic() - t) * 1e3:.1f} ms", flush=True)

    th = threading.Thread(target=control)
    th.start()
    t0 = time.monotonic()
    try:
        ddp(x).sum().backward()
        print("rank 0: backward returned without error (UNEXPECTED)", flush=True)
        rc = 1
    except Exception as e:  # noqa: BLE001
        print(f"rank 0: backward raised after {time.monotonic() - t0:.2f}s: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
        rc = 0
    th.join()
    store.set("release", "1")
    store.wait(["r1_done"], datetime.timedelta(seconds=60))
    print("rank 0 ok" if rc == 0 else "rank 0 FAILED", flush=True)
    sys.stdout.flush()
    os._exit(rc)
os._exit(0)
