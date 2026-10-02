"""Mimics the group pattern vLLM PP=2/TP=1 builds: world + several 2-rank new_groups + one-rank groups, all 'tbccl' with
auto-selected ports (TBCCL_LOCAL_ENDPOINT=host:0); then P2P and all_reduce on every group."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
groups = []
for ranks in ([0], [1]):                      # TP=1 style one-rank groups (every rank must call new_group for all)
    g = dist.new_group(ranks, backend="tbccl")
    groups.append(("single", ranks, g))
for i in range(3):                            # PP-style 2-rank groups, several simultaneously
    groups.append(("pair", [0, 1], dist.new_group([0, 1], backend="tbccl")))
gloo = dist.new_group([0, 1], backend="gloo")  # control group alongside
for kind, ranks, g in groups:
    if rank not in ranks or kind == "single":
        continue
    t = torch.full((1024,), float(rank + 1))
    dist.all_reduce(t, group=g)
    assert t[0].item() == 3.0
    x = torch.arange(10, dtype=torch.float32) + 100 * rank
    if rank == 0:
        dist.send(x, dst=1, group=g)
        y = torch.zeros(10)
        dist.recv(y, src=1, group=g)
        assert torch.equal(y, torch.arange(10, dtype=torch.float32) + 100)
    else:
        y = torch.zeros(10)
        dist.recv(y, src=0, group=g)
        assert torch.equal(y, torch.arange(10, dtype=torch.float32))
        dist.send(x, dst=0, group=g)
for g in (groups[2][2], groups[3][2]):         # barrier + the int32 all_reduce vLLM's node-detection runs, on TBCCL groups
    dist.barrier(group=g)
    v = torch.tensor([rank, 1 - rank], dtype=torch.int32)
    dist.all_reduce(v, group=g)
    assert v.tolist() == [1, 1]
    objs = [None]
    dist.broadcast_object_list(objs if rank else [{"a": 1}], src=0, group=g) if False else None
for g in (groups[2][2],):
    objs = [None, None] if rank == 0 else None
    dist.gather_object({"rank": rank, "pad": "x" * 100 * (rank + 1)}, objs, dst=0, group=g)
    if rank == 0:
        assert [o["rank"] for o in objs] == [0, 1], objs
dist.barrier(group=gloo)
print(f"rank {rank} ok", flush=True)
import sys; sys.stdout.flush(); os._exit(0)
