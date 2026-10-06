"""Small two-rank workload for the physical (two-host) validation: exact-value checks first, a few timings second. Launch with torchrun on each host (static c10d rendezvous
over the Thunderbolt addresses, one process per host); everything is a handful of operations, not a benchmark.

    # host A (rank 0)                                       # host B (rank 1)
    TBCCL_LOCAL_ENDPOINT=<A tb ip>:0 torchrun --nnodes 2    TBCCL_LOCAL_ENDPOINT=<B tb ip>:0 torchrun --nnodes 2 --node-rank 1 ...
        --nproc-per-node 1 --node-rank 0 --master-addr <A tb ip> --master-port P tools/p71_physical_ops.py --mode p2p

--mode p2p        blocking send/recv in both directions + simultaneous isend/irecv, sizes 4 KiB / 1 MiB / 16 MiB (float32), payloads bit-compared
--mode allreduce  SUM all_reduce float32 (4 KiB / 1 MiB / 16 MiB), bfloat16 1 MiB, int64 4 KiB, exact integer values, async variant once
--mode both       p2p then allreduce (one rendezvous)
--cuda-rank R     rank R puts its tensors on cuda:0 (the other rank stays on CPU): the heterogeneous CUDA <-> CPU case
Timings are wall-clock per operation (median of --repeats) and are characterization only.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--mode", default="both", choices=["p2p", "allreduce", "both"])
p.add_argument("--cuda-rank", type=int, default=-1)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--json", default=None)
a = p.parse_args()

dist.init_process_group("tbccl", timeout=timedelta(seconds=120))
rank, peer = dist.get_rank(), 1 - dist.get_rank()
assert dist.get_world_size() == 2
dev = torch.device("cuda" if a.cuda_rank == rank else "cpu")
SIZES = {"4KiB": 1 << 10, "1MiB": 1 << 18, "16MiB": 1 << 22}  # float32 elements


def pattern(sender, n, dtype=torch.float32):
    return ((torch.arange(n, dtype=torch.int64) * 7 + sender * 13) % 101).to(dtype)


def timed(f):
    ts = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 3)


out = {"rank": rank, "device": str(dev), "p2p": {}, "allreduce": {}, "ok": True}


def check(cond, what):
    if not cond:
        out["ok"] = False
        print(f"rank {rank} MISMATCH: {what}", flush=True)


def run_p2p():
    for label, n in SIZES.items():
        res = {"bytes": n * 4}

        def pingpong():
            mine, theirs = pattern(rank, n).to(dev), torch.zeros(n).to(dev)
            if rank == 0:
                dist.send(mine, dst=1)
                dist.recv(theirs, src=1)
            else:
                dist.recv(theirs, src=0)
                dist.send(mine, dst=0)
            check(torch.equal(theirs.cpu(), pattern(peer, n)), f"p2p {label} payload from rank {peer}")

        res["pingpong_ms"] = timed(pingpong)

        def both_async():
            mine, theirs = pattern(rank + 2, n).to(dev), torch.zeros(n).to(dev)
            ws = [dist.isend(mine, dst=peer), dist.irecv(theirs, src=peer)]
            for w in ws:
                w.wait()
            check(torch.equal(theirs.cpu(), pattern(peer + 2, n)), f"isend/irecv {label} payload from rank {peer}")

        res["isend_irecv_ms"] = timed(both_async)
        out["p2p"][label] = res
        print(f"rank {rank} p2p {label}: {res}", flush=True)


def run_allreduce():
    cases = [("f32_" + k, torch.float32, n) for k, n in SIZES.items()] + [("bf16_1MiB", torch.bfloat16, 1 << 19), ("i64_4KiB", torch.int64, 512)]
    for label, dtype, n in cases:
        def once():
            x = (pattern(0, n) % 5 + rank).to(dtype).to(dev)  # values 0..5: exact in every dtype, and their sum too
            dist.all_reduce(x)
            want = ((pattern(0, n) % 5) * 2 + 1).to(torch.float64)
            check(torch.equal(x.cpu().to(torch.float64), want), f"all_reduce {label}")

        out["allreduce"][label] = {"ms": timed(once)}
        print(f"rank {rank} all_reduce {label}: {out['allreduce'][label]}", flush=True)
    x = (pattern(0, 1 << 16) % 5 + rank).to(dev)
    w = dist.all_reduce(x, async_op=True)
    w.wait()
    check(torch.equal(x.cpu(), ((pattern(0, 1 << 16) % 5) * 2 + 1)), "async all_reduce")
    out["allreduce"]["async_256KiB"] = {"ok": True}


if a.mode in ("p2p", "both"):
    run_p2p()
if a.mode in ("allreduce", "both"):
    dist.barrier()  # a collective after the P2P phase: all P2P Works are complete here
    run_allreduce()
dist.barrier()
dist.destroy_process_group()
print("RESULT " + json.dumps(out), flush=True)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
sys.exit(0 if out["ok"] else 1)
