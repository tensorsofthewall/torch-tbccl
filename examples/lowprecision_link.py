"""Two-host low-precision all_reduce over TBCCL (one process per host): exactness, latency, and the 57-reduction decode diagnostic.

Launch convention as cross_host_allreduce.py (MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE + TBCCL_LOCAL_ENDPOINT). Each rank picks its own --device, so the real-link shape is
rank 0 = Linux CUDA, rank 1 = Mac CPU (or the reverse).

  --mode correctness  every dtype x shape: all_reduce SUM, result compared bit for bit with the documented semantics computed on CPU from both ranks' regenerated inputs
                      (16-bit floats: widen to float32, add, round once to nearest even; int8/uint8: modulo 2^N). Verification is never inside a timed loop.
  --mode bench        per dtype x shape: warm-up, then --iters timed all_reduce calls (input refilled and the device synchronized OUTSIDE the timed region); prints one JSON
                      line per case with median / p25 / p75 / min / max microseconds.
  --mode seq57        57 sequential all_reduces of [T, 1024] (the audited Qwen3-0.6B TP=2 decode count: 1 embedding + 2 x 28 layers) per repetition; prints median total.
"""
import argparse
import datetime
import json
import statistics as st
import time

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--mode", default="correctness", choices=["correctness", "bench", "seq57"])
parser.add_argument("--dtypes", default="float16,bfloat16,int8,uint8")
parser.add_argument("--shapes", default="1x1,1x17,1x2048,1x1024,4x1024,12x1024,128x1024,512x1024,1x524288", help="RxC shapes, comma separated")
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--reps", type=int, default=30, help="seq57 repetitions")
parser.add_argument("--timeout", type=float, default=120.0)
args = parser.parse_args()

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=args.timeout))
rank = dist.get_rank()
dev = torch.device(args.device)
FLOATS = (torch.float16, torch.bfloat16)


def dt(name):
    return getattr(torch, name)


def contribution(r, shape, dtype, seed):
    g = torch.Generator().manual_seed(1000 * seed + r)
    n = shape[0] * shape[1]
    if dtype in FLOATS:
        return torch.randint(-32768, 32768, (n,), dtype=torch.int16, generator=g).view(dtype).reshape(shape)
    if dtype in (torch.int8, torch.uint8):
        return torch.randint(-128, 128, (n,), dtype=torch.int16, generator=g).to(dtype).reshape(shape)
    return (torch.rand(n, generator=g) * 4 - 2).to(dtype).reshape(shape)  # float32


def expected(dtype, a, b):
    return (a.float() + b.float()).to(dtype) if dtype in FLOATS else a + b


def same(dtype, got, want):
    if dtype in FLOATS:
        nan = torch.isnan(want)
        return torch.equal(torch.isnan(got), nan) and torch.equal(got.view(torch.int16)[~nan], want.view(torch.int16)[~nan])
    return torch.equal(got, want)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
dtypes = [dt(n) for n in args.dtypes.split(",")]
dist.all_reduce(torch.zeros(1024, dtype=torch.float32, device=dev))  # warm-up: connection, staging, CUDA lazy init
sync()
bad = 0

if args.mode == "correctness":
    for dtype in dtypes:
        for i, shape in enumerate(shapes):
            x = contribution(rank, shape, dtype, i).to(dev)
            dist.all_reduce(x)
            ok = same(dtype, x.cpu(), expected(dtype, contribution(0, shape, dtype, i), contribution(1, shape, dtype, i)))
            bad += not ok
            print(f"rank {rank}: {str(dtype).replace('torch.', '')} {shape[0]}x{shape[1]} {'OK' if ok else 'MISMATCH'}", flush=True)

elif args.mode == "bench":
    for dtype in dtypes:
        for i, shape in enumerate(shapes):
            src = contribution(rank, shape, dtype, i).to(dev)
            x = src.clone()
            samples = []
            for it in range(args.warmup + args.iters):
                x.copy_(src)  # untimed refill so the values never saturate
                sync()
                t0 = time.perf_counter()
                dist.all_reduce(x)
                sync()
                t1 = time.perf_counter()
                if it >= args.warmup:
                    samples.append((t1 - t0) * 1e6)
            samples.sort()
            if rank == 0:
                print(json.dumps({
                    "dtype": str(dtype).replace("torch.", ""), "shape": list(shape), "bytes": x.numel() * x.element_size(), "device": args.device, "iters": args.iters,
                    "median_us": round(st.median(samples), 1), "p25_us": round(samples[len(samples) // 4], 1), "p75_us": round(samples[3 * len(samples) // 4], 1),
                    "min_us": round(samples[0], 1), "max_us": round(samples[-1], 1)}), flush=True)

else:  # seq57
    for dtype in dtypes:
        for shape in shapes:
            x = contribution(rank, shape, dtype, 3).to(dev)
            totals = []
            for rep in range(args.warmup // 10 + args.reps):
                sync()
                t0 = time.perf_counter()
                for _ in range(57):
                    dist.all_reduce(x)
                sync()
                t1 = time.perf_counter()
                if rep >= args.warmup // 10:
                    totals.append((t1 - t0) * 1e3)
                x.copy_(contribution(rank, shape, dtype, 3).to(dev))  # untimed refill between sequences
            totals.sort()
            if rank == 0:
                print(json.dumps({
                    "dtype": str(dtype).replace("torch.", ""), "shape": list(shape), "reductions": 57, "device": args.device, "reps": args.reps,
                    "median_total_ms": round(st.median(totals), 2), "p25_ms": round(totals[len(totals) // 4], 2), "p75_ms": round(totals[3 * len(totals) // 4], 2),
                    "min_ms": round(totals[0], 2), "per_reduction_us": round(1e3 * st.median(totals) / 57, 1)}), flush=True)

dist.destroy_process_group()
raise SystemExit(1 if bad else 0)
