"""Two-host Float32 SUM all_reduce over TBCCL (one process per host).

Rendezvous (PyTorch Store) and TBCCL data traffic are separate: MASTER_ADDR/MASTER_PORT locate the
Store; each rank's TBCCL_LOCAL_ENDPOINT is its own data endpoint (TBCCL also uses that port + 1000
on rank 0). Example, Linux (CUDA tensor) = rank 0, Mac (CPU tensor) = rank 1 over Thunderbolt:

  Linux: MASTER_ADDR=192.168.3.2 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 \
         TBCCL_LOCAL_ENDPOINT=192.168.3.2:29600 python cross_host_allreduce.py --device cuda
  Mac:   MASTER_ADDR=192.168.3.2 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 \
         TBCCL_LOCAL_ENDPOINT=192.168.3.1:29600 python cross_host_allreduce.py --device cpu
"""
import argparse
import datetime
import statistics
import time

import torch
import torch.distributed as dist

import torch_tbccl

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--count", type=int, default=262144, help="float32 elements (262144 = 1 MiB)")
parser.add_argument("--timeout", type=float, default=60.0, help="init/bootstrap timeout, seconds")
parser.add_argument("--iters", type=int, default=0, help="extra timed async rounds (0 = none)")
args = parser.parse_args()

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=args.timeout))
rank = dist.get_rank()
dev = torch.device(args.device)
print(f"rank {rank}: device={dev} backend={dist.get_backend()} torch={torch.__version__} "
      f"torch_tbccl={torch_tbccl.__version__} tbccl_runtime={torch_tbccl.runtime_version()}", flush=True)

n = args.count
base = torch.arange(n, dtype=torch.float32) % 4096  # keeps every sum exactly representable
inputs = {0: base + 1, 1: base + 10}
expected = inputs[0] + inputs[1]  # 2*base + 11


def check(x, label):
    ok = torch.equal(x.cpu(), expected)
    print(f"rank {rank}: {label} {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit(1)


warm = torch.ones(1024, device=dev)
dist.all_reduce(warm)
dist.all_reduce(warm)

x = inputs[rank].to(dev, copy=True)
dist.all_reduce(x)
check(x, f"sync  n={n}")

x = inputs[rank].to(dev, copy=True)
work = dist.all_reduce(x, async_op=True)
work.wait()
check(x, f"async n={n}")

if args.iters:
    times, submits = [], []
    for _ in range(args.iters):
        x = inputs[rank].to(dev, copy=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        w = dist.all_reduce(x, async_op=True)
        t1 = time.perf_counter()
        w.wait()
        times.append(time.perf_counter() - t0)
        submits.append(t1 - t0)
    check(x, "last timed round")
    print(f"rank {rank}: total median {statistics.median(times)*1e3:.2f} ms "
          f"(min {min(times)*1e3:.2f}), submit median {statistics.median(submits)*1e6:.0f} us", flush=True)

dist.destroy_process_group()
