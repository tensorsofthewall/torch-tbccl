"""Two-host broadcast and all_gather correctness over TBCCL (one process per host).

Same launch convention as cross_host_allreduce.py. Runs, for each size, broadcast from both roots
and an all_gather, on this host's --device, checking every byte against a deterministic pattern.
"""
import argparse
import datetime

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--sizes", default="8,4096,1048576,16777216", help="bytes, comma separated")
parser.add_argument("--timeout", type=float, default=60.0)
args = parser.parse_args()

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=args.timeout))
rank = dist.get_rank()
dev = torch.device(args.device)


def pattern(nbytes, seed):
    return ((torch.arange(nbytes, dtype=torch.int64) * 131 + seed * 17 + 3) % 251).to(torch.uint8)


dist.broadcast(torch.zeros(16, dtype=torch.uint8, device=dev), src=0)  # warm-up
bad = 0
for nbytes in (int(s) for s in args.sizes.split(",")):
    for root in (0, 1):
        src = pattern(nbytes, 9 + root)
        x = src.to(dev, copy=True) if rank == root else torch.full((nbytes,), 0xEE, dtype=torch.uint8, device=dev)
        dist.broadcast(x, src=root)
        ok = torch.equal(x.cpu(), src)
        bad += not ok
        print(f"rank {rank}: broadcast root={root} bytes={nbytes} {'OK' if ok else 'MISMATCH'}", flush=True)
    outs = [torch.full((nbytes,), 0xEE, dtype=torch.uint8, device=dev) for _ in range(2)]
    dist.all_gather(outs, pattern(nbytes, 21 + rank).to(dev, copy=True))
    ok = torch.equal(outs[0].cpu(), pattern(nbytes, 21)) and torch.equal(outs[1].cpu(), pattern(nbytes, 22))
    bad += not ok
    print(f"rank {rank}: all_gather bytes={nbytes} {'OK' if ok else 'MISMATCH'}", flush=True)

counts = torch.tensor([1000 + rank * 17], dtype=torch.int64, device=dev)
gathered = [torch.zeros(1, dtype=torch.int64, device=dev) for _ in range(2)]
dist.all_gather(gathered, counts)
ok = [int(g.item()) for g in gathered] == [1000, 1017]
bad += not ok
print(f"rank {rank}: int64 metadata all_gather {'OK' if ok else 'MISMATCH'}", flush=True)
dist.destroy_process_group()
raise SystemExit(1 if bad else 0)
