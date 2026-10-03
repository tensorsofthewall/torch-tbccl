"""N-rank DDP smoke (stretch goal): a tiny CPU MLP, different seeds per rank (DDP must sync to rank 0's), one optimizer step on different inputs,
checked against an independent reference (mean of the N ranks' local gradients, computed locally with no backend)."""
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()


def make(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def batch(r):
    g = torch.Generator().manual_seed(7000 + r)
    return torch.randn(8, 16, generator=g), torch.randn(8, 4, generator=g)


model = make(100 + rank)
ddp = DDP(model)
ref = make(100)  # rank 0's initial weights
assert all(torch.equal(a, b) for a, b in zip(model.parameters(), ref.parameters())), "constructor sync"

opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
x, y = batch(rank)
opt.zero_grad()
nn.functional.mse_loss(ddp(x), y).backward()
opt.step()

ref_grads = []
for r in range(world):
    ref.zero_grad()
    xr, yr = batch(r)
    nn.functional.mse_loss(ref(xr), yr).backward()
    ref_grads.append([p.grad.clone() for p in ref.parameters()])
with torch.no_grad():
    for i, p in enumerate(ref.parameters()):
        p -= 0.1 * sum(g[i] for g in ref_grads) / world
err = max((a - b).abs().max().item() for a, b in zip(model.parameters(), ref.parameters()))
assert err < 1e-5, f"parameters differ from the reference by {err}"

flat = torch.cat([p.detach().flatten() for p in model.parameters()])
gathered = [torch.zeros_like(flat) for _ in range(world)]
dist.all_gather(gathered, flat)
assert all(torch.equal(gathered[0], g) for g in gathered), "ranks diverged"
dist.destroy_process_group()
print(f"rank {rank} ok")
sys.exit(0)
