"""Minimal DistributedDataParallel training over the "tbccl" backend, checked against a single-process reference.

    TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --nproc-per-node 2 examples/ddp_mlp.py

Every rank trains the same small MLP on its shard of one deterministic global batch (rank r takes rows [r*GB/W, (r+1)*GB/W)); DDP averages the shard
gradients, which for equal shards equals the gradient of the global-batch mean loss. Each rank also trains a plain, non-distributed copy on the whole
global batch in the same process and compares per-step loss, gradient norm and the final parameters. No random numbers, datasets or downloads: the
initial weights and the data are exact integer arithmetic scaled by powers of two, so every rank on every architecture starts bit-identical.

Exit status 0 only when the DDP run agrees with the reference within --tol and all ranks hold identical parameters.
Devices: --devices cpu | cuda | cuda,cpu (one entry for every rank, or one per rank). --unused adds a parameter that forward never touches
(find_unused_parameters=True). --cycles N repeats init -> train -> destroy in one process.
"""
import argparse
import json
import math
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_tbccl  # noqa: F401  (registers the "tbccl" backend)

IN, OUT = 8, 4


def det(shape, a, b, m, div):
    """Deterministic values ((i*a + b) % m - m//2) / div: exact in float32, no RNG."""
    n = math.prod(shape)
    i = torch.arange(n, dtype=torch.int64)
    return (((i * a + b) % m) - m // 2).to(torch.float32).div(div).reshape(shape)


class MLP(nn.Module):
    def __init__(self, hidden, unused):
        super().__init__()
        self.fc1 = nn.Linear(IN, hidden)
        self.fc2 = nn.Linear(hidden, OUT)
        self.aux = nn.Linear(IN, OUT) if unused else None  # never used by forward
        with torch.no_grad():
            for k, p in enumerate(self.parameters()):
                p.copy_(det(tuple(p.shape), 5 + k, 3 * k + 1, 11, 16))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def batch(step, global_batch):
    return det((global_batch, IN), 7 + step, 13, 17, 8), det((global_batch, OUT), 3 + step, 5, 13, 8)


def grad_norm(model):
    return math.sqrt(sum(float((p.grad.detach().cpu().double() ** 2).sum()) for p in model.parameters() if p.grad is not None))


def flat(model):
    return torch.cat([p.detach().cpu().double().flatten() for p in model.parameters()])


def reference(args):
    model, losses, norms = MLP(args.hidden, args.unused), [], []
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    for step in range(args.steps):
        x, y = batch(step, args.global_batch)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        losses.append(float(loss.detach()))
        norms.append(grad_norm(model))
        opt.step()
    return losses, norms, flat(model)


def train_cycle(args):
    dist.init_process_group("tbccl", timeout=timedelta(seconds=args.timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    names = args.devices.split(",")
    dev = torch.device(names[rank] if len(names) > 1 else names[0])
    assert args.global_batch % world == 0, f"--global-batch must be divisible by the world size ({world})"
    shard = args.global_batch // world

    ddp = DDP(MLP(args.hidden, args.unused).to(dev), bucket_cap_mb=args.bucket_cap_mb, find_unused_parameters=args.unused)
    opt = torch.optim.SGD(ddp.parameters(), lr=args.lr, momentum=0.9)
    losses, norms = [], []
    for step in range(args.steps):
        x, y = batch(step, args.global_batch)
        x, y = x[rank * shard:(rank + 1) * shard].to(dev), y[rank * shard:(rank + 1) * shard].to(dev)
        opt.zero_grad()
        loss = nn.functional.mse_loss(ddp(x), y)
        loss.backward()
        mean_loss = loss.detach().clone().reshape(1)
        dist.all_reduce(mean_loss)  # SUM of the shard means
        losses.append(float(mean_loss.cpu()) / world)
        norms.append(grad_norm(ddp.module))
        opt.step()

    mine = flat(ddp.module)
    mine32 = mine.float().to(dev)  # the parameters are float32 already; MPS has no float64
    gathered = [torch.zeros_like(mine32) for _ in range(world)]
    dist.all_gather(gathered, mine32)
    identical = all(torch.equal(g.cpu(), gathered[0].cpu()) for g in gathered)
    cross_rank_diff = max(float((g.cpu() - gathered[0].cpu()).abs().max()) for g in gathered)
    # Gradients are reduced identically, but the optimizer step runs on each rank's own device: CPU and MPS (and CPU and CUDA kernels in general) may differ by an ulp
    # per step. Ranks on the same device type must be bit-identical; an MPS rank next to another device type is held to --cross-device-tol.
    mixed_mps = "mps" in names and len(set(names)) > 1
    consistent = identical or (mixed_mps and cross_rank_diff <= args.cross_device_tol)

    ref_losses, ref_norms, ref_params = reference(args)
    loss_err = max(abs(a - b) / max(1e-12, abs(b)) for a, b in zip(losses, ref_losses))
    norm_err = max(abs(a - b) / max(1e-12, abs(b)) for a, b in zip(norms, ref_norms))
    param_err = float((mine - ref_params).abs().max())
    ok = consistent and loss_err < args.tol and norm_err < args.tol and param_err < args.tol
    result = {"rank": rank, "world": world, "device": str(dev), "steps": args.steps, "bucket_cap_mb": args.bucket_cap_mb, "unused": args.unused,
              "loss_rel_err": loss_err, "grad_norm_rel_err": norm_err, "param_max_abs_err": param_err, "ranks_identical": identical, "cross_rank_max_abs_diff": cross_rank_diff,
              "first_loss": losses[0], "last_loss": losses[-1], "ref_last_loss": ref_losses[-1], "ok": ok}
    dist.destroy_process_group()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--global-batch", type=int, default=24)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--cross-device-tol", type=float, default=1e-6, help="max abs parameter difference between ranks on different device types when one is MPS")
    p.add_argument("--bucket-cap-mb", type=float, default=25.0)
    p.add_argument("--devices", default="cpu")
    p.add_argument("--unused", action="store_true")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--json", default=None, help="write the per-cycle results here (rank 0 only)")
    args = p.parse_args()
    results = [train_cycle(args) for _ in range(args.cycles)]
    for r in results:
        os.write(1, ("RESULT " + json.dumps(r) + "\n").encode())  # one write: ranks sharing a stdout pipe must not interleave inside a line
    if args.json and results[0]["rank"] == 0:
        json.dump(results, open(args.json, "w"), indent=1)
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
