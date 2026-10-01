"""Real torch.nn.parallel.DistributedDataParallel over torch-tbccl (one process per rank, world_size 2).

Deterministic Float32 MLP; checks against an independent CPU reference that never touches the backend.
Modes (--mode):
  train        different-seed constructor sync (parameters must equal rank 0's), --steps SGD steps with
               different inputs per rank, DDP gradients vs reference mean-of-local-gradients, updated
               parameters vs reference; with --bucket-cap-mb small this also exercises >= 2 buckets
  buffers      registered buffer that differs between ranks; forward must propagate rank 0's
  (--layers N --width W selects a deep Linear stack instead of the 3-layer MLP; with --json it records, per
   step, application-side stamps plus every gradient-ready time, for examples/analyze_ddp_trace.py)
  count-mismatch  rank 1 has an extra parameter; DDP construction must fail on BOTH ranks, not hang
  shape-mismatch  rank 1 has a differently shaped parameter; PyTorch verifies shapes against rank 0's
                  broadcast metadata, so the NON-root rank must fail promptly (rank 0 may construct)
Launch like cross_host_allreduce.py (MASTER_ADDR is the rank-0 host).
"""
import argparse
import faulthandler
import os
import datetime
import json
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

import torch_tbccl
from torch.nn.parallel import DistributedDataParallel as DDP

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--devices", default=None, help="per-rank devices, e.g. cuda,cpu (overrides --device)")
p.add_argument("--mode", default="train", choices=["train", "buffers", "count-mismatch", "shape-mismatch"])
p.add_argument("--steps", type=int, default=20)
p.add_argument("--bucket-cap-mb", type=float, default=25.0)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--timeout", type=float, default=60.0)
p.add_argument("--layers", type=int, default=0, help="deep Linear stack (bucket matrix model); 0 = small MLP")
p.add_argument("--width", type=int, default=512)
p.add_argument("--warmup", type=int, default=0, help="leading steps excluded from the analysis")
p.add_argument("--stages", action="store_true", help="print bring-up stage markers (cross-host order)")
p.add_argument("--json", default=None, help="write the trace + per-step summary here (needs TORCH_TBCCL_TRACE=1)")
args = p.parse_args()
if os.environ.get("DDP_HANG_DUMP"):  # debugging aid: dump all Python stacks if still running after N seconds
    faulthandler.dump_traceback_later(float(os.environ["DDP_HANG_DUMP"]), exit=True)

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=args.timeout))
rank = dist.get_rank()
dev = torch.device(args.devices.split(",")[rank] if args.devices else args.device)
log = lambda *a: print(f"rank {rank}:", *a, flush=True)  # noqa: E731
stage = (lambda s: log("stage:", s)) if args.stages else (lambda s: None)


class MLP(nn.Module):
    def __init__(self, hidden=128, extra_param=False, wide=False):
        super().__init__()
        h2 = hidden + 2 if wide else hidden
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, h2)
        self.fc3 = nn.Linear(h2, 10)
        self.extra = nn.Parameter(torch.zeros(3)) if extra_param else None
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x):
        y = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
        return y + 0 * self.scale.sum()


class DeepMLP(nn.Module):
    """Synthetic bucket-matrix model: `layers` Linear(width, width) (~width^2*4 bytes of gradient each)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(args.width, args.width) for _ in range(args.layers))
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x + 0 * self.scale.sum()


def make(seed, **kw):
    torch.manual_seed(seed)
    return DeepMLP() if args.layers else MLP(args.hidden, **kw)


def batch(step, r):
    g = torch.Generator().manual_seed(5000 + step * 10 + r)
    if args.layers:
        return torch.randn(64, args.width, generator=g), torch.randn(64, args.width, generator=g)
    return torch.randn(32, 64, generator=g), torch.randn(32, 10, generator=g)


def local_grads(model, step, r):
    model.zero_grad()
    x, y = batch(step, r)
    nn.functional.mse_loss(model(x), y).backward()
    return [q.grad.clone() for q in model.parameters()]


def expect_failure(build):
    try:
        build()
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:300]
        log("DDP construction failed as expected:", type(e).__name__, msg)
        return True
    return False


SEED = {0: 1000, 1: 2000}
bad = 0

if args.mode in ("count-mismatch", "shape-mismatch"):
    kw = {"count-mismatch": {"extra_param": True}, "shape-mismatch": {"wide": True}}[args.mode]
    model = make(SEED[rank], **(kw if rank == 1 else {})).to(dev)
    t0 = time.monotonic()
    ok = expect_failure(lambda: DDP(model, device_ids=[0] if dev.type == "cuda" else None,
                                    bucket_cap_mb=args.bucket_cap_mb))
    must_fail = args.mode == "count-mismatch" or rank == 1
    log(f"{args.mode}: {'OK' if ok or not must_fail else 'DDP CONSTRUCTED (unexpected)'} after {time.monotonic() - t0:.2f}s"
        + ("" if ok else " (constructed; rank 0 does not verify shapes)"))
    sys.stdout.flush()
    # a failed construction leaves the group in an undefined collective state; do not reuse it
    sys.exit(0 if ok or not must_fail else 1)

model = make(SEED[rank]).to(dev)
stage("DDP construction")
ddp = DDP(model, device_ids=[0] if dev.type == "cuda" else None, bucket_cap_mb=args.bucket_cap_mb)
stage("DDP constructed")
ref = make(SEED[0])  # rank 0's initial weights, CPU, independent of the backend
ok = all(torch.equal(a.cpu(), b) for a, b in zip(model.parameters(), ref.parameters()))
log("constructor sync (parameters == rank 0's seed):", "OK" if ok else "MISMATCH")
bad += not ok

if args.mode == "buffers":
    with torch.no_grad():
        model.scale.fill_(7.0 if rank == 0 else float(100 + rank))
    x, _ = batch(0, rank)
    ddp(x.to(dev))
    ok = torch.equal(model.scale.cpu(), torch.full((4,), 7.0))
    log("forward buffer broadcast (rank 0's value everywhere):", "OK" if ok else f"MISMATCH {model.scale.tolist()}")
    bad += not ok
    dist.destroy_process_group()
    sys.exit(1 if bad else 0)

opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)
max_grad_err = max_param_err = 0.0
step_ms = []
now = torch_tbccl.trace_now_ns
ready = []   # (step, param index, ns): when each gradient was accumulated (diagnostic; no synchronization)
steps_rec = []
cur_step = [0]
for i, q in enumerate(model.parameters()):
    q.register_post_accumulate_grad_hook(lambda _q, i=i: ready.append((cur_step[0], i, now())))
for step in range(args.steps):
    if step == 0:
        stage("forward")
    t0 = time.monotonic()
    cur_step[0] = step
    x, y = batch(step, rank)
    opt.zero_grad()
    t_start = now()
    loss = nn.functional.mse_loss(ddp(x.to(dev)), y.to(dev))
    t_fwd = now()
    if step == 0:
        stage("backward")
    loss.backward()
    t_bwd = now()
    if step == 0:
        stage("optimizer step")
    got = [q.grad.detach().cpu().clone() for q in model.parameters()]
    opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    step_ms.append((time.monotonic() - t0) * 1e3)
    steps_rec.append({"step": step, "start_ns": t_start, "fwd_end_ns": t_fwd, "bwd_end_ns": t_bwd, "end_ns": now()})

    # reference: mean of the two ranks' local gradients, computed locally on CPU
    g0, g1 = local_grads(ref, step, 0), local_grads(ref, step, 1)
    for q, a, b in zip(ref.parameters(), g0, g1):
        q.grad = (a + b) / 2
    ref_opt.step()
    gerr = max((a - (b + c) / 2).abs().max().item() for a, b, c in zip(got, g0, g1))
    perr = max((a.cpu() - b).abs().max().item() for a, b in zip(model.parameters(), ref.parameters()))
    max_grad_err, max_param_err = max(max_grad_err, gerr), max(max_param_err, perr)
    finite = torch.isfinite(loss).item() and all(torch.isfinite(g).all().item() for g in got)
    if not finite or gerr > 1e-5 or perr > 1e-4:
        log(f"step {step}: FAIL finite={finite} grad_err={gerr:.2e} param_err={perr:.2e}")
        bad += 1
    if step == 0:
        stage("first step done")
log(f"{args.steps} steps: max|grad - reference| = {max_grad_err:.2e}, max|param - reference| = {max_param_err:.2e}",
    "OK" if not bad else "FAIL")

if torch_tbccl.trace_enabled():
    ev = torch_tbccl.trace_events()
    log("collectives recorded:", {k: sum(e["op"] == k for e in ev) for k in sorted({e["op"] for e in ev})})
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rank": rank, "device": str(dev), "bucket_cap_mb": args.bucket_cap_mb, "layers": args.layers,
                       "width": args.width, "warmup": args.warmup, "step_ms": step_ms, "steps": steps_rec,
                       "grad_ready": ready, "events": ev}, fh)

dist.destroy_process_group()
sys.exit(1 if bad else 0)
