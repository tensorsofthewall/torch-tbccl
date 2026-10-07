"""The MPS adapter work two-rank workload for accelerator <-> accelerator communication (CUDA <-> MPS physically; CPU <-> MPS locally). Exact-value checks first, a few timings second.
Launch with torchrun on each host (static c10d rendezvous, one process per host), exactly like tools/physical_ops.py; locally one `torchrun --standalone --nproc-per-node 2`
(add --local-addr 127.0.0.1 if the host name does not resolve).

    --devices D0,D1      the device of rank 0 and rank 1 (cpu | cuda | mps), e.g. cuda,mps (Linux rank 0 on CUDA, Mac rank 1 on MPS) or mps,cuda (reversed)
    --steps S[,S...]     p2p_one (rank 0 -> rank 1), p2p_rev (rank 1 -> rank 0), p2p_both (simultaneous isend/irecv), mixed (the ordering-domain repair work: async all_reduce + isend/irecv in flight together, same and opposite relative order), allreduce (float32 4 KiB/1 MiB/16 MiB, then float16
                         and bfloat16 1 MiB at world size 2), broadcast, all_gather. Collectives and P2P are never in flight together: each step waits for its Works.

Sizes 4 KiB / 1 MiB / 16 MiB, a few repetitions, every received payload bit-compared; after every operation the local tensor must still be on this rank's device. No saturation loop.
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
p.add_argument("--devices", default="cpu,cpu")
p.add_argument("--steps", default="p2p_one,p2p_rev,p2p_both,allreduce,broadcast,all_gather")
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--json", default=None)
a = p.parse_args()

dist.init_process_group("tbccl", timeout=timedelta(seconds=120))
rank, peer = dist.get_rank(), 1 - dist.get_rank()
assert dist.get_world_size() == 2
names = a.devices.split(",")
dev = torch.device(names[rank])
SIZES = {"4KiB": 1 << 10, "1MiB": 1 << 18, "16MiB": 1 << 22}  # float32 elements
out = {"rank": rank, "device": str(dev), "peer_device": names[peer], "steps": {}, "ok": True, "mismatches": []}


def pattern(sender, n, dtype=torch.float32, k=0):
    return ((torch.arange(n, dtype=torch.int64) * 7 + sender * 13 + k * 3) % 101).to(dtype)


def check(cond, what):
    if not cond:
        out["ok"] = False
        out["mismatches"].append(what)
        print(f"rank {rank} MISMATCH: {what}", flush=True)


def on_device(t, what):
    check(t.device.type == dev.type, f"{what}: tensor is on {t.device}, expected {dev.type}")  # no public CPU staging result


def bits_equal(got, want):
    return torch.equal(got.cpu().contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))


def timed(f):
    ts = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 3)


def p2p_one(src, tag):
    res = {}
    for label, n in SIZES.items():
        def once():
            if rank == src:
                dist.send(pattern(src, n, k=1).to(dev), dst=peer)
            else:
                b = torch.zeros(n, device=dev)
                dist.recv(b, src=src)
                on_device(b, f"recv {label}")
                check(bits_equal(b, pattern(src, n, k=1)), f"{tag} {label} payload from rank {src}")
        res[label] = {"bytes": n * 4, "ms": timed(once)}
    return res


def p2p_both():
    res = {}
    for label, n in SIZES.items():
        def once():
            mine, theirs = pattern(rank, n, k=2).to(dev), torch.zeros(n, device=dev)
            ws = [dist.isend(mine, dst=peer), dist.irecv(theirs, src=peer)]
            for w in ws:
                w.wait()
            on_device(theirs, f"irecv {label}")
            check(bits_equal(theirs, pattern(peer, n, k=2)), f"simultaneous {label} payload from rank {peer}")
        res[label] = {"bytes": n * 4, "ms": timed(once)}
    return res


def allreduce():
    res = {}
    cases = [("f32_" + k, torch.float32, n) for k, n in SIZES.items()] + [("f16_1MiB", torch.float16, 1 << 19), ("bf16_1MiB", torch.bfloat16, 1 << 19)]
    for label, dtype, n in cases:
        def once():
            x = (pattern(0, n) % 5 + rank).to(dtype).to(dev)  # 0..5: exact in every dtype, and so is the sum
            dist.all_reduce(x)
            on_device(x, f"all_reduce {label}")
            check(torch.equal(x.cpu().to(torch.float64), ((pattern(0, n) % 5) * 2 + 1).to(torch.float64)), f"all_reduce {label}")
        res[label] = {"ms": timed(once)}
    return res


def mixed():
    """An async fp32 all_reduce and a simultaneous isend/irecv pair in flight together on one group, 4 KiB and 1 MiB, with the SAME relative order on both ranks and
    with the two OPPOSITE orders (rank 0 collective first / rank 1 P2P first, then the mirror). Collectives and P2P are independent ordering domains in libtbccl wire 4."""
    res = {}
    for label, n in (("4KiB", 1 << 10), ("1MiB", 1 << 18)):
        for order in ("same", "opposite", "mirror"):
            def once():
                x = (pattern(0, n) % 5 + rank).to(dev)
                mine, theirs = pattern(rank, n, k=7).to(dev), torch.zeros(n, device=dev)
                coll_first = {"same": True, "opposite": rank == 0, "mirror": rank == 1}[order]
                works = []
                p2p = lambda: works.extend([dist.irecv(theirs, src=peer), dist.isend(mine, dst=peer)])  # noqa: E731
                if coll_first:
                    works.append(dist.all_reduce(x, async_op=True))
                    p2p()
                else:
                    p2p()
                    works.append(dist.all_reduce(x, async_op=True))
                for w in works:
                    w.wait()
                on_device(x, f"mixed all_reduce {label} {order}")
                on_device(theirs, f"mixed recv {label} {order}")
                check(torch.equal(x.cpu(), (pattern(0, n) % 5) * 2 + 1), f"mixed all_reduce {label} {order}")
                check(bits_equal(theirs, pattern(peer, n, k=7)), f"mixed P2P payload {label} {order}")
            res[f"{label}_{order}"] = {"ms": timed(once)}
    return res


def broadcast():
    res = {}
    for root in (0, 1):
        def once():
            n = 1 << 18
            b = pattern(root, n, k=4).to(dev) if rank == root else torch.zeros(n, device=dev)
            dist.broadcast(b, src=root)
            on_device(b, f"broadcast root {root}")
            check(bits_equal(b, pattern(root, n, k=4)), f"broadcast from root {root}")
        res[f"root{root}_1MiB"] = {"ms": timed(once)}
    return res


def all_gather():
    def once():
        n = 1 << 18
        outs = [torch.zeros(n, device=dev) for _ in range(2)]
        dist.all_gather(outs, pattern(rank, n, k=5).to(dev))
        for r in range(2):
            on_device(outs[r], "all_gather")
            check(bits_equal(outs[r], pattern(r, n, k=5)), f"all_gather slot {r}")
    return {"1MiB": {"ms": timed(once)}}


STEPS = {"p2p_one": lambda: p2p_one(0, "p2p_one"), "p2p_rev": lambda: p2p_one(1, "p2p_rev"), "p2p_both": p2p_both, "allreduce": allreduce, "broadcast": broadcast, "all_gather": all_gather, "mixed": mixed}
for s in a.steps.split(","):
    dist.barrier()  # separates the steps: every earlier Work is complete here
    out["steps"][s] = STEPS[s]()
    print(f"rank {rank} {s}: {out['steps'][s]}", flush=True)
dist.barrier()
dist.destroy_process_group()
print("RESULT " + json.dumps(out), flush=True)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
sys.exit(0 if out["ok"] else 1)
