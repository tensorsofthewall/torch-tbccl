"""PyTorch -> torch-tbccl -> TBCCL -> real link: does an async all_reduce overlap real CUDA compute?

Run one process per host with the same arguments (launch convention as cross_host_allreduce.py). The
CUDA rank is instrumented; the CPU rank only follows the same all_reduce schedule, so both ranks must be
given identical --sizes-mib / --fractions / --reps. Every timeline claim uses monotonic timestamps from
ONE process (the CUDA rank); the CPU rank's own timings are printed separately and never subtracted.

Per payload size:
  comm-only     all_reduce(async_op=True); wait()
  compute-only  independent matmul chain, synchronized only for measurement
  serial        all_reduce + wait, then compute
  overlap       all_reduce(async_op=True); compute; wait()    (compute never touches the reduced tensor)
Compute is calibrated to each fraction of the clean comm-only median, and serial/overlap are run in a
drift-resistant S,O,O,S order. Result verification happens outside the timed regions.
"""
import argparse
import datetime
import json
import statistics
import subprocess
import time

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--sizes-mib", default="4,16,25")
p.add_argument("--fractions", default="0.25,0.6,1.0")
p.add_argument("--reps", type=int, default=3, help="S,O,O,S blocks per fraction (6 serial + 6 overlap at 3)")
p.add_argument("--comm-reps", type=int, default=8)
p.add_argument("--matrix", type=int, default=2048)
p.add_argument("--out", default=None, help="JSON lines output (CUDA rank)")
p.add_argument("--timeout", type=float, default=120.0)
args = p.parse_args()

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=args.timeout))
rank = dist.get_rank()
dev = torch.device(args.device)
is_cuda = dev.type == "cuda"
sizes = [float(s) for s in args.sizes_mib.split(",")]
fractions = [float(f) for f in args.fractions.split(",")]
now = time.monotonic_ns
ms = lambda ns: ns / 1e6  # noqa: E731


def gpu_state():
    if not is_cuda:
        return None
    q = "temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,pstate,clocks_throttle_reasons.active,power.draw"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {"gpu": out.strip(), "compute_apps": [l for l in apps.strip().splitlines() if l]}


if is_cuda:
    a = torch.randn(args.matrix, args.matrix, device=dev)
    b = torch.randn(args.matrix, args.matrix, device=dev)
    c = torch.empty_like(a)

    def compute(n):
        for _ in range(n):
            torch.matmul(a, b, out=c)

    def timed_compute(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); compute(n); e.record(); e.synchronize()
        return s.elapsed_time(e)

    for _ in range(3):
        timed_compute(20)
else:
    compute = timed_compute = None


def fresh(x):
    x.fill_(1.0)
    if is_cuda:
        torch.cuda.synchronize()


def sync():
    if is_cuda:
        torch.cuda.synchronize()


def comm_only(x):
    fresh(x)
    t0 = now(); w = dist.all_reduce(x, async_op=True); w.wait(); return now() - t0


def serial(x, n):
    fresh(x)
    t0 = now(); w = dist.all_reduce(x, async_op=True); w.wait()
    if is_cuda:
        compute(n); torch.cuda.synchronize()
    return now() - t0


def overlap(x, n):
    fresh(x)
    done = []
    t0 = now()
    w = dist.all_reduce(x, async_op=True)
    t_ret = now()
    w.get_future().add_done_callback(lambda f: done.append(now()))
    if is_cuda:
        t_c0 = now(); compute(n)
        e = torch.cuda.Event(); e.record(); e.synchronize(); t_c1 = now()
    else:
        t_c0 = t_c1 = t_ret
    incomplete_at_compute_end = not w.is_completed()
    w.wait(); t_w = now()
    tw = done[0] if done else t_w  # callback timestamp = when the completion worker finished the Work
    return {
        "wall_ms": ms(t_w - t0), "submit_ms": ms(t_ret - t0), "compute_ms": ms(t_c1 - t_c0),
        "work_complete_after_return_ms": ms(tw - t_ret),
        "order": "work_done_before_compute_end" if tw < t_c1 else "work_done_after_compute_end",
        "work_incomplete_at_compute_end": incomplete_at_compute_end,
        "compute_started_before_work_complete": t_c0 < tw,
        "timeline_ok": t_ret < t_c0 < min(t_c1, tw) or t_ret < t_c0 < t_c1 < tw,
    }


def verify(x):
    fresh(x)
    dist.all_reduce(x)
    assert torch.equal(x.cpu(), torch.full_like(x.cpu(), 2.0)), "all_reduce result mismatch"


results = []
for size in sizes:
    count = int(size * 1024 * 1024 / 4)
    x = torch.ones(count, device=dev)
    gs0 = gpu_state()
    for _ in range(3):
        comm_only(x)  # warm-up (CUDA staging, sockets, allocator)
    comm = [comm_only(x) for _ in range(args.comm_reps)]
    comm_med = statistics.median(comm)
    verify(x)

    row = {"size_mib": size, "rank": rank, "comm_only_ms": [round(ms(v), 2) for v in comm], "comm_only_median_ms": round(ms(comm_med), 2),
           "gpu_before": gs0, "fractions": []}
    if is_cuda:
        unit = statistics.median([timed_compute(20) for _ in range(5)]) / 20  # ms per matmul
    for frac in fractions:
        n = max(1, round(frac * ms(comm_med) / unit)) if is_cuda else 0
        comp = [timed_compute(n) for _ in range(5)] if is_cuda else []
        ser, ovl, details = [], [], []
        for _ in range(args.reps):
            for kind in ("S", "O", "O", "S"):
                if kind == "S":
                    ser.append(ms(serial(x, n)))
                else:
                    d = overlap(x, n)
                    ovl.append(d["wall_ms"]); details.append(d)
        verify(x)
        f = {
            "fraction": frac, "matmuls": n,
            "compute_only_median_ms": round(statistics.median(comp), 2) if comp else None,
            "serial_ms": [round(v, 2) for v in ser], "overlap_ms": [round(v, 2) for v in ovl],
            "serial_median_ms": round(statistics.median(ser), 2), "overlap_median_ms": round(statistics.median(ovl), 2),
            "overlap_vs_serial": round(statistics.median(ovl) / statistics.median(ser), 3),
            "submit_median_ms": round(statistics.median(d["submit_ms"] for d in details), 2),
            "timeline_ok_reps": sum(d["timeline_ok"] for d in details),
            "work_incomplete_at_compute_end_reps": sum(d["work_incomplete_at_compute_end"] for d in details),
            "order_counts": {k: sum(d["order"] == k for d in details) for k in ("work_done_before_compute_end", "work_done_after_compute_end")},
            "example_timeline": details[0],
        }
        row["fractions"].append(f)
    row["gpu_after"] = gpu_state()
    results.append(row)
    s = {k: v for k, v in row.items() if k not in ("fractions", "gpu_before", "gpu_after")}
    print(f"rank {rank} size={size}MiB {json.dumps(s)}", flush=True)
    for f in row["fractions"]:
        print(f"rank {rank}   frac={f['fraction']} serial={f['serial_median_ms']}ms overlap={f['overlap_median_ms']}ms "
              f"ratio={f['overlap_vs_serial']} submit={f['submit_median_ms']}ms compute_only={f['compute_only_median_ms']}ms "
              f"incomplete_at_compute_end={f['work_incomplete_at_compute_end_reps']}/{len(f['overlap_ms'])} order={f['order_counts']}", flush=True)
    print(f"rank {rank}   gpu before: {gs0}\nrank {rank}   gpu after:  {row['gpu_after']}", flush=True)

if args.out and is_cuda:
    with open(args.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
dist.destroy_process_group()
