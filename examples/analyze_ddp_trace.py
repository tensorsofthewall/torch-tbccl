"""Summarize DDP bucket traces written by `ddp_train.py --json` (one file per rank).

Every number comes from ONE process's own monotonic stamps; files from different ranks are summarized
side by side and never subtracted from each other.

Per measured step (after --warmup and after the bucket layout has stabilized), for the AllReduce events of that step:
  submit_us        = return_ns - entry_ns            time the DDP hook was inside the backend call
  total_work_us    = complete_ns - entry_ns
  submit_fraction  = submit_us / total_work_us
  lower_bound_overlap_i = max(0, min(entry_{i+1}, complete_i) - return_i)
      = a LOWER BOUND, from application-visible events only, on how long bucket i's communication ran
        while autograd was still producing/launching the next bucket (not a GPU hardware trace)
  chained_ok_i     = return_i < entry_{i+1} < complete_i   (bucket i still active when i+1 launched)
"""
import argparse
import json
import statistics


def per_step(d):
    ev = [e for e in d["events"] if e["op"] == "allreduce"]
    out = []
    for st in d["steps"]:
        es = sorted((e for e in ev if st["start_ns"] <= e["entry_ns"] <= st["end_ns"]), key=lambda e: e["seq"])
        out.append((st, es))
    return out


def summarize(d, warmup):
    steps = per_step(d)
    layouts = [tuple(e["bytes"] for e in es) for _, es in steps]
    final = layouts[-1]
    stable_from = next(i for i in range(len(layouts)) if all(l == final for l in layouts[i:]))
    first = max(warmup, stable_from)
    rows, step_stats = [], []
    for st, es in steps[first:]:
        t0 = st["start_ns"]
        for i, e in enumerate(es):
            nxt = es[i + 1]["entry_ns"] if i + 1 < len(es) else None
            submit = (e["return_ns"] - e["entry_ns"]) / 1e3
            total = (e["complete_ns"] - e["entry_ns"]) / 1e3
            lb = max(0, min(nxt, e["complete_ns"]) - e["return_ns"]) / 1e3 if nxt is not None else None
            rows.append({"step": st["step"], "bucket": i, "bytes": e["bytes"], "submit_us": submit, "total_us": total,
                         "submit_fraction": submit / total if total else None, "lower_bound_overlap_us": lb,
                         "chained_ok": (e["return_ns"] < nxt < e["complete_ns"]) if nxt is not None else None,
                         "entry_rel_ms": (e["entry_ns"] - t0) / 1e6})
        ready = [r for r in d["grad_ready"] if r[0] == st["step"]]
        last_ready = max(r[2] for r in ready) if ready else None
        step_stats.append({"iter_ms": (st["end_ns"] - st["start_ns"]) / 1e6, "fwd_ms": (st["fwd_end_ns"] - st["start_ns"]) / 1e6,
                           "bwd_ms": (st["bwd_end_ns"] - st["fwd_end_ns"]) / 1e6,
                           "tail_after_last_grad_ms": (st["bwd_end_ns"] - last_ready) / 1e6 if last_ready else None,
                           "n_buckets": len(es), "bytes": sum(e["bytes"] for e in es)})
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    nb = [r for r in rows if r["lower_bound_overlap_us"] is not None]
    return {
        "rank": d["rank"], "device": d["device"], "bucket_cap_mb": d["bucket_cap_mb"],
        "stable_from_step": stable_from, "measured_steps": len(steps) - first,
        "buckets_per_step": med([s["n_buckets"] for s in step_stats]),
        "bucket_bytes_median": med([r["bytes"] for r in rows]), "bucket_bytes_total": med([s["bytes"] for s in step_stats]),
        "iter_ms_median": med([s["iter_ms"] for s in step_stats]), "fwd_ms_median": med([s["fwd_ms"] for s in step_stats]),
        "bwd_ms_median": med([s["bwd_ms"] for s in step_stats]),
        "tail_after_last_grad_ms_median": med([s["tail_after_last_grad_ms"] for s in step_stats if s["tail_after_last_grad_ms"] is not None]),
        "submit_us_median": med([r["submit_us"] for r in rows]), "total_work_us_median": med([r["total_us"] for r in rows]),
        "submit_fraction_median": med([r["submit_fraction"] for r in rows if r["submit_fraction"] is not None]),
        "submit_fraction_p90": sorted(r["submit_fraction"] for r in rows if r["submit_fraction"] is not None)[int(0.9 * (len(rows) - 1))] if rows else None,
        "lower_bound_overlap_us_median": med([r["lower_bound_overlap_us"] for r in nb]),
        "chained_ok_fraction": (sum(r["chained_ok"] for r in nb) / len(nb)) if nb else None,
        "entry_rel_ms_first_bucket_median": med([r["entry_rel_ms"] for r in rows if r["bucket"] == 0]),
        "entry_rel_ms_last_bucket_median": med([r["entry_rel_ms"] for r in rows if r["bucket"] == step_stats[0]["n_buckets"] - 1]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    for f in a.files:
        d = json.load(open(f))
        s = summarize(d, a.warmup if a.warmup is not None else d.get("warmup", 0))
        if a.json:
            print(json.dumps(s))
        else:
            print(f"{f}: " + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in s.items()))
