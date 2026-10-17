#!/usr/bin/env python3
"""Standalone two-rank PyTorch communication benchmark on this machine (loopback), without the TBCCL benchmark suite.

    python standalone.py --op all_reduce --bytes 4096 [--backend tbccl|gloo] [--devices cpu,cpu] [--warmup 10] [--iterations 100] [--json]

It starts the two ranks of ``comm_bench.py`` as child processes with the environment ``torch.distributed`` needs, waits for them under a timeout and ends both if either fails or
the timeout expires; it only ever signals processes it started. The reported value is the one the shared suite reports: the per-iteration maximum over the two ranks for collectives, the source rank's time for point-to-point operations. This is a
smoke measurement on one machine: for two hosts, a physical link, remote cleanup and reports use the shared runner (``python -m benchmarks.suite`` from the tbccl checkout), or start
the same ``comm_bench.py`` on each host yourself (see the README in this directory). Values from different machines or configurations must not be compared without the runner's checks.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = "TBCCL_TORCH_BENCH"
# operation -> the per-iteration sample series that is the operation's completion time (the same series the suite reports)
SERIES = {"all_reduce": "completion_us", "broadcast": "completion_us", "all_gather": "completion_us", "p2p_ack": "completion_confirmed_us", "p2p_rtt": "round_trip_us",
          "all_reduce_async": "lifetime_us", "isend_irecv_ack": "completion_confirmed_us"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def terminate(procs, grace=3.0):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def run(args):
    port = free_port()
    procs, outputs = [], []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for rank in (0, 1):
                env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_")}
                env.update(RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
                if args.backend == "tbccl":
                    env["TBCCL_LOCAL_ENDPOINT"] = "127.0.0.1:0"
                argv = [sys.executable, str(HERE / "comm_bench.py"), "--backend", args.backend, "--devices", args.devices, "--op", args.op, "--bytes", str(args.bytes),
                        "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--timeout", str(args.timeout), "--expect-addresses", "127.0.0.1,127.0.0.1"]
                if args.backend == "gloo":
                    argv += ["--gloo-address", "127.0.0.1"]
                out = Path(tmp, f"rank{rank}.out")
                outputs.append(out)
                procs.append(subprocess.Popen(argv, env=env, stdout=open(out, "w"), stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL, start_new_session=True))
            deadline = time.monotonic() + args.timeout + 30
            while any(p.poll() is None for p in procs):
                if any(p.poll() not in (None, 0) for p in procs):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the ranks did not finish within {args.timeout + 30:.0f} s")
                time.sleep(0.05)
        finally:
            terminate(procs)
        errors = [p.stderr.read()[-300:] for p in procs]
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError("a rank failed: " + " | ".join(e.strip() for e in errors if e.strip()))
        lines = []
        for rank, out in enumerate(outputs):
            found = [json.loads(l[len(PREFIX) + 1:]) for l in out.read_text().splitlines() if l.startswith(PREFIX + " ")]
            if len(found) != 1:
                raise RuntimeError(f"rank {rank} printed {len(found)} result line(s), expected one")
            lines.append(found[0])
    return lines


SOURCE_RANK_ONLY = ("p2p_ack", "p2p_rtt", "isend_irecv_ack")  # timed on the sending rank; collectives report the per-iteration maximum over both ranks


def summarize(lines, op):
    series = SERIES[op]
    if op in SOURCE_RANK_ONLY:
        maxima = list(lines[0]["samples_us"][series])
    else:
        maxima = [max(a, b) for a, b in zip(*[r["samples_us"][series] for r in lines])]
    ordered = sorted(maxima)
    q = lambda f: ordered[min(len(ordered) - 1, int(round(f * (len(ordered) - 1))))]  # noqa: E731 - nearest rank; the suite's report uses interpolation
    return {"operation": op, "series": series, "samples": len(maxima), "median_us": statistics.median(maxima), "p95_us": q(0.95), "min_us": ordered[0], "max_us": ordered[-1],
            "verified": all(r.get("verify_ok") for r in lines), "backend": lines[0]["backend"], "devices": [lines[0]["device"], lines[1]["device"]], "bytes": lines[0]["bytes"],
            "environment": lines[0].get("environment")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--op", required=True, choices=sorted(SERIES))
    p.add_argument("--bytes", type=int, default=4096)
    p.add_argument("--backend", default="tbccl", choices=["tbccl", "gloo"])
    p.add_argument("--devices", default="cpu,cpu")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = p.parse_args(argv)
    try:
        lines = run(args)
    except (RuntimeError, TimeoutError) as error:
        print(f"standalone benchmark failed: {error}", file=sys.stderr)
        return 1
    s = summarize(lines, args.op)
    if not s["verified"]:
        print("standalone benchmark failed: the result did not verify on every rank; no measurement is reported", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print(f"{s['backend']} {s['operation']} {s['bytes']} B on {','.join(s['devices'])}: median {s['median_us']:.1f} us, p95 {s['p95_us']:.1f} us over {s['samples']} iterations "
              f"({'timed on the source rank' if args.op in SOURCE_RANK_ONLY else 'maximum of the two ranks per iteration'}; {s['series']}; data verified; loopback, one machine)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
