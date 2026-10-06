"""Characterization (not a test of a supported feature): a collective and a point-to-point operation IN FLIGHT AT THE SAME TIME on one group.

TBCCL's contract: collectives and P2P are independent ordering domains over one peer lane, so they must not overlap on one communicator. This shows what the unsupported
overlap does (a bounded structured error, not a hang and not silent corruption) next to the supported pattern (wait in between).

    python tools/p71_mixed_inflight.py            # launches 2 ranks on loopback, prints one line per scenario
"""
import os
import subprocess
import sys
import time
from datetime import timedelta

MODES = ["overlap_collective_then_p2p", "overlap_p2p_then_collective", "serialized_collective_then_p2p"]


def child():
    import torch
    import torch.distributed as dist

    import torch_tbccl  # noqa: F401

    mode = os.environ["MODE"]
    dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
    rank, peer = dist.get_rank(), 1 - dist.get_rank()
    n = 1 << 24
    a, b = torch.full((n,), float(rank + 1)), torch.zeros(n)
    c = torch.full((n,), float(rank + 1))
    t0 = time.monotonic()
    try:
        if mode == "overlap_collective_then_p2p":
            ws = [dist.all_reduce(c, async_op=True), dist.isend(a, peer), dist.irecv(b, peer)]
        elif mode == "overlap_p2p_then_collective":
            ws = [dist.isend(a, peer), dist.irecv(b, peer), dist.all_reduce(c, async_op=True)]
        else:
            w = dist.all_reduce(c, async_op=True)
            w.wait()
            ws = [dist.isend(a, peer), dist.irecv(b, peer)]
        for w in ws:
            w.wait()
        c_ok, b_ok = bool((c == 3.0).all()), bool((b == float(peer + 1)).all())
        bad_c, bad_b = int((c != 3.0).sum()), int((b != float(peer + 1)).sum())
        print(f"rank {rank} {mode}: completed in {time.monotonic() - t0:.2f}s all_reduce_correct={c_ok} (wrong elements {bad_c}) p2p_correct={b_ok} (wrong elements {bad_b})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"rank {rank} {mode}: error after {time.monotonic() - t0:.2f}s: {str(e)[:170]}", flush=True)
    os._exit(0)


def main():
    for mode in MODES:
        with __import__("socket").socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        procs = [subprocess.Popen([sys.executable, __file__, "--child"], env=dict(os.environ, MODE=mode, RANK=str(r), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                                 TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", OMP_NUM_THREADS="2"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(2)]
        for p in procs:
            try:
                out, _ = p.communicate(timeout=90)
                print(out.strip())
            except subprocess.TimeoutExpired:
                p.kill()
                print(f"{mode}: HANG (killed after 90 s)")


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
