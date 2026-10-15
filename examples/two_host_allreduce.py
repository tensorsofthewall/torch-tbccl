"""First two-host check: a SUM all_reduce over TBCCL, one process per machine.

Each rank contributes rank + 1, so with two ranks every rank must end with 3.0. A rank prints
`rank <r>: backend=tbccl result=3.0 PASS` and exits 0 only if the backend really is tbccl and the sum is right.
The whole run is bounded by --timeout seconds; nothing here changes any network setting.

Rank 0 on the machine whose Thunderbolt address is 192.168.77.1:
  TBCCL_LOCAL_ENDPOINT=192.168.77.1:0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank 0 \
      --local-addr 192.168.77.1 --master-addr 192.168.77.1 --master-port 29500 --rdzv-conf timeout=60 two_host_allreduce.py
Rank 1 on the other machine (192.168.77.2):
  TBCCL_LOCAL_ENDPOINT=192.168.77.2:0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank 1 \
      --local-addr 192.168.77.2 --master-addr 192.168.77.1 --master-port 29500 --rdzv-conf timeout=60 two_host_allreduce.py
"""
import argparse
import datetime
import socket
import sys

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401  registers the "tbccl" backend


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="tensor device; start with cpu")
    ap.add_argument("--elements", type=int, default=1, help="float32 elements per tensor; use 1048576 (4 MiB) to make the traffic visible in interface counters")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds allowed for rendezvous and for the collective")
    args = ap.parse_args(argv)
    timeout = datetime.timedelta(seconds=args.timeout)

    dist.init_process_group("tbccl", timeout=timeout)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        backend = dist.get_backend()
        print(f"rank {rank}: host={socket.gethostname()} world={world} backend={backend} "
              f"torch_tbccl={torch_tbccl.__version__} tbccl_runtime={torch_tbccl.runtime_version()}", flush=True)
        x = torch.full((args.elements,), rank + 1.0, device=args.device)
        work = dist.all_reduce(x, async_op=True)
        work.wait(timeout)
        result = x[0].item()
        expected = world * (world + 1) / 2
        ok = backend == "tbccl" and bool((x == expected).all())
        print(f"rank {rank}: backend={backend} result={result:.1f} {'PASS' if ok else f'FAIL (expected {expected:.1f} on tbccl)'}", flush=True)
        return 0 if ok else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
