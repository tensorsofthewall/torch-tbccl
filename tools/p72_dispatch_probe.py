"""Phase 72 milestone probe: every supported c10d op on an MPS tensor must reach ProcessGroupTBCCL (not fail in the c10d dispatcher). One process per rank.
Prints one line per op with the exception text; with the MPS adapter in place the ops succeed."""
import os
import sys

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl")
rank = dist.get_rank()
dev = "mps" if int(os.environ.get("MPS_RANK", rank)) == rank else "cpu"
x = torch.ones(16, device=dev)
ops = {
    "send": lambda: dist.send(x, 1 - rank) if rank == 0 else None,
    "recv": lambda: dist.recv(x, 1 - rank) if rank == 1 else None,
    "broadcast": lambda: dist.broadcast(x, 0),
    "all_reduce": lambda: dist.all_reduce(x),
    "all_gather": lambda: dist.all_gather([torch.empty_like(x) for _ in range(2)], x),
    "gather": lambda: dist.gather(x, [torch.empty_like(x) for _ in range(2)] if rank == 0 else None, dst=0),
    "barrier": lambda: dist.barrier(device_ids=None),
}
for name, fn in ops.items():
    try:
        fn()
        print(f"rank{rank} {dev} {name}: OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"rank{rank} {dev} {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)
    if "MPS adapter not implemented" in str(sys.exc_info()[1] or ""):
        pass
os._exit(0)
