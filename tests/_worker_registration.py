"""The selected backend is really ProcessGroupTBCCL: nothing may substitute gloo or nccl behind backend="tbccl"."""
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

spec = sys.argv[1] if len(sys.argv) > 1 else "tbccl"
dist.init_process_group(spec, timeout=timedelta(seconds=60))
rank = dist.get_rank()
assert dist.get_backend() in ("tbccl", "cpu:tbccl"), dist.get_backend()
pg = dist.distributed_c10d._get_default_group()
backend = pg._get_backend(torch.device("cpu"))
assert backend.name() == "tbccl", backend.name()
assert "gloo" not in str(type(backend)).lower() and "nccl" not in str(type(backend)).lower(), type(backend)
# an operation only the tbccl backend rejects proves which implementation answered: gloo and nccl implement reduce
try:
    dist.reduce(torch.ones(2), dst=0)
except NotImplementedError as e:
    assert "torch-tbccl: unsupported operation: reduce" in str(e), e
else:
    raise AssertionError("reduce succeeded: a different backend answered")
dist.destroy_process_group()
print(f"rank {rank} ok")
