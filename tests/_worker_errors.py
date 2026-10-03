"""One rank: every invalid call must raise promptly (no hang) and leave the group usable."""
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
pg = dist.distributed_c10d._get_default_group()


def expect(exc, match, fn):
    t0 = time.monotonic()
    try:
        fn()
    except exc as e:
        assert match in str(e), f"wrong message: {e}"
    else:
        raise AssertionError(f"no error for {match!r}")
    assert time.monotonic() - t0 < 5, "error was not prompt"


f = lambda: torch.ones(8)  # noqa: E731
# float16/bfloat16/int8/uint8 are reducible since low-precision datatype; dtypes with no reduction arithmetic still fail promptly and name the
# supported set.
expect(NotImplementedError, "has no reduction", lambda: dist.all_reduce(torch.ones(8, dtype=torch.int16)))
expect(NotImplementedError, "has no reduction", lambda: dist.all_reduce(torch.ones(8, dtype=torch.bool)))
if hasattr(torch, "float8_e4m3fn"):
    expect(NotImplementedError, "has no reduction", lambda: dist.all_reduce(torch.zeros(8, dtype=torch.uint8).view(torch.float8_e4m3fn)))
# only SUM is mapped; the message names the dtype and the op
expect(NotImplementedError, "supported ops: SUM", lambda: dist.all_reduce(f(), op=dist.ReduceOp.MAX))
expect(NotImplementedError, "dtype=float32 op=MAX", lambda: dist.all_reduce(f(), op=dist.ReduceOp.MAX))
expect(NotImplementedError, "dtype=float16 op=PRODUCT", lambda: dist.all_reduce(torch.ones(8, dtype=torch.float16), op=dist.ReduceOp.PRODUCT))
expect(NotImplementedError, "supported ops: SUM", lambda: dist.all_reduce(f(), op=dist.ReduceOp.AVG))
expect(ValueError, "contiguous", lambda: dist.all_reduce(torch.ones(16)[::2]))
expect(ValueError, "contiguous", lambda: dist.all_reduce(torch.ones(4, 4).t()))
expect(ValueError, "exactly one tensor", lambda: pg.allreduce([f(), f()]))
expect(Exception, "", lambda: dist.all_reduce(torch.ones(4).to_sparse()))
expect(Exception, "does not support", lambda: dist.reduce(f(), dst=0))
expect(Exception, "does not support", lambda: dist.reduce_scatter_tensor(torch.ones(8), torch.ones(16)))
# meta tensors never reach the backend: c10d dispatches them to a no-op meta kernel.

# group still usable afterwards
x = torch.full((16,), float(rank + 1))
dist.all_reduce(x)
assert torch.equal(x, torch.full((16,), 3.0)), x

dist.destroy_process_group()
print(f"rank {rank} ok")
