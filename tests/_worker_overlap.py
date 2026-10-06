"""Collectives and point-to-point transfers must not be in flight together on one group (TBCCL shares one connection per peer between the two ordering domains; the
overlap silently corrupts data). The adapter refuses it; waiting in between is the supported pattern."""
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank, peer = dist.get_rank(), 1 - dist.get_rank()
dev = torch.device(os.environ["DEVICE_RANK0"] if os.environ.get("DEVICE_RANK0") in ("cuda", "mps") and rank == 0 else "cpu")
n = 1 << 24  # 64 MiB of float32: cannot finish while the other rank has not joined


def expect_refused(fn, what):
    try:
        fn()
    except NotImplementedError as e:
        assert "torch-tbccl: unsupported operation" in str(e) and "must not overlap" in str(e), str(e)
        return
    raise AssertionError(f"{what}: the overlap was accepted")


if rank == 1:
    time.sleep(0.5)
x = torch.full((n,), float(rank + 1), device=dev)
coll = dist.all_reduce(x, async_op=True)
assert not coll.is_completed()
buf = torch.zeros(n, device=dev)
expect_refused(lambda: dist.isend(torch.ones(4, device=dev), peer), "isend while an all_reduce is in flight")
expect_refused(lambda: dist.irecv(buf, peer), "irecv while an all_reduce is in flight")
coll.wait()
assert torch.equal(x.cpu(), torch.full((n,), 3.0)), "the refused submissions disturbed the in-flight collective"

# point-to-point in flight, then a collective: refused; after waiting, everything works and the data is exact
if rank == 1:
    time.sleep(0.5)
a = torch.full((n,), float(rank + 1), device=dev)
b = torch.zeros(n, device=dev)
more, b2 = torch.full((n,), float(rank + 7), device=dev), torch.zeros(n, device=dev)
ws = [dist.isend(a, peer), dist.irecv(b, peer), dist.isend(more, peer), dist.irecv(b2, peer)]  # several P2P in flight together is fine
expect_refused(lambda: dist.all_reduce(torch.ones(4, device=dev)), "all_reduce while isend/irecv are in flight")
expect_refused(lambda: dist.barrier(), "barrier while isend/irecv are in flight")
for w in ws:
    w.wait()
assert torch.equal(b.cpu(), torch.full((n,), float(peer + 1))) and torch.equal(b2.cpu(), torch.full((n,), float(peer + 7)))

# the group is healthy afterwards: collective, then P2P, then collective, each waited for
y = torch.full((8,), float(rank + 1), device=dev)
dist.all_reduce(y)
assert y[0].item() == 3.0
if rank == 0:
    dist.send(torch.arange(5.0, device=dev), dst=1)
else:
    z = torch.zeros(5)
    dist.recv(z, src=0)
    assert torch.equal(z, torch.arange(5.0))
dist.barrier()
t0 = time.monotonic()
dist.destroy_process_group()
assert time.monotonic() - t0 < 10
print(f"rank {rank} ok")
sys.exit(0)
