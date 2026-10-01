"""One rank: broadcast / all_gather through torch.distributed. CUDA_RANKS (default none) lists the
ranks whose tensors live on the GPU; the others use CPU."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
cuda_ranks = {int(r) for r in os.environ.get("CUDA_RANKS", "").split(",") if r}
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
dev = torch.device("cuda", 0) if rank in cuda_ranks else torch.device("cpu")


def pattern(dtype, n, seed):
    base = (torch.arange(n, dtype=torch.int64) * 7 + seed) % 100
    return base.to(dtype)


def to_dev(t):
    return t.to(dev, copy=True)


if mode == "broadcast":
    for dtype in (torch.float32, torch.int64, torch.float16, torch.uint8, torch.bool):
        for root in (0, 1):
            for n in (1, 5, 1000, 262144 + 3):
                src = pattern(dtype, n, 11 + root)
                x = to_dev(src if rank == root else torch.ones(n, dtype=dtype))
                dist.broadcast(x, src=root)
                assert torch.equal(x.cpu(), src), (dtype, root, n)
    # async + Future
    src = pattern(torch.float32, 4096, 3)
    x = to_dev(src if rank == 1 else torch.zeros(4096))
    w = dist.broadcast(x, src=1, async_op=True)
    fut = w.get_future()
    w.wait()
    fut.wait()
    assert torch.equal(x.cpu(), src)
    z = torch.empty(0)
    dist.broadcast(z.to(dev), src=0)

elif mode == "all_gather":
    for dtype, n in ((torch.int64, 1), (torch.float32, 300000), (torch.uint8, 7), (torch.float64, 33)):
        mine = pattern(dtype, n, 5 + rank)
        outs = [torch.zeros(n, dtype=dtype, device=dev) for _ in range(2)]
        dist.all_gather(outs, to_dev(mine))
        assert torch.equal(outs[0].cpu(), pattern(dtype, n, 5)), (dtype, n)
        assert torch.equal(outs[1].cpu(), pattern(dtype, n, 6)), (dtype, n)
    # the DDP parameter-count shape: int64 [N_rank] -> [N0, N1]
    counts = torch.tensor([1000 + rank * 17], dtype=torch.int64, device=dev)
    gathered = [torch.zeros(1, dtype=torch.int64, device=dev) for _ in range(2)]
    dist.all_gather(gathered, counts)
    assert [int(g.item()) for g in gathered] == [1000, 1017]
    # async
    outs = [torch.zeros(4096, device=dev) for _ in range(2)]
    w = dist.all_gather(outs, to_dev(pattern(torch.float32, 4096, rank)), async_op=True)
    w.wait()
    assert torch.equal(outs[1].cpu(), pattern(torch.float32, 4096, 1))

elif mode == "mixed_order":
    # broadcast, all_gather and all_reduce interleaved, async, must stay in issue order on both ranks
    works, checks = [], []
    for k in range(6):
        a = to_dev(torch.full((50000,), float(rank + 1 + k)))
        works.append(dist.all_reduce(a, async_op=True))
        checks.append(lambda a=a, k=k: torch.equal(a.cpu(), torch.full((50000,), float(3 + 2 * k))))
        b = to_dev(torch.full((1000,), 5.0 + k) if rank == 0 else torch.zeros(1000))
        works.append(dist.broadcast(b, src=0, async_op=True))
        checks.append(lambda b=b, k=k: torch.equal(b.cpu(), torch.full((1000,), 5.0 + k)))
        outs = [torch.zeros(10, dtype=torch.int64, device=dev) for _ in range(2)]
        works.append(dist.all_gather(outs, to_dev(torch.full((10,), rank + k, dtype=torch.int64)), async_op=True))
        checks.append(lambda o=outs, k=k: torch.equal(o[0].cpu(), torch.full((10,), k)) and torch.equal(o[1].cpu(), torch.full((10,), 1 + k)))
    for w in works:
        w.wait()
    assert all(c() for c in checks)

elif mode == "errors":
    # Argument errors must be raised promptly on BOTH ranks consistently (no collective is issued).
    def raises(exc, f):
        try:
            f()
        except exc:
            return
        raise AssertionError(f"expected {exc}")

    x = torch.zeros(4)
    raises(ValueError, lambda: dist.broadcast(torch.zeros(4, 2).t(), src=0))  # non-contiguous
    raises(ValueError, lambda: dist.all_gather([torch.zeros(3), torch.zeros(4)], torch.zeros(4)))  # numel mismatch
    raises(ValueError, lambda: dist.all_gather([torch.zeros(4, dtype=torch.int64), torch.zeros(4)], torch.zeros(4)))  # dtype
    raises((NotImplementedError, RuntimeError), lambda: dist.reduce(x, dst=0))
    raises((NotImplementedError, RuntimeError), lambda: dist.all_gather_into_tensor(torch.zeros(8), x))
    # still healthy afterwards
    y = torch.ones(3)
    dist.all_reduce(y)
    assert y.tolist() == [2.0, 2.0, 2.0]

print(f"rank {rank} ok")
dist.destroy_process_group()
