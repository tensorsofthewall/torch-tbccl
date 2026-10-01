"""One rank: verify blocking Float32 SUM allreduce over many shapes/rounds."""
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()


def contribution(r, n, round_):
    # Distinct per rank and per round; small integers => exact in float32.
    return (torch.arange(n, dtype=torch.float32) % 1000) * (r + 1) + (round_ * 7 + r * 100)


def check(n, round_, async_op):
    x = contribution(rank, n, round_)
    work = dist.all_reduce(x, async_op=async_op)
    if async_op:
        work.wait()
    expected = contribution(0, n, round_) + contribution(1, n, round_)
    assert torch.equal(x, expected), f"n={n} round={round_}: mismatch at {(x != expected).nonzero()[:3]}"


# the plan's literal example
x = torch.tensor([1.0, 2.0, 3.0]) if rank == 0 else torch.tensor([4.0, 5.0, 6.0])
dist.all_reduce(x)
assert x.tolist() == [5.0, 7.0, 9.0], x

for n in (1, 3, 255, 256, 1001, 262144):  # 1 elt, odd, ~1KiB, 1 MiB
    check(n, 0, False)
for r in range(20):  # repeated rounds, distinct data
    check(4099, r, r % 2 == 1)

# zero elements
z = torch.empty(0)
dist.all_reduce(z)
assert z.numel() == 0

# input aliasing: a view into a larger contiguous tensor
big = torch.zeros(64)
big[8:40] = contribution(rank, 32, 3)
dist.all_reduce(big[8:40])
assert torch.equal(big[8:40], contribution(0, 32, 3) + contribution(1, 32, 3))
assert big[:8].sum() == 0 and big[40:].sum() == 0  # neighbors untouched

dist.destroy_process_group()
print(f"rank {rank} ok")
