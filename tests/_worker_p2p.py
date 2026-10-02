"""One rank: dist.send/recv/isend/irecv through torch-tbccl. CUDA_RANKS lists ranks whose tensors live on the GPU."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
cuda_ranks = {int(r) for r in os.environ.get("CUDA_RANKS", "").split(",") if r}
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
peer = 1 - rank
dev = torch.device("cuda", 0) if rank in cuda_ranks else torch.device("cpu")


def pattern(dtype, n, seed):
    return (((torch.arange(n, dtype=torch.int64) * 7 + seed) % 100)).to(dtype)


if mode == "matrix":
    for dtype in (torch.uint8, torch.float16, torch.float32, torch.int64, torch.bfloat16):
        for n in (1, 3, 1024, 262144 + 5, 4 * 1024 * 1024 // 2):
            for sender in (0, 1):
                src = pattern(dtype, n, 3 + sender)
                if rank == sender:
                    dist.send(src.to(dev), dst=peer)
                else:
                    out = torch.zeros(n, dtype=dtype, device=dev)
                    dist.recv(out, src=sender)
                    assert torch.equal(out.cpu(), src), (dtype, n, sender)
    # async isend/irecv, many in flight, FIFO matching
    works, bufs = [], []
    for k in range(8):
        n = 1000 * (k + 1)
        if rank == 0:
            works.append(dist.isend(pattern(torch.float32, n, k).to(dev), dst=1))
        else:
            b = torch.zeros(n, device=dev)
            bufs.append((b, pattern(torch.float32, n, k)))
            works.append(dist.irecv(b, src=0))
    for w in works:
        w.wait()
    for b, ref in bufs:
        assert torch.equal(b.cpu(), ref)
    # zero-element
    z = torch.empty(0, device=dev)
    dist.send(z, dst=peer) if rank == 0 else dist.recv(z, src=peer)
    # CUDA sender on rank 0: delayed non-blocking producer, no host sync before send; consumer on another stream after recv
    if 0 in cuda_ranks:
        n = 1 << 16
        if rank == 0:
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                torch.cuda._sleep(300_000_000)
                x = torch.full((n,), 5.0, device=dev)
                dist.send(x, dst=1)
        else:
            out = torch.zeros(n, device=dev)
            dist.recv(out, src=0)
            if dev.type == "cuda":
                c = torch.cuda.Stream()
                with torch.cuda.stream(c):
                    y = out * 2
                c.synchronize()
                out = y
            assert torch.all(out.cpu() >= 5.0) and torch.all(out.cpu() % 5 == 0) and out.cpu()[0] in (5.0, 10.0)

elif mode == "blocked_abort":
    store = dist.distributed_c10d._get_default_store()
    if rank == 1:
        store.wait(["release"], timedelta(seconds=60))
        store.set("r1_done", "1")
    else:
        import time
        out = torch.zeros(1 << 16, device=dev)
        w = dist.irecv(out, src=1)
        time.sleep(0.3)
        assert not w.is_completed()
        t = time.monotonic()
        dist.distributed_c10d._abort_process_group()
        assert time.monotonic() - t < 5
        try:
            w.wait()
            raise SystemExit("recv should have failed")
        except RuntimeError as e:
            assert "abort" in str(e).lower(), e
        store.set("release", "1")
        store.wait(["r1_done"], timedelta(seconds=60))

print(f"rank {rank} ok", flush=True)
import sys; sys.stdout.flush(); os._exit(0)
