"""Phase 72: PyTorch MPS tensors through ProcessGroupTBCCL, one scenario per TEST_MODE. Two ranks on loopback: rank MPS_RANK (default 1) holds MPS tensors,
the other rank holds CPU tensors (CPU<->MPS; MPS<->MPS on one GPU is not required). Every payload is bit-compared.

  p2p          blocking send/recv in both directions, then simultaneous isend/irecv, 4 KiB / 1 MiB / 16 MiB float32
  dtypes       byte transport (send/recv, broadcast, all_gather) for every dtype MPS can create; reductions for the dtypes TBCCL reduces; rejected dtypes
  offset       a contiguous slice with a non-zero storage offset is sent / received / reduced; the rest of the backing allocation is untouched
  lifetime     the tensor is dropped (and gc'd) right after isend / irecv: the Work keeps the Metal storage alive
  noncontig    transposed and strided MPS tensors raise the same 'must be contiguous' error as CPU/CUDA, nothing is communicated
  collectives  broadcast (both roots), all_reduce SUM, all_gather, gather, barrier at three sizes; results stay on their device and are usable by the next MPS kernel
"""
import gc
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

mode = os.environ["TEST_MODE"]
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
peer = 1 - rank
MPS_RANK = int(os.environ.get("MPS_RANK", "1"))
dev = torch.device("mps" if rank == MPS_RANK else "cpu")
SIZES = {"4KiB": 4 << 10, "1MiB": 1 << 20, "16MiB": 16 << 20}


def raw(sender, nbytes, tag=0):
    g = torch.Generator().manual_seed(1000 * sender + 7 * tag + nbytes % 997)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)


def payload(sender, nbytes, dtype=torch.float32, tag=0):
    """`nbytes` of deterministic bytes viewed as `dtype` (NaN patterns included: only bits are compared)."""
    return raw(sender, nbytes, tag).view(dtype)


def same_bits(got, want):
    return torch.equal(got.cpu().contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))


def check_device(t):
    assert t.device.type == dev.type, f"tensor moved to {t.device}"


def p2p():
    for name, nb in SIZES.items():
        for src in (0, 1):  # blocking, both directions
            if rank == src:
                dist.send(payload(rank, nb, tag=1).to(dev), dst=peer)
            else:
                b = torch.zeros(nb // 4, device=dev)
                dist.recv(b, src=src)
                check_device(b)
                assert same_bits(b, payload(src, nb, tag=1)), f"{name} blocking {src}->{rank}"
        for rep in range(3):  # simultaneous
            b = torch.zeros(nb // 4, device=dev)
            ws = [dist.isend(payload(rank, nb, tag=2 + rep).to(dev), dst=peer), dist.irecv(b, src=peer)]
            for w in ws:
                w.wait()
            check_device(b)
            assert same_bits(b, payload(peer, nb, tag=2 + rep)), f"{name} simultaneous {rep}"
    # visibility: recv into an MPS tensor then read it with an MPS kernel only
    n = 1 << 18
    if rank == MPS_RANK:
        b = torch.zeros(n, device=dev)
        dist.recv(b, src=peer)
        out = (b * 2 + 1).cpu()
        want = torch.arange(n, dtype=torch.float32) * 2 + 1
        assert torch.equal(out, want), "MPS kernel did not see the received data"
    else:
        dist.send(torch.arange(n, dtype=torch.float32).to(dev), dst=peer)


ALL = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8, torch.bool, torch.complex64,
       torch.float64, getattr(torch, "float8_e4m3fn", None)]


def can_create(dtype):
    try:
        torch.zeros(4, device="mps", dtype=dtype)
        return True
    except Exception:  # noqa: BLE001
        return False


def dtypes():
    ok, unavailable = [], []
    for dt in [d for d in ALL if d is not None]:
        (ok if can_create(dt) else unavailable).append(dt)
    # every rank must run the same sequence: the MPS rank decides, and both create on CPU when MPS cannot
    for dt in ok:
        n = 4096 // torch.empty((), dtype=dt).element_size()
        mine = payload(rank, n * torch.empty((), dtype=dt).element_size(), dt, tag=3)
        # send/recv both ways
        for s in (0, 1):
            if rank == s:
                dist.send(mine.to(dev), dst=peer)
            else:
                b = torch.zeros(n, dtype=dt, device=dev)
                dist.recv(b, src=s)
                assert same_bits(b, payload(s, n * b.element_size(), dt, tag=3)), f"send/recv {dt} {s}->{rank}"
        # broadcast from rank 0, all_gather
        b = (mine if rank == 0 else torch.zeros(n, dtype=dt)).to(dev)
        dist.broadcast(b, src=0)
        assert same_bits(b, payload(0, n * b.element_size(), dt, tag=3)), f"broadcast {dt}"
        outs = [torch.zeros(n, dtype=dt, device=dev) for _ in range(2)]
        dist.all_gather(outs, mine.to(dev))
        assert all(same_bits(outs[r], payload(r, n * b.element_size(), dt, tag=3)) for r in range(2)), f"all_gather {dt}"
    # reductions: exact small integers
    for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.int8, torch.uint8, torch.float64):
        if dt not in ok:
            continue
        x = torch.full((256,), rank + 1, dtype=dt, device=dev)
        dist.all_reduce(x)
        check_device(x)
        assert torch.equal(x.cpu(), torch.full((256,), 3, dtype=dt)), f"all_reduce {dt}"
    for dt in (torch.int16, torch.bool):
        try:
            dist.all_reduce(torch.ones(8, dtype=dt, device=dev))
        except NotImplementedError as e:
            assert "has no reduction" in str(e), str(e)
        else:
            raise AssertionError(f"all_reduce accepted {dt}")
    if rank == MPS_RANK:
        print("MPS cannot create: " + ", ".join(str(d) for d in unavailable), flush=True)
        print("MPS dtypes moved: " + ", ".join(str(d) for d in ok), flush=True)


def offset():
    base_n = 2048
    lo, hi = 512, 1536
    ref = torch.arange(base_n, dtype=torch.float32)
    if rank == MPS_RANK:
        base = ref.clone().to(dev)
        x = base[lo:hi]
        assert x.is_contiguous() and x.storage_offset() == lo
        dist.send(x, dst=peer)  # only the slice goes out, not the start of the allocation
        # receive into a slice of a zeroed allocation: neighbours stay untouched
        into = torch.zeros(base_n, device=dev)
        dist.recv(into[lo:hi], src=peer)
        got = into.cpu()
        assert torch.equal(got[lo:hi], ref[lo:hi] + 5000) and not got[:lo].any() and not got[hi:].any(), "recv into a slice touched the rest of the allocation"
        # all_reduce on a slice
        y = torch.ones(base_n, device=dev)
        dist.all_reduce(y[lo:hi])
        got = y.cpu()
        assert torch.equal(got[lo:hi], torch.full((hi - lo,), 3.0)) and torch.equal(got[:lo], torch.ones(lo)) and torch.equal(got[hi:], torch.ones(base_n - hi))
    else:
        b = torch.zeros(hi - lo)
        dist.recv(b, src=peer)
        assert torch.equal(b, ref[lo:hi]), "the wrong region of the MPS allocation was sent"
        dist.send(ref[lo:hi] + 5000, dst=peer)
        y = torch.full((hi - lo,), 2.0)
        dist.all_reduce(y)


def lifetime():
    n = 4 << 20  # float32 elements: 16 MiB
    if rank == MPS_RANK:
        x = payload(rank, n * 4, tag=9).to(dev)
        w = dist.isend(x, dst=peer)
        del x
        gc.collect()
        w.wait()
        b = torch.zeros(n, device=dev)
        w2 = dist.irecv(b, src=peer)
        del b
        gc.collect()
        w2.wait()
        res = w2.result()
        assert same_bits(res[0], payload(peer, n * 4, tag=10)), "retained Work lost the received payload"
        res = None
    else:
        b = torch.zeros(n)
        dist.recv(b, src=peer)
        assert same_bits(b, payload(peer, n * 4, tag=9))
        dist.send(payload(rank, n * 4, tag=10), dst=peer)


def noncontig():
    t = torch.arange(16.0, device=dev).reshape(4, 4)
    for name, view in (("transpose", t.t()), ("strided slice", torch.arange(32.0, device=dev)[::2])):
        for what, fn in (
            ("all_reduce", lambda v=view: dist.all_reduce(v)),
            ("broadcast", lambda v=view: dist.broadcast(v, src=0)),
            ("send", lambda v=view: dist.send(v, dst=peer)),
            ("recv", lambda v=view: dist.recv(v, src=peer)),
            ("all_gather", lambda v=view: dist.all_gather([torch.empty_like(v), torch.empty_like(v)], v)),
        ):
            try:
                fn()
            except ValueError as e:
                assert "tensor must be contiguous" in str(e), str(e)
            else:
                raise AssertionError(f"{what} accepted a {name}")
    y = torch.ones(8, device=dev)
    dist.all_reduce(y)  # the group is still healthy and in step
    assert y[0].item() == 2.0


def collectives():
    for name, nb in SIZES.items():
        n = nb // 4
        x = torch.full((n,), float(rank + 1), device=dev)
        dist.all_reduce(x)
        check_device(x)
        assert torch.equal(x.cpu(), torch.full((n,), 3.0)), f"all_reduce {name}"
        assert ((x * 2).cpu() == 6.0).all(), "the next MPS kernel saw stale data after all_reduce"
        for root in (0, 1):
            b = payload(root, nb, tag=11).to(dev) if rank == root else torch.zeros(n, device=dev)
            dist.broadcast(b, src=root)
            check_device(b)
            assert same_bits(b, payload(root, nb, tag=11)), f"broadcast {name} root {root}"
        outs = [torch.zeros(n, device=dev) for _ in range(2)]
        dist.all_gather(outs, payload(rank, nb, tag=12).to(dev))
        assert all(same_bits(outs[r], payload(r, nb, tag=12)) for r in range(2)), f"all_gather {name}"
        for dst in (0, 1):
            outs = [torch.zeros(n, device=dev) for _ in range(2)] if rank == dst else None
            dist.gather(payload(rank, nb, tag=13).to(dev), gather_list=outs, dst=dst)
            if rank == dst:
                assert all(same_bits(outs[r], payload(r, nb, tag=13)) for r in range(2)), f"gather {name} dst {dst}"
    dist.barrier()
    # barrier addressed to the accelerator device (c10d::barrier / MPS)
    pg = dist.distributed_c10d._get_default_group()
    opts = dist.BarrierOptions()
    opts.device = dev
    pg.barrier(opts).wait()


globals()[mode]()
dist.barrier()
dist.destroy_process_group()
print(f"rank {rank} ok", flush=True)
sys.exit(0)
