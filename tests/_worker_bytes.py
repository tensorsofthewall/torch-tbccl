"""One rank of an opaque-byte transport scenario (Phase 49): FP8 payloads, arbitrary random bytes and a packed-INT4 quantized bundle must cross
send/recv, broadcast and all_gather bit-exactly, with no dtype ever interpreted. Comparison is always on the underlying bytes (view(torch.uint8)),
never on floating-point values (FP8 NaN encodings and negative zero must survive too).

CUDA_RANKS (default none) lists the ranks whose tensors live on the GPU; the others use CPU. Optional dtypes are looked up with getattr, never assumed.
"""
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


def raw_bytes(n, seed):
    """n random bytes: every value 0..255 occurs, so every FP8 encoding (NaN, inf-like, subnormal, +/-0) is exercised."""
    g = torch.Generator().manual_seed(seed)
    head = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    body = torch.randint(0, 256, (max(n - 256, 0),), dtype=torch.uint8, generator=g)
    return torch.cat([head, body])[:n].clone()


# (name, torch dtype, bytes per element). 1-byte shell types: FP8 variants; float4_e2m1fn_x2 is two packed FP4 values per byte.
CANDIDATES = [
    "float8_e4m3fn", "float8_e5m2",  # the two formats the plan requires
    "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu", "float4_e2m1fn_x2",  # extras, only if this torch build has them
]
BYTE_DTYPES = [(n, getattr(torch, n)) for n in CANDIDATES if hasattr(torch, n)]
assert {"float8_e4m3fn", "float8_e5m2"} <= {n for n, _ in BYTE_DTYPES}, "this torch build lacks the FP8 dtypes the test needs"
BYTE_DTYPES += [("int8", torch.int8), ("uint8", torch.uint8), ("bfloat16", torch.bfloat16), ("float16", torch.float16), ("bool", torch.bool)]


def make(dtype, n_bytes, seed):
    b = raw_bytes(n_bytes, seed)
    if dtype == torch.bool:
        b = b % 2  # a bool tensor must hold 0/1
    esize = torch.empty(0, dtype=dtype).element_size()
    b = b[: (n_bytes // esize) * esize]
    return b.view(dtype)


def as_bytes(t):
    return t.cpu().contiguous().view(torch.uint8)


def other_like(t):
    """A destination buffer with the same dtype/shape whose every byte differs from `t` (no fill kernel needed: some shell dtypes have none)."""
    if t.dtype == torch.bool:
        return torch.logical_not(t)
    return (as_bytes(t) ^ 0x5A).view(t.dtype).reshape(t.shape)


def exact(a, b):
    return a.shape == b.shape and torch.equal(as_bytes(a), as_bytes(b))


SIZES = [1, 3, 17, 1021, 4099, (1 << 20) + 3]  # odd byte counts on purpose

if mode == "p2p":
    for name, dtype in BYTE_DTYPES:
        esize = torch.empty(0, dtype=dtype).element_size()
        for size in SIZES:
            if size < esize:
                continue
            for sender in (0, 1):
                src = make(dtype, size, 100 + sender)
                if rank == sender:
                    dist.send(src.to(dev), dst=1 - sender)
                else:
                    out = other_like(src).to(dev)
                    dist.recv(out, src=sender)
                    assert exact(out, src), f"{name}: send/recv from rank {sender} not byte exact (size {size})"

elif mode == "broadcast_gather":
    for name, dtype in BYTE_DTYPES:
        esize = torch.empty(0, dtype=dtype).element_size()
        for size in (1021, 4099, 262147):
            if size < esize:
                continue
            for root in (0, 1):
                src = make(dtype, size, 200 + root)
                x = (src if rank == root else other_like(src)).to(dev)
                dist.broadcast(x, src=root)
                assert exact(x, src), f"{name}: broadcast from root {root} not byte exact (size {size})"
            mine = make(dtype, size, 300 + rank)
            outs = [other_like(mine).to(dev) for _ in range(2)]
            dist.all_gather(outs, mine.to(dev))
            for r in range(2):
                assert exact(outs[r], make(dtype, size, 300 + r)), f"{name}: all_gather slot {r} not byte exact (size {size})"
    # arbitrary random bytes that belong to no dtype at all
    for size in (1, 2, 5, 4097, 1 << 20):
        blob = raw_bytes(size, 999)
        if rank == 0:
            dist.send(blob.to(dev), dst=1)
        else:
            got = torch.empty(size, dtype=torch.uint8, device=dev)
            dist.recv(got, src=0)
            assert torch.equal(got.cpu(), blob), f"random {size}-byte blob not exact"

elif mode == "bundle":
    # A representative AWQ/GPTQ-style quantized layer: packed 4-bit weights (two values per byte, low nibble first), per-group scales (fp16 / bf16)
    # and per-group zero points (uint8 / int8). TBCCL moves the three buffers independently and knows nothing about what they mean; shape metadata is
    # test harness knowledge.
    ROWS, COLS, GROUP = 64, 256, 32

    def make_bundle(seed, scale_dtype, zero_dtype):
        g = torch.Generator().manual_seed(seed)
        q = torch.randint(0, 16, (ROWS, COLS), dtype=torch.uint8, generator=g)  # 4-bit values
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()  # [ROWS, COLS / 2] bytes
        scales = (torch.rand(ROWS, COLS // GROUP, generator=g) * 0.1 + 0.01).to(scale_dtype)
        if zero_dtype == torch.uint8:
            zeros = torch.randint(0, 16, (ROWS, COLS // GROUP), dtype=torch.uint8, generator=g)
        else:
            zeros = torch.randint(-8, 8, (ROWS, COLS // GROUP), dtype=torch.int8, generator=g)
        return q, packed, scales, zeros

    def unpack(packed):
        return torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(packed.shape[0], -1)

    def dequant(packed, scales, zeros):
        z = zeros.repeat_interleave(GROUP, dim=1).to(torch.float32)
        s = scales.repeat_interleave(GROUP, dim=1).to(torch.float32)
        return (unpack(packed).to(torch.float32) - z) * s

    for scale_dtype in (torch.float16, torch.bfloat16):
        for zero_dtype in (torch.uint8, torch.int8):
            for sender in (0, 1):
                q, packed, scales, zeros = make_bundle(7 + sender, scale_dtype, zero_dtype)
                if rank == sender:
                    for t in (packed, scales, zeros):
                        dist.send(t.to(dev), dst=1 - sender)
                else:
                    rp = other_like(packed).to(dev)
                    rs = other_like(scales).to(dev)
                    rz = other_like(zeros).to(dev)
                    for t in (rp, rs, rz):
                        dist.recv(t, src=sender)
                    assert exact(rp, packed), "packed 4-bit weight bytes changed in transit"
                    assert exact(rs, scales), f"{scale_dtype} scale bytes changed in transit"
                    assert exact(rz, zeros), f"{zero_dtype} zero-point bytes changed in transit"
                    assert torch.equal(unpack(rp.cpu()), q), "unpacked 4-bit values differ from the source"
                    assert torch.equal(dequant(rp.cpu(), rs.cpu(), rz.cpu()), dequant(packed, scales, zeros)), "reconstruction differs from the source"
            # the whole bundle also survives broadcast and all_gather
            q, packed, scales, zeros = make_bundle(21, scale_dtype, zero_dtype)
            for t_src in (packed, scales, zeros):
                x = (t_src if rank == 1 else other_like(t_src)).to(dev)
                dist.broadcast(x, src=1)
                assert exact(x, t_src), "bundle member changed in broadcast"
                outs = [other_like(t_src).to(dev) for _ in range(2)]
                dist.all_gather(outs, t_src.to(dev))
                assert all(exact(o, t_src) for o in outs), "bundle member changed in all_gather"

elif mode == "reduction_rejected":
    # Opaque payload dtypes have no reduction: all_reduce must fail before any network activity (rank 1 never even calls it).
    import time

    for name, dtype in BYTE_DTYPES:
        if name in ("int8", "uint8", "bfloat16", "float16"):
            continue  # these four are reducible since Phase 49
        t0 = time.monotonic()
        try:
            dist.all_reduce(make(dtype, 64, 5).to(dev))
        except NotImplementedError as e:
            assert "has no reduction" in str(e), e
        else:
            raise AssertionError(f"all_reduce({name}) did not fail")
        assert time.monotonic() - t0 < 5, f"all_reduce({name}) failure was not prompt"
    x = torch.full((8,), float(rank + 1))
    dist.all_reduce(x.to(dev))  # group still healthy

else:
    raise SystemExit(f"unknown TEST_MODE {mode}")

dist.destroy_process_group()
print(f"rank {rank} ok")
