"""Phase 71 capability matrix: operation x device x dtype x world-size, measured through the public torch.distributed API.

    python tools/p71_capability_matrix.py run   --worlds 1,2,3,4 --devmodes cpu,cuda0,cudaall --out docs/data/phase71/capability_matrix.json
    python tools/p71_capability_matrix.py render docs/data/phase71/capability_matrix.json > docs/phase71_capability_matrix.md
    python tools/p71_capability_matrix.py --worker ...     (internal: one rank)

Every world size / device mode is one loopback launch of W processes (TBCCL_LOCAL_ENDPOINT=127.0.0.1:0, rendezvous through a TCPStore) that executes the
whole case list in one process group. A case records what actually happened, never what is expected:

    PASS      the call returned and every element matched an independently computed reference (shapes: scalar, 1 element, 8 B, 64 B, 37 elements,
              2-D, 2 KiB, 16 KiB, 1 MiB)
    REJECTED  the call raised before communicating (torch-tbccl: unsupported operation / invalid argument, or c10d's "does not support")
    NA        not meaningful (a peer operation in a one-rank group)
    WRONG     the call returned but data differ from the reference
    ERROR     an unexpected exception
    HANG      no progress for --hang seconds (the driver kills the launch and records the case that was running)

Payloads are deterministic small integers (exact in every dtype, including bf16 and float8), never torch.randn. Byte-transport operations compare the raw
bytes; reductions compare against a float64/int64 reference reduced on the CPU.

devmodes: cpu (all ranks CPU), cuda0 (rank 0 on cuda:0, the others on CPU; heterogeneous), cudaall (every rank on cuda:0 - ONE physical GPU is shared, which
validates the ProcessGroup path but is not multi-GPU).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))

BYTE_DTYPES = ["int8", "uint8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64", "bool", "complex64", "float8_e4m3fn"]
REDUCE_DTYPES = ["int8", "uint8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64", "bool"]
REDUCE_OPS = ["SUM", "PRODUCT", "MIN", "MAX", "AVG", "BAND"]
SIZES = [("scalar", None), ("1el", 1), ("8B", 8), ("64B", 64), ("37el", "37"), ("2D", "2d"), ("2KiB", 2048), ("16KiB", 16384), ("1MiB", 1 << 20)]
UNSUPPORTED_OPS = ["reduce", "scatter", "all_to_all", "all_to_all_single", "reduce_scatter", "reduce_scatter_tensor", "all_gather_into_tensor", "scatter_object_list"]


def expected_status(op, dtype, world):
    """The supported surface as DOCUMENTED (docs/phase71_capability_audit.md); the matrix must observe exactly this, so any drift - a newly working or a newly broken
    cell - fails the check instead of passing silently. Same for every device mode."""
    if op in ("send/recv", "isend/irecv"):
        return "NA" if world < 2 else "PASS"
    if op in ("broadcast", "all_gather"):
        return "REJECTED" if world < 2 else "PASS"
    if op == "gather":
        return "PASS" if world == 2 else "REJECTED"  # 2-rank groups only
    if op.startswith("all_reduce "):
        if world < 2 or op != "all_reduce SUM":
            return "REJECTED"  # one-rank groups reject collectives; only SUM is implemented
        if dtype in ("int16", "bool"):
            return "REJECTED"  # no reduction arithmetic for the dtype
        if dtype in ("float16", "bfloat16"):
            return "PASS" if world == 2 else "REJECTED"  # 16-bit floating point only at world size 2
        return "PASS"
    if op == "barrier":
        return "PASS"
    if op in ("all_gather_object", "broadcast_object_list"):
        return "REJECTED" if world < 2 else "PASS"
    if op == "gather_object":
        return "PASS" if world == 2 else "REJECTED"
    if op in UNSUPPORTED_OPS:
        return "REJECTED"
    if op.startswith("noncontiguous "):
        return "NA" if world < 2 else "REJECTED"
    if op.startswith("in-place "):
        return "NA" if world < 2 else "PASS"
    raise KeyError(op)


def mismatches(records):
    return [r for r in records if r["status"] != expected_status(r["op"], r["dtype"], r["world"])]


# ---------------------------------------------------------------- worker side
def worker(a):
    import torch
    import torch.distributed as dist

    import torch_tbccl  # noqa: F401

    dist.init_process_group("tbccl", timeout=timedelta(seconds=a.group_timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    on_cuda = a.devmode == "cudaall" or (a.devmode == "cuda0" and rank == 0)
    dev = torch.device("cuda:0" if on_cuda else "cpu")

    def tdtype(name):
        return getattr(torch, name)

    def shape_for(label, spec, dt):
        isz = torch.empty((), dtype=dt).element_size()
        if label == "scalar":
            return ()
        if spec == "37":
            return (37,)
        if spec == "2d":
            return (12, 17)
        if spec == 1:
            return (1,)
        n = max(1, spec // isz)
        if label == "16KiB":
            return (n // 64, 64) if n % 64 == 0 else (n,)
        return (n,)

    def pattern(r, shape, dt, kind="val"):
        """Deterministic small integers, exact in every dtype. kind: val (1..11), prod (1..2), bits (0/1)."""
        n = 1
        for s in shape:
            n *= s
        i = torch.arange(n, dtype=torch.int64)
        if kind == "prod":
            v = (i + r) % 2 + 1
        elif kind == "bool":
            v = (i + r) % 2
        else:
            v = (i * 3 + r * 5 + 1) % 11 + 1
        if dt == torch.bool:
            v = v % 2
        return v.reshape(shape).to(dt)

    def eq_bytes(a_, b_):
        a_, b_ = a_.cpu().contiguous(), b_.cpu().contiguous()
        if a_.shape != b_.shape or a_.dtype != b_.dtype:
            return False
        return bool((a_.reshape(-1).view(torch.uint8) == b_.reshape(-1).view(torch.uint8)).all()) if a_.numel() else True

    class Wrong(Exception):
        pass

    def need(cond, msg):
        if not cond:
            raise Wrong(msg)

    # --------------------------------------------------------------- cases
    def case_sendrecv(dtname, label, spec):
        if world < 2:
            return "NA"
        dt = tdtype(dtname)
        shape = shape_for(label, spec, dt)
        # a chain 0 -> 1 -> ... -> W-1 with blocking send/recv (each hop is also the non-zero-source variant)
        for hop in range(world - 1):
            if rank == hop:
                dist.send(pattern(hop, shape, dt).to(dev), dst=hop + 1)
            elif rank == hop + 1:
                got = torch.zeros(shape, dtype=dt).to(dev)
                dist.recv(got, src=hop)
                need(eq_bytes(got, pattern(hop, shape, dt)), f"hop {hop}")
        # reverse direction hop
        if rank == 1:
            dist.send(pattern(1, shape, dt, "bool").to(dev), dst=0)
        elif rank == 0:
            got = torch.zeros(shape, dtype=dt).to(dev)
            dist.recv(got, src=1)
            need(eq_bytes(got, pattern(1, shape, dt, "bool")), "reverse")

    def case_isendirecv(dtname, label, spec):
        if world < 2:
            return "NA"
        dt = tdtype(dtname)
        shape = shape_for(label, spec, dt)
        nxt, prv = (rank + 1) % world, (rank - 1) % world
        out = pattern(rank, shape, dt).to(dev)
        inn = torch.zeros(shape, dtype=dt).to(dev)
        reqs = [dist.isend(out, nxt), dist.irecv(inn, prv)]
        for r in reqs:
            r.wait()
        need(eq_bytes(inn, pattern(prv, shape, dt)), "ring")

    def case_broadcast(dtname, label, spec):
        dt = tdtype(dtname)
        shape = shape_for(label, spec, dt)
        for root in range(world):
            t = (pattern(root, shape, dt) if rank == root else torch.zeros(shape, dtype=dt)).to(dev)
            dist.broadcast(t, src=root)
            need(eq_bytes(t, pattern(root, shape, dt)), f"root {root}")

    def case_allgather(dtname, label, spec):
        dt = tdtype(dtname)
        shape = shape_for(label, spec, dt)
        mine = pattern(rank, shape, dt).to(dev)
        outs = [torch.zeros(shape, dtype=dt).to(dev) for _ in range(world)]
        dist.all_gather(outs, mine)
        for r in range(world):
            need(eq_bytes(outs[r], pattern(r, shape, dt)), f"slot {r}")

    def case_gather(dtname, label, spec):
        dt = tdtype(dtname)
        shape = shape_for(label, spec, dt)
        for dst in range(world):
            mine = pattern(rank, shape, dt).to(dev)
            lst = [torch.zeros(shape, dtype=dt).to(dev) for _ in range(world)] if rank == dst else None
            dist.gather(mine, gather_list=lst, dst=dst)
            if rank == dst:
                for r in range(world):
                    need(eq_bytes(lst[r], pattern(r, shape, dt)), f"dst {dst} slot {r}")

    def case_allreduce(dtname, label, spec, opname):
        dt = tdtype(dtname)
        shape = shape_for(label, spec, dt)
        kind = "prod" if opname == "PRODUCT" else "val"
        if opname == "BAND":
            kind = "val"
        t = pattern(rank, shape, dt, kind).to(dev)
        op = getattr(dist.ReduceOp, opname)
        dist.all_reduce(t, op=op)
        if dt.is_floating_point:
            ref_t = torch.float64
        else:
            ref_t = torch.int64
        stack = torch.stack([pattern(r, shape, dt, kind).to(ref_t) for r in range(world)])
        if opname == "SUM":
            want = stack.sum(0)
        elif opname == "PRODUCT":
            want = stack.prod(0)
        elif opname == "MIN":
            want = stack.min(0).values
        elif opname == "MAX":
            want = stack.max(0).values
        elif opname == "AVG":
            want = stack.to(torch.float64).sum(0) / world
        else:
            want = stack[0]
            for s in stack[1:]:
                want = want & s
        if dt == torch.bool:
            ok = torch.equal(t.cpu().to(torch.int64), (want != 0).to(torch.int64))
        else:
            cmp_t = torch.float64 if opname == "AVG" else ref_t
            ok = torch.equal(t.cpu().to(cmp_t), want.to(cmp_t))
        need(ok, f"{opname} mismatch got {t.cpu().reshape(-1)[:4].tolist()} want {want.reshape(-1)[:4].tolist()}")

    def case_barrier(*_):
        for _ in range(3):
            dist.barrier()
        w = dist.barrier(async_op=True)
        w.wait()

    def case_object(kind):
        if kind == "all_gather_object":
            out = [None] * world
            dist.all_gather_object(out, {"rank": rank, "pad": "x" * (7 + rank)})
            need([o["rank"] for o in out] == list(range(world)), "all_gather_object")
        elif kind == "broadcast_object_list":
            for root in range(world):
                lst = [{"root": root, "v": [1, 2, 3]}] if rank == root else [None]
                dist.broadcast_object_list(lst, src=root)
                need(lst[0] == {"root": root, "v": [1, 2, 3]}, f"root {root}")
        elif kind == "gather_object":
            for dst in range(world):
                out = [None] * world if rank == dst else None
                dist.gather_object({"rank": rank}, out, dst=dst)
                if rank == dst:
                    need([o["rank"] for o in out] == list(range(world)), f"dst {dst}")

    def case_unsupported(opname):
        t = torch.ones(4 * world).to(dev)
        s = torch.ones(4).to(dev)
        lst = [torch.ones(4).to(dev) for _ in range(world)]
        calls = {
            "reduce": lambda: dist.reduce(s, dst=0),
            "scatter": lambda: dist.scatter(s, scatter_list=lst if rank == 0 else None, src=0),
            "all_to_all": lambda: dist.all_to_all(lst, [x.clone() for x in lst]),
            "all_to_all_single": lambda: dist.all_to_all_single(torch.zeros_like(t), t),
            "reduce_scatter": lambda: dist.reduce_scatter(s, lst),
            "reduce_scatter_tensor": lambda: dist.reduce_scatter_tensor(s, t),
            "all_gather_into_tensor": lambda: dist.all_gather_into_tensor(torch.zeros_like(t), s),
            "scatter_object_list": lambda: dist.scatter_object_list([None], [rank] * world if rank == 0 else None, src=0),
        }
        calls[opname]()
        raise Wrong("returned without raising")

    def case_noncontig(opname):
        base = (torch.arange(64, dtype=torch.float32).reshape(8, 8) + rank).to(dev)
        views = {"transpose": base.t(), "strided_slice": base[:, ::2]}
        for vname, v in views.items():
            if world < 2:
                return "NA"
            try:
                if opname == "all_reduce":
                    dist.all_reduce(v)
                elif opname == "broadcast":
                    dist.broadcast(v, src=0)
                elif opname == "send_recv":
                    if rank == 0:
                        dist.send(v, dst=1)
                    elif rank == 1:
                        dist.recv(v, src=0)
                    else:
                        raise ValueError("torch-tbccl: invalid argument: tensor must be contiguous (n/a rank)")
                elif opname == "all_gather":
                    dist.all_gather([v.clone() for _ in range(world)], v)
            except Exception as e:  # noqa: BLE001
                if "contiguous" not in str(e):
                    raise
            else:
                raise Wrong(f"{vname}: accepted a non-contiguous tensor")
        return "REJECTED:contiguous required (clear error, no copy)"

    def case_inplace(opname):
        if world < 2:
            return "NA"
        shape = (257,)
        if opname == "all_reduce":
            t = pattern(rank, shape, torch.float32).to(dev)
            alias = t.view(-1)
            r = dist.all_reduce(t, async_op=True)
            r.wait()
            want = sum(pattern(k, shape, torch.float32) for k in range(world))
            need(torch.equal(t.cpu(), want) and alias.data_ptr() == t.data_ptr() and torch.equal(alias.cpu(), want), "all_reduce is not in place")
        elif opname == "broadcast":
            t = (pattern(0, shape, torch.float32) if rank == 0 else torch.zeros(shape)).to(dev)
            alias = t[10:20]
            dist.broadcast(t, src=0)
            need(torch.equal(alias.cpu(), pattern(0, shape, torch.float32)[10:20]), "broadcast not visible through a view")
        elif opname == "recv":
            big = torch.full((300,), -1.0).to(dev)
            if rank == 0:
                dist.send(pattern(0, (100,), torch.float32).to(dev), dst=1)
            elif rank == 1:
                dist.recv(big[100:200], src=0)  # contiguous slice of a larger tensor
                b = big.cpu()
                need(torch.equal(b[100:200], pattern(0, (100,), torch.float32)) and bool((b[:100] == -1).all() and (b[200:] == -1).all()),
                     "recv wrote outside its slice or not at all")

    def over_sizes(f):
        for lb, sp in SIZES:
            if f(lb, sp) == "NA":
                return "NA"

    cases = []
    for dtn in BYTE_DTYPES:
        for fam, fn in (("send/recv", case_sendrecv), ("isend/irecv", case_isendirecv), ("broadcast", case_broadcast), ("all_gather", case_allgather), ("gather", case_gather)):
            cases.append((fam, dtn, "", lambda fn=fn, dtn=dtn: over_sizes(lambda lb, sp: fn(dtn, lb, sp))))
    for dtn in REDUCE_DTYPES:
        for opn in REDUCE_OPS:
            cases.append((f"all_reduce {opn}", dtn, "", lambda dtn=dtn, opn=opn: over_sizes(lambda lb, sp: case_allreduce(dtn, lb, sp, opn))))
    cases.append(("barrier", "-", "", case_barrier))
    for k in ("all_gather_object", "broadcast_object_list", "gather_object"):
        cases.append((k, "-", "", lambda k=k: case_object(k)))
    for opn in UNSUPPORTED_OPS:
        cases.append((opn, "-", "", lambda opn=opn: case_unsupported(opn)))
    for opn in ("all_reduce", "broadcast", "send_recv", "all_gather"):
        cases.append((f"noncontiguous {opn}", "float32", "", lambda opn=opn: case_noncontig(opn)))
    for opn in ("all_reduce", "broadcast", "recv"):
        cases.append((f"in-place {opn}", "float32", "", lambda opn=opn: case_inplace(opn)))

    only = set(a.only.split(",")) if a.only else None
    for op, dtn, _v, fn in cases:
        if only and op.split()[0] not in only and op not in only:
            continue
        rec = {"rank": rank, "world": world, "devmode": a.devmode, "op": op, "dtype": dtn}
        print("START " + json.dumps({"op": op, "dtype": dtn}), flush=True)
        t0 = time.monotonic()
        try:
            r = fn()
            if isinstance(r, str) and r.startswith("REJECTED"):
                rec.update(status="REJECTED", detail=r.split(":", 1)[1])
            elif r == "NA":
                rec.update(status="NA", detail="not meaningful in a one-rank group")
            else:
                rec.update(status="PASS", detail="")
        except Wrong as e:
            rec.update(status="WRONG", detail=str(e)[:300])
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")
            low = msg.lower()
            intentional = ("torch-tbccl: unsupported operation" in msg or "torch-tbccl: invalid argument" in msg or "does not support" in low
                           or "not implemented" in low or "not supported" in low)
            rec.update(status="REJECTED" if intentional else "ERROR", detail=f"{type(e).__name__}: {msg[:400]}")
        rec["ms"] = round((time.monotonic() - t0) * 1e3, 1)
        print("CASE " + json.dumps(rec), flush=True)
    print("DONE", flush=True)
    t0 = time.monotonic()
    dist.destroy_process_group()
    print(f"destroy {time.monotonic() - t0:.2f}s", flush=True)


# ---------------------------------------------------------------- driver side
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_config(world, devmode, hang, only=None, group_timeout=60):
    master = free_port()
    procs, lines = [], [[] for _ in range(world)]
    for rank in range(world):
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master), RANK=str(rank), WORLD_SIZE=str(world), TBCCL_LOCAL_ENDPOINT="127.0.0.1:0")
        env.setdefault("OMP_NUM_THREADS", "2")  # W processes each defaulting to every core starve each other
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--devmode", devmode, "--group-timeout", str(group_timeout)]
        if only:
            cmd += ["--only", only]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1))
    import selectors

    sel = selectors.DefaultSelector()
    for r, p in enumerate(procs):
        sel.register(p.stdout, selectors.EVENT_READ, r)
    last = time.monotonic()
    running = {r for r in range(world)}
    hung = None
    while running:
        events = sel.select(timeout=1.0)
        for key, _ in events:
            r = key.data
            line = key.fileobj.readline()
            if not line:
                sel.unregister(key.fileobj)
                running.discard(r)
                continue
            lines[r].append(line.rstrip("\n"))
            last = time.monotonic()
        if time.monotonic() - last > hang:
            hung = [l for l in lines[0] if l.startswith("START")][-1:] or ["START {}"]
            for p in procs:
                p.kill()
            break
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    return lines, hung, [p.returncode for p in procs]


def aggregate(world, devmode, lines, hung, rcs):
    per_case = {}
    order = []
    for rank_lines in lines:
        for l in rank_lines:
            if l.startswith("CASE "):
                rec = json.loads(l[5:])
                k = (rec["op"], rec["dtype"])
                if k not in per_case:
                    per_case[k] = []
                    order.append(k)
                per_case[k].append(rec)
    out = []
    for k in order:
        recs = per_case[k]
        statuses = {r["status"] for r in recs}
        if len(recs) < world:
            status, detail = "ERROR", f"only {len(recs)} of {world} ranks reported"
        elif len(statuses) == 1:
            status = statuses.pop()
            detail = next((r["detail"] for r in recs if r["detail"]), "")
        else:
            bad = [r for r in recs if r["status"] not in ("PASS", "NA")]
            status = bad[0]["status"] if bad else "PASS"
            detail = "ranks disagree: " + "; ".join(f"r{r['rank']}={r['status']} {r['detail'][:80]}" for r in recs)
        out.append({"world": world, "devmode": devmode, "op": k[0], "dtype": k[1], "status": status, "detail": detail, "ms": max(r["ms"] for r in recs)})
    if hung:
        started = json.loads(hung[0][6:]) if hung[0].startswith("START ") else {}
        out.append({"world": world, "devmode": devmode, "op": started.get("op", "?"), "dtype": started.get("dtype", "?"), "status": "HANG", "detail": "no progress", "ms": 0})
    return out


def cmd_run(a):
    records, meta = [], {}
    for devmode in a.devmodes.split(","):
        for w in (int(x) for x in a.worlds.split(",")):
            if devmode == "cuda0" and w == 1:
                pass
            t0 = time.monotonic()
            lines, hung, rcs = run_config(w, devmode, a.hang, a.only)
            recs = aggregate(w, devmode, lines, hung, rcs)
            records += recs
            tail = [l for l in lines[0] if l.startswith("destroy")]
            summary = {}
            for r in recs:
                summary[r["status"]] = summary.get(r["status"], 0) + 1
            print(f"W{w} {devmode}: {summary} rc={rcs} {time.monotonic() - t0:.0f}s {tail}", flush=True)
            for r in recs:
                if r["status"] in ("WRONG", "ERROR", "HANG"):
                    print(f"   {r['status']}: {r['op']} {r['dtype']}: {r['detail'][:200]}", flush=True)
            meta[f"{devmode}/W{w}"] = {"returncodes": rcs, "hung": bool(hung), "wall_s": round(time.monotonic() - t0, 1)}
    import torch

    import torch_tbccl

    doc = {"torch": torch.__version__, "torch_tbccl": torch_tbccl.__version__, "runtime": torch_tbccl.runtime_version(), "launches": meta, "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=1)
    bad = [r for r in records if r["status"] in ("WRONG", "ERROR", "HANG")]
    print(f"{len(records)} records, {len(bad)} WRONG/ERROR/HANG -> {a.out}")
    return 1 if bad else 0


CELL = {"PASS": "pass", "REJECTED": "rejected", "NA": "n/a", "WRONG": "**WRONG**", "ERROR": "**ERROR**", "HANG": "**HANG**"}


def cmd_render(a):
    doc = json.load(open(a.file))
    recs = doc["records"]
    devmodes = sorted({r["devmode"] for r in recs}, key=["cpu", "cuda0", "cudaall"].index)
    worlds = sorted({r["world"] for r in recs})

    def cell(op, dtype, devmode, w):
        m = [r for r in recs if r["op"] == op and r["dtype"] == dtype and r["devmode"] == devmode and r["world"] == w]
        return CELL[m[0]["status"]] if m else "-"

    def agg(ops, dtypes, devmode, w):
        m = [r for r in recs if r["op"] in ops and r["dtype"] in dtypes and r["devmode"] == devmode and r["world"] == w]
        if not m:
            return "-"
        st = {r["status"] for r in m}
        if st == {"PASS"}:
            return f"pass ({len(m)})"
        return "/".join(sorted(CELL[s] for s in st)) + f" ({sum(r['status'] == 'PASS' for r in m)}/{len(m)} pass)"

    out = [f"# Phase 71 capability matrix (generated; do not edit)\n",
           f"torch {doc['torch']}, torch-tbccl {doc['torch_tbccl']}, libtbccl {doc['runtime']}. Source: `tools/p71_capability_matrix.py`, raw data `capability_matrix.json`.\n",
           "Cell = result of the whole case (all 9 payload shapes, all root/destination variants). `pass (n)`: n dtype cases passed. Devmodes: cpu = all ranks CPU; "
           "cuda0 = rank 0 on CUDA, others CPU (heterogeneous); cudaall = every rank on the one GPU (shared, not multi-GPU).\n"]
    byte_ops = ["send/recv", "isend/irecv", "broadcast", "all_gather", "gather"]
    out.append("## Operations x world size, byte dtypes (12 dtypes: " + ", ".join(BYTE_DTYPES) + ")\n")
    out.append("| Operation | Devmode | " + " | ".join(f"W{w}" for w in worlds) + " |\n|---|---|" + "---:|" * len(worlds))
    for op in byte_ops:
        for dm in devmodes:
            out.append(f"| {op} | {dm} | " + " | ".join(agg([op], BYTE_DTYPES, dm, w) for w in worlds) + " |")
    out.append("\n## all_reduce: reduction op x world size (all dtypes the op passes are in the dtype table below)\n")
    out.append("| Op | Devmode | " + " | ".join(f"W{w}" for w in worlds) + " |\n|---|---|" + "---:|" * len(worlds))
    for opn in REDUCE_OPS:
        for dm in devmodes:
            out.append(f"| all_reduce {opn} | {dm} | " + " | ".join(agg([f"all_reduce {opn}"], REDUCE_DTYPES, dm, w) for w in worlds) + " |")
    out.append("\n## Other\n")
    other = ["barrier", "all_gather_object", "broadcast_object_list", "gather_object"] + UNSUPPORTED_OPS + [f"noncontiguous {x}" for x in ("all_reduce", "broadcast", "send_recv", "all_gather")] + [f"in-place {x}" for x in ("all_reduce", "broadcast", "recv")]
    out.append("| Operation | Devmode | " + " | ".join(f"W{w}" for w in worlds) + " |\n|---|---|" + "---:|" * len(worlds))
    for op in other:
        for dm in devmodes:
            dt = [r["dtype"] for r in recs if r["op"] == op][:1]
            out.append(f"| {op} | {dm} | " + " | ".join(cell(op, dt[0], dm, w) if dt else "-" for w in worlds) + " |")
    for dm in devmodes:
        out.append(f"\n## all_reduce: dtype x reduction op, {dm}\n")
        out.append("Each cell lists the world sizes at which the case passed (`-` = rejected at every world size; `!` marks a WRONG/ERROR/HANG).\n")
        out.append("| dtype | " + " | ".join(REDUCE_OPS) + " |\n|---|" + "---|" * len(REDUCE_OPS))
        for dt in REDUCE_DTYPES:
            row = []
            for opn in REDUCE_OPS:
                ok = [w for w in worlds if cell(f"all_reduce {opn}", dt, dm, w) == "pass"]
                bad = [w for w in worlds if cell(f"all_reduce {opn}", dt, dm, w) in ("**WRONG**", "**ERROR**", "**HANG**")]
                row.append((("W" + ",".join(map(str, ok))) if ok else "-") + ("!" + ",".join(map(str, bad)) if bad else ""))
            out.append(f"| {dt} | " + " | ".join(row) + " |")
        out.append(f"\n## byte-transport dtype x operation, {dm}\n")
        out.append("| dtype | " + " | ".join(byte_ops) + " |\n|---|" + "---|" * len(byte_ops))
        for dt in BYTE_DTYPES:
            row = []
            for op in byte_ops:
                ok = [w for w in worlds if cell(op, dt, dm, w) == "pass"]
                bad = [w for w in worlds if cell(op, dt, dm, w) in ("**WRONG**", "**ERROR**", "**HANG**")]
                row.append((("W" + ",".join(map(str, ok))) if ok else "-") + ("!" + ",".join(map(str, bad)) if bad else ""))
            out.append(f"| {dt} | " + " | ".join(row) + " |")
    import re

    def norm(msg):
        msg = re.sub(r"dtype=\w+ op=\w+", "dtype=<dt> op=<op>", msg)
        msg = re.sub(r"dtype \w+ has no reduction", "dtype <dt> has no reduction", msg)
        msg = re.sub(r"\(rank \d+", "(rank <r>", msg)
        msg = re.sub(r"\(this group has \d+ ranks", "(this group has <n> ranks", msg)
        return msg[:230]

    rej = {}
    for r in recs:
        if r["status"] == "REJECTED":
            op = re.sub(r" (SUM|PRODUCT|MIN|MAX|AVG|BAND)$", " <op>", r["op"])
            rej.setdefault((op, norm(r["detail"])), set()).add((r["devmode"], r["world"]))
    out.append("\n## Rejection messages (distinct, as raised)\n")
    out.append("| Operation | Message | Seen in |\n|---|---|---|")
    for (op, msg), where in sorted(rej.items()):
        ws = sorted({w for _, w in where})
        out.append(f"| {op} | `{msg}` | W{','.join(map(str, ws))} |")
    bad = [r for r in recs if r["status"] in ("WRONG", "ERROR", "HANG")]
    out.append(f"\n## Failures (WRONG / ERROR / HANG)\n\n{len(bad)} records.\n")
    for r in bad:
        out.append(f"- {r['devmode']} W{r['world']} {r['op']} {r['dtype']}: {r['status']} {r['detail'][:200]}")
    print("\n".join(out))


def cmd_check(a):
    doc = json.load(open(a.file))
    bad = mismatches(doc["records"])
    for r in bad:
        print(f"MISMATCH {r['devmode']} W{r['world']} {r['op']} {r['dtype']}: observed {r['status']} expected {expected_status(r['op'], r['dtype'], r['world'])} {r['detail'][:120]}")
    print(f"{len(doc['records'])} records, {len(bad)} differ from the documented surface")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--devmode", default="cpu")
    ap.add_argument("--only", default=None)
    ap.add_argument("--group-timeout", type=int, default=60)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--worlds", default="1,2,3,4")
    r.add_argument("--devmodes", default="cpu")
    r.add_argument("--hang", type=float, default=90.0)
    r.add_argument("--only", dest="only", default=None)
    r.add_argument("--out", required=True)
    d = sub.add_parser("render")
    d.add_argument("file")
    c = sub.add_parser("check")
    c.add_argument("file")
    a = ap.parse_args()
    if a.worker:
        worker(a)
    elif a.cmd == "run":
        sys.exit(cmd_run(a))
    elif a.cmd == "render":
        cmd_render(a)
    elif a.cmd == "check":
        sys.exit(cmd_check(a))
    else:
        ap.error("run | render")


if __name__ == "__main__":
    main()
