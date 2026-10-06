"""Clean-install check: run with the Python of a venv that has torch-tbccl installed from a WHEEL, from a directory unrelated to the source checkout.

    cd <empty dir> && <venv>/bin/python clean_install_check.py [--device cpu|cuda] [--json OUT]

Parent: verifies the package was imported from an installed location (not a source tree, no PYTHONPATH, no editable finder), prints its info, then launches two ranks of this
same script on loopback (rendezvous through PyTorch's env:// TCPStore; TBCCL endpoints on host:0). Each rank selects the backend only through torch.distributed
(`init_process_group("tbccl")`) and runs: all_reduce SUM (float32, bfloat16), broadcast from rank 1, all_gather, send/recv, barrier, async all_reduce, destroy.
Exit status 0 only when both ranks pass every check.
"""
import argparse
import json
import os
import socket
import subprocess
import sys

CHILD = "--child" in sys.argv


def checks(device):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    import torch_tbccl  # noqa: F401  (registers the backend; the only non-torch import an application needs)

    rank = int(os.environ["RANK"])
    dev = torch.device(device if rank == 0 or device == "cpu" else "cpu")
    dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
    assert dist.get_backend() == "tbccl", dist.get_backend()
    out = {}
    x = (torch.arange(16, dtype=torch.float32) + rank * 100).to(dev)
    dist.all_reduce(x)
    assert torch.equal(x.cpu(), torch.arange(16, dtype=torch.float32) * 2 + 100), x
    out["all_reduce_f32"] = "ok"
    b = (torch.arange(8, dtype=torch.float32) + 1).to(torch.bfloat16).to(dev) + rank
    dist.all_reduce(b)
    assert torch.equal(b.cpu(), (torch.arange(8, dtype=torch.float32) + 1).to(torch.bfloat16) * 2 + 1), b
    out["all_reduce_bf16"] = "ok"
    y = torch.full((5,), 7.0 if rank == 1 else 0.0).to(dev)
    dist.broadcast(y, src=1)
    assert torch.equal(y.cpu(), torch.full((5,), 7.0))
    out["broadcast"] = "ok"
    g = [torch.zeros(3).to(dev) for _ in range(2)]
    dist.all_gather(g, torch.full((3,), float(rank + 1)).to(dev))
    assert [t.cpu()[0].item() for t in g] == [1.0, 2.0]
    out["all_gather"] = "ok"
    if rank == 0:
        dist.send((torch.arange(10) * 3).to(torch.int32).to(dev), dst=1)
        r = torch.zeros(4).to(dev)
        dist.recv(r, src=1)
        assert torch.equal(r.cpu(), torch.arange(4, dtype=torch.float32))
    else:
        r = torch.zeros(10, dtype=torch.int32).to(dev)
        dist.recv(r, src=0)
        assert torch.equal(r.cpu(), (torch.arange(10) * 3).to(torch.int32))
        dist.send(torch.arange(4, dtype=torch.float32).to(dev), dst=0)
    out["send_recv"] = "ok"
    dist.barrier()
    z = torch.ones(1 << 16).to(dev)
    w = dist.all_reduce(z, async_op=True)
    w.wait()
    assert z.cpu()[0].item() == 2.0
    out["async_all_reduce"] = "ok"
    dist.destroy_process_group()
    return out


def child():
    import torch_tbccl

    res = {"rank": int(os.environ["RANK"]), "checks": checks(os.environ["TORCH_TBCCL_CHECK_DEVICE"]), "torch_tbccl_file": torch_tbccl.__file__}
    print("CHILD " + json.dumps(res), flush=True)


def parent():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    import torch
    import torch_tbccl

    here = os.path.dirname(os.path.abspath(torch_tbccl.__file__))
    problems = []
    if "site-packages" not in here:
        problems.append(f"torch_tbccl was imported from {here}, not from an installed site-packages directory")
    if os.environ.get("PYTHONPATH"):
        problems.append("PYTHONPATH is set")
    if os.path.exists(os.path.join(os.path.dirname(here), "setup.py")):
        problems.append("the import location sits next to a setup.py (a source checkout)")
    info = torch_tbccl.info()
    print("INFO " + json.dumps(info), flush=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        problems.append("cuda requested but unavailable")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = []
    for rank in range(2):
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", TORCH_TBCCL_CHECK_DEVICE=a.device)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    results = []
    for p in procs:
        try:
            out, _ = p.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            problems.append("a rank timed out")
        if p.returncode != 0:
            problems.append(f"a rank exited {p.returncode}: {out[-800:]}")
        results += [json.loads(l[6:]) for l in out.splitlines() if l.startswith("CHILD ")]
    report = {"cwd": os.getcwd(), "device": a.device, "info": info, "ranks": results, "problems": problems}
    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if problems or len(results) != 2 else 0)


if __name__ == "__main__":
    child() if CHILD else parent()
