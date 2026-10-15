"""examples/two_host_allreduce.py runs under the same torchrun flags the quickstart documents (both nodes on loopback), and the docs show the commands the example's docstring does."""
import os
import re
import subprocess
import sys

import pytest

from conftest import free_ports

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "two_host_allreduce.py")
QUICKSTART = open(os.path.join(ROOT, "docs", "getting-started", "quickstart.md")).read()
TORCHRUN = os.path.join(os.path.dirname(sys.executable), "torchrun")


def test_quickstart_two_host_commands_match_the_example():
    section = QUICKSTART.split("## Across two hosts")[1].split("## Devices")[0]
    for rank, own in ((0, "192.168.77.1"), (1, "192.168.77.2")):
        pattern = (rf"TBCCL_LOCAL_ENDPOINT={re.escape(own)}:0 \.venv/bin/torchrun --nnodes 2 --nproc-per-node 1 --node-rank {rank} \\\n"
                   rf"\s+--local-addr {re.escape(own)} --master-addr 192\.168\.77\.1 --master-port 29500 --rdzv-conf timeout=60 two_host_allreduce\.py")
        assert re.search(pattern, section), rank
    doc = open(EXAMPLE).read().split('"""')[1]
    assert "--node-rank 0" in doc and "--node-rank 1" in doc and "--local-addr 192.168.77.1" in doc
    assert "rank 0: backend=tbccl result=3.0 PASS" in section


@pytest.mark.multiprocess
@pytest.mark.skipif(not os.path.exists(TORCHRUN), reason="torchrun not installed next to the interpreter")
def test_two_nodes_on_loopback(tmp_path):
    master = free_ports(1)[0]
    env = dict(os.environ, TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", OMP_NUM_THREADS="2")
    procs = [subprocess.Popen([TORCHRUN, "--nnodes", "2", "--nproc-per-node", "1", "--node-rank", str(r), "--local-addr", "127.0.0.1",
                               "--master-addr", "127.0.0.1", "--master-port", str(master), "--rdzv-conf", "timeout=60", EXAMPLE, "--elements", "4096"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=tmp_path) for r in (0, 1)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    for r, out in enumerate(outs):
        assert f"rank {r}: backend=tbccl result=3.0 PASS" in out, out


def _load_example():
    import importlib.util

    spec = importlib.util.spec_from_file_location("two_host_allreduce", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Work:
    def __init__(self, calls):
        self.calls = calls

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        return True


def _fake_group(monkeypatch, mod, backend="tbccl", reduce=True, world=2):
    calls = []
    monkeypatch.setattr(mod.dist, "init_process_group", lambda name, timeout=None: calls.append(("init", name, timeout)))
    monkeypatch.setattr(mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mod.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(mod.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(mod.dist, "destroy_process_group", lambda: calls.append(("destroy",)))

    def all_reduce(x, async_op=False):
        if reduce:
            x.add_(sum(range(2, world + 1)))
        return _Work(calls)

    monkeypatch.setattr(mod.dist, "all_reduce", all_reduce)
    return calls


def test_a_correct_sum_on_tbccl_passes_with_bounded_waits_and_cleanup(monkeypatch, capsys):
    mod = _load_example()
    calls = _fake_group(monkeypatch, mod)
    assert mod.main(["--timeout", "7"]) == 0
    assert "rank 0: backend=tbccl result=3.0 PASS" in capsys.readouterr().out
    init = next(c for c in calls if c[0] == "init")
    assert init[1] == "tbccl" and init[2].total_seconds() == 7
    assert ("wait", init[2]) in calls and calls[-1] == ("destroy",)


def test_a_wrong_sum_fails_with_a_nonzero_exit_and_still_cleans_up(monkeypatch, capsys):
    mod = _load_example()
    calls = _fake_group(monkeypatch, mod, reduce=False)
    assert mod.main([]) == 1
    assert "FAIL" in capsys.readouterr().out and calls[-1] == ("destroy",)


def test_a_different_backend_fails_even_if_the_numbers_are_right(monkeypatch, capsys):
    mod = _load_example()
    _fake_group(monkeypatch, mod, backend="gloo")
    assert mod.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_an_exception_in_the_collective_still_destroys_the_group(monkeypatch):
    mod = _load_example()
    calls = _fake_group(monkeypatch, mod)
    monkeypatch.setattr(mod.dist, "all_reduce", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        mod.main([])
    assert calls[-1] == ("destroy",)


@pytest.mark.multiprocess
@pytest.mark.skipif(not os.path.exists(TORCHRUN), reason="torchrun not installed next to the interpreter")
def test_a_missing_peer_makes_the_documented_command_fail_fast(tmp_path):
    env = dict(os.environ, TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", OMP_NUM_THREADS="2")
    p = subprocess.run([TORCHRUN, "--nnodes", "2", "--nproc-per-node", "1", "--node-rank", "0", "--local-addr", "127.0.0.1", "--master-addr", "127.0.0.1",
                        "--master-port", str(free_ports(1)[0]), "--rdzv-conf", "timeout=5", EXAMPLE, "--timeout", "3"],
                       env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert p.returncode != 0
