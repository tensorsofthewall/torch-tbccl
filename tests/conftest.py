import os
import socket
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _bindable(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def free_ports(n):
    # Endpoint ports also need their TBCCL data port (+1000) free.
    out = []
    while len(out) < n:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if p + 1000 < 65536 and _bindable(p + 1000) and p not in out:
            out.append(p)
    return out


@pytest.fixture
def run_two_ranks():
    """Run a worker script as two processes on loopback; return their (rc, output)."""

    def run(script, timeout=60, extra_env=None, args=()):
        # Endpoint ports are chosen by the kernel (host:0): a port probed free here can be taken as an ephemeral source port by the time a rank binds it. The fixed
        # host:port convention (data port = port + 1000) is covered in-process by tests/test_bootstrap.py.
        master = free_ports(1)[0]
        procs = []
        for rank in range(2):
            env = dict(
                os.environ,
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(master),
                RANK=str(rank),
                WORLD_SIZE="2",
                TBCCL_LOCAL_ENDPOINT="127.0.0.1:0",
            )
            env.update(extra_env or {})
            procs.append(
                subprocess.Popen(
                    [sys.executable, os.path.join(HERE, script), *args],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
            )
        results = []
        for p in procs:
            try:
                out, _ = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                results.append((-9, "TIMEOUT\n" + out))
                continue
            results.append((p.returncode, out))
        return results

    return run


@pytest.fixture
def run_ranks():
    """Run a worker script as `world` processes on loopback with dynamically allocated TBCCL ports (host:0); return their (rc, output)."""

    def run(script, world, timeout=90, extra_env=None, per_rank_env=None, args=()):
        master = free_ports(1)[0]
        procs = []
        for rank in range(world):
            env = dict(
                os.environ,
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(master),
                RANK=str(rank),
                WORLD_SIZE=str(world),
                TBCCL_LOCAL_ENDPOINT="127.0.0.1:0",
            )
            env.update(extra_env or {})
            env.update((per_rank_env or {}).get(rank, {}))
            procs.append(
                subprocess.Popen(
                    [sys.executable, os.path.join(HERE, script), *args],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
            )
        results = []
        for p in procs:
            try:
                out, _ = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                results.append((-9, "TIMEOUT\n" + out))
                continue
            results.append((p.returncode, out))
        return results

    return run


def pytest_addoption(parser):
    parser.addoption("--physical", action="store_true", default=False, help="run tests marked 'physical' (they touch a second host / the Thunderbolt link)")


def pytest_collection_modifyitems(config, items):
    """Group the tests for CI selection (-m "not multiprocess", -m cuda, -m ddp, ...) without decorating every old test; 'physical' never runs unless asked for."""
    skip_physical = pytest.mark.skip(reason="physical tests need --physical (two hosts / the Thunderbolt link)")
    for item in items:
        fixtures = set(getattr(item, "fixturenames", ()))
        if fixtures & {"run_two_ranks", "run_ranks"}:
            item.add_marker(pytest.mark.multiprocess)
        if "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.cuda)
        if "mps" in item.nodeid.lower():
            item.add_marker(pytest.mark.mps)
        if "ddp" in item.nodeid.lower():
            item.add_marker(pytest.mark.ddp)
        if "physical" in item.keywords and not config.getoption("--physical"):
            item.add_marker(skip_physical)
