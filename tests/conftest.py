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
        master, p0, p1 = free_ports(3)
        procs = []
        for rank, port in enumerate((p0, p1)):
            env = dict(
                os.environ,
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(master),
                RANK=str(rank),
                WORLD_SIZE="2",
                TBCCL_LOCAL_ENDPOINT=f"127.0.0.1:{port}",
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
