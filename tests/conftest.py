import os
import socket
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


@pytest.fixture
def run_two_ranks():
    """Run a worker script as two processes on loopback; return their (rc, output)."""

    def run(script, timeout=60, extra_env=None):
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
                    [sys.executable, os.path.join(HERE, script)],
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
