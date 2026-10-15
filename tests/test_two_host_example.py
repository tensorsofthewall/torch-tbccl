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
                   rf"\s+--local-addr {re.escape(own)} --master-addr 192\.168\.77\.1 --master-port 29500 two_host_allreduce\.py")
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
                               "--master-addr", "127.0.0.1", "--master-port", str(master), EXAMPLE, "--elements", "4096"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=tmp_path) for r in (0, 1)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    for r, out in enumerate(outs):
        assert f"rank {r}: backend=tbccl result=3.0 PASS" in out, out
