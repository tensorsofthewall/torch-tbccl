"""Real DistributedDataParallel over the "tbccl" backend (examples/ddp_mlp.py), each run checked by the script against a single-process,
non-distributed reference trained on the same global batch (per-step loss, gradient norm, final parameters, identical parameters on every rank).
Runs go through the run_ranks fixture (env:// rendezvous) and, for the launcher tests, through `torchrun`."""
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

import torch_tbccl
from conftest import HERE

pytestmark = [pytest.mark.ddp, pytest.mark.multiprocess]
HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
SCRIPT = "../examples/ddp_mlp.py"


def results_of(outs):
    res = []
    for out in outs:
        lines = [json.loads(l[len("RESULT "):]) for l in out.splitlines() if l.startswith("RESULT ")]
        assert lines, out
        res += lines
    return res


def run_ddp(run_ranks, world, *args, timeout=240, per_rank_env=None):
    rs = run_ranks(SCRIPT, world, timeout=timeout, args=args, per_rank_env=per_rank_env, extra_env={"OMP_NUM_THREADS": "2"})
    for rank, (rc, out) in enumerate(rs):
        assert rc == 0, f"rank {rank}:\n{out}"
    res = results_of([out for _, out in rs])
    assert all(r["ok"] for r in res), res
    assert all(r["ranks_identical"] for r in res) or any(r["device"].startswith("mps") for r in res), res  # MPS next to another device type is checked by tolerance (examples/ddp_mlp.py)
    return res


@pytest.mark.parametrize("world", [2, 3, 4])
def test_cpu_ddp_matches_the_single_process_reference(run_ranks, world):
    res = run_ddp(run_ranks, world, "--steps", "10")
    assert len(res) == world and all(r["world"] == world and r["device"] == "cpu" for r in res)
    assert all(r["last_loss"] < r["first_loss"] for r in res), "the loss must go down"


@pytest.mark.parametrize("cap", ["0.0005", "25"])
def test_bucket_cap_configurations(run_ranks, cap):
    run_ddp(run_ranks, 2, "--steps", "6", "--bucket-cap-mb", cap)


def test_unused_parameters(run_ranks):
    run_ddp(run_ranks, 2, "--steps", "6", "--unused")


@needs_cuda
def test_heterogeneous_cuda_and_cpu_ranks(run_ranks):
    # PyTorch's DDP accepts one rank on CUDA and one on CPU; parameter broadcast, gradient all-reduce and the local devices all behave.
    res = run_ddp(run_ranks, 2, "--steps", "6", "--devices", "cuda,cpu")
    assert {r["device"] for r in res} == {"cuda", "cpu"}


@needs_cuda
def test_two_cuda_ranks_sharing_one_gpu_smoke(run_ranks):
    # NOT multi-GPU DDP: both ranks use cuda:0 of the one GPU. It only shows the CUDA path of the reducer works.
    run_ddp(run_ranks, 2, "--steps", "6", "--devices", "cuda")


@needs_cuda
def test_one_rank_ddp_is_rejected_clearly(run_ranks):
    # DDP's constructor verifies parameter shapes with an all_gather; a one-rank group rejects collectives (documented), so a world of one fails up front.
    rs = run_ranks(SCRIPT, 1, timeout=120, args=("--devices", "cuda"))
    rc, out = rs[0]
    assert rc != 0 and "needs a group of at least 2 ranks" in out, out


def test_repeated_ddp_lifecycle_in_one_process(run_ranks):
    # create model -> wrap in DDP -> train -> destroy the group, three times, in the same interpreters (env:// builds a new store each time)
    res = run_ddp(run_ranks, 2, "--steps", "4", "--cycles", "3")
    assert len(res) == 6


TORCHRUN = os.path.join(os.path.dirname(sys.executable), "torchrun")


@pytest.mark.skipif(not os.path.exists(TORCHRUN), reason="torchrun not installed next to the interpreter")
@pytest.mark.parametrize("nproc,cycles", [(2, 1), (2, 3), (4, 1)])
def test_torchrun_launch(nproc, cycles):
    # torchrun owns the store (agent store): with --cycles 3 every cycle re-initializes over the SAME store (the v2 stale-record regression).
    # --local-addr 127.0.0.1: torchrun otherwise registers the host's FQDN, which on a Mac whose hostname does not resolve (seen after a reboot) hangs the rendezvous.
    env = dict(os.environ, TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", OMP_NUM_THREADS="2")
    p = subprocess.run(
        [TORCHRUN, "--standalone", "--local-addr", "127.0.0.1", "--nproc-per-node", str(nproc), os.path.join(HERE, SCRIPT), "--steps", "4", "--cycles", str(cycles)],
        env=env, capture_output=True, text=True, timeout=240,
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    res = results_of([p.stdout])
    assert len(res) == nproc * cycles and all(r["ok"] for r in res)


@pytest.mark.parametrize("world", [2, 3])
def test_ddp_with_concurrent_application_p2p_on_the_same_group(run_ranks, world):
    # An application thread exchanges deterministic messages (isend/irecv) while DDP's backward runs its gradient all_reduces on the same ProcessGroupTBCCL.
    res = run_ddp(run_ranks, world, "--steps", "10", "--side-p2p", "60")
    assert all(r["side_p2p_ok"] and r["side_p2p_messages"] == 60 for r in res), res


@needs_cuda
def test_ddp_with_concurrent_application_p2p_cuda_cpu(run_ranks):
    res = run_ddp(run_ranks, 2, "--steps", "10", "--devices", "cuda,cpu", "--side-p2p", "60")
    assert all(r["side_p2p_ok"] for r in res), res
