"""Peer failure, abort and teardown semantics (loopback; the dying peer is always a local process). Scenarios: tests/_worker_failure71.py."""
import os

import pytest
import torch

import torch_tbccl

pytestmark = pytest.mark.multiprocess
HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
DEVICES = ["", pytest.param("cuda", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA"))]


def survivors_ok(results, dead=(), dead_rc=9):
    if os.environ.get("TORCH_TBCCL_TEST_ECHO"):  # evidence capture: TORCH_TBCCL_TEST_ECHO=1 pytest -s tests/test_failure_lifecycle.py
        for rank, (rc, out) in enumerate(results):
            print(f"--- rank {rank} rc={rc}\n{out}")
    for rank, (rc, out) in enumerate(results):
        if rank in dead:
            assert rc == dead_rc, f"rank {rank} should have died with {dead_rc}: rc={rc}\n{out}"
        else:
            assert rc == 0, f"rank {rank}:\n{out}"
            assert out.strip().splitlines()[-1] == f"rank {rank} ok", out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "mode", ["recv_peer_exit", "recv_peer_exit_blocking", "send_peer_exit", "allreduce_peer_exit_before", "allreduce_peer_exit_during", "ddp_peer_exit"]
)
def test_peer_exit_fails_survivor_boundedly(run_two_ranks, mode, device):
    survivors_ok(run_two_ranks("_worker_failure71.py", timeout=150, extra_env={"TEST_MODE": mode, "DEVICE_RANK0": device}), dead=(1,))


def test_peer_exit_in_a_three_rank_all_reduce(run_ranks):
    survivors_ok(run_ranks("_worker_failure71.py", 3, timeout=150, extra_env={"TEST_MODE": "allreduce_w3"}), dead=(2,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["abort_recv", "abort_send", "abort_collective", "destroy_outstanding"])
def test_abort_and_destroy_terminate_every_pending_work(run_two_ranks, mode, device):
    survivors_ok(run_two_ranks("_worker_failure71.py", timeout=150, extra_env={"TEST_MODE": mode, "DEVICE_RANK0": device}))
