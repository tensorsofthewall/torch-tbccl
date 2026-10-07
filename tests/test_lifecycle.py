"""Repeated group lifecycle with resource accounting, interpreter shutdown, endpoint reuse (tests/_worker_lifecycle.py, tests/_worker_shutdown.py)."""
import os
import subprocess
import sys
import time

import pytest
import torch

import torch_tbccl
from conftest import HERE

pytestmark = pytest.mark.multiprocess
HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA"))]
HAS_MPS = torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]
SHUTDOWN_DEVICES = DEVICES + [pytest.param("mps", marks=pytest.mark.skipif(not HAS_MPS, reason="needs MPS"))]


def ok(results):
    for rank, (rc, out) in enumerate(results):
        if os.environ.get("P71_ECHO"):
            print(f"--- rank {rank} rc={rc}\n{out}")
        assert rc == 0, f"rank {rank}:\n{out}"
        assert out.strip().splitlines()[-1] == f"rank {rank} ok", out


@pytest.mark.parametrize("device", DEVICES)
def test_ten_init_collective_destroy_cycles_are_flat(run_ranks, device):
    ok(run_ranks("_worker_lifecycle.py", 2, timeout=240, extra_env={"TEST_MODE": "cycles", "CYCLES": "10", "DEVICE": device}))


def test_one_hundred_group_create_destroy_cycles_are_flat(run_ranks):
    ok(run_ranks("_worker_lifecycle.py", 2, timeout=300, extra_env={"TEST_MODE": "groups", "GROUPS": "100", "DEVICE": "cpu"}))


@pytest.mark.parametrize("world", [3, 4])
def test_cycles_at_larger_world_sizes(run_ranks, world):
    ok(run_ranks("_worker_lifecycle.py", world, timeout=300, extra_env={"TEST_MODE": "cycles", "CYCLES": "5", "DEVICE": "cpu"}))


@pytest.mark.parametrize("device", SHUTDOWN_DEVICES)
def test_normal_interpreter_exit_never_hangs(run_two_ranks, device):
    # init, one operation, and then plain interpreter exit: no destroy_process_group, no os._exit; repeated.
    for i in range(6):
        t0 = time.monotonic()
        results = run_two_ranks("_worker_shutdown.py", timeout=60, extra_env={"DEVICE": device, "AUTO_PORT": "1"})
        for rank, (rc, out) in enumerate(results):
            assert rc == 0, f"run {i} rank {rank}:\n{out}"
        assert time.monotonic() - t0 < 30, f"run {i} took {time.monotonic() - t0:.1f}s"
