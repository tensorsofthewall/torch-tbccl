"""Phase 73: P2P and collectives overlap safely on one process group (libtbccl wire 4 separates the two ordering domains). Replaces the Phase 71 overlap guard tests.
Scenarios: tests/_worker_mixed.py."""
import pytest
import torch

import torch_tbccl

HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
HAS_MPS = torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]
DEVICES = ["", pytest.param("cuda", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")), pytest.param("mps", marks=pytest.mark.skipif(not HAS_MPS, reason="needs MPS"))]

pytestmark = pytest.mark.multiprocess


def check(results):
    for rank, (rc, out) in enumerate(results):
        assert rc == 0, f"rank {rank}:\n{out}"
        assert out.strip().splitlines()[-1] == f"rank {rank} ok", out


@pytest.mark.parametrize("device", DEVICES)
def test_collectives_and_p2p_overlap_in_either_relative_order(run_two_ranks, device):
    check(run_two_ranks("_worker_mixed.py", timeout=300, extra_env={"AUTO_PORT": "1", "OMP_NUM_THREADS": "2", "DEVICE_RANK0": device}))


@pytest.mark.parametrize("world", [3, 4])
def test_p2p_ring_overlaps_all_reduce(run_ranks, world):
    check(run_ranks("_worker_mixed.py", world, timeout=300, extra_env={"OMP_NUM_THREADS": "2"}))
