"""Phase 71: collective and point-to-point operations in flight together on one group are refused, never silently corrupted (tests/_worker_overlap.py).
Phase 72: the guard applies to MPS tensors too."""
import pytest
import torch

import torch_tbccl

HAS_MPS = torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]


@pytest.mark.multiprocess
@pytest.mark.parametrize("device", ["", pytest.param("mps", marks=pytest.mark.skipif(not HAS_MPS, reason="needs MPS"))])
def test_overlapping_collective_and_p2p_is_refused_and_the_group_survives(run_two_ranks, device):
    for rc, out in run_two_ranks("_worker_overlap.py", timeout=120, extra_env={"AUTO_PORT": "1", "OMP_NUM_THREADS": "2", "DEVICE_RANK0": device}):
        assert rc == 0, out
        assert out.strip().splitlines()[-1].endswith("ok"), out
