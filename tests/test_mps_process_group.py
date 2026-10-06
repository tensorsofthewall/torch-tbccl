"""Phase 72: CPU <-> MPS ProcessGroup operations on one Mac (loopback). Scenarios: tests/_worker_mps.py. The MPS rank is either rank, so both orientations run."""
import pytest
import torch

import torch_tbccl

pytestmark = [
    pytest.mark.multiprocess,
    pytest.mark.skipif(not (torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]), reason="needs MPS and an MPS-enabled build"),
]


def run(run_two_ranks, mode, mps_rank, timeout=240):
    results = run_two_ranks("_worker_mps.py", timeout=timeout, extra_env={"TEST_MODE": mode, "MPS_RANK": str(mps_rank), "AUTO_PORT": "1", "OMP_NUM_THREADS": "2"})
    report = "\n".join(f"--- rank {r} rc={rc}\n{out}" for r, (rc, out) in enumerate(results))
    for rank, (rc, out) in enumerate(results):
        assert rc == 0, report
        assert out.strip().splitlines()[-1] == f"rank {rank} ok", out
    return [out for _, out in results]


@pytest.mark.parametrize("mps_rank", [1, 0])
@pytest.mark.parametrize("mode", ["p2p", "collectives", "dtypes", "offset", "lifetime", "noncontig"])
def test_cpu_mps_scenario(run_two_ranks, mode, mps_rank):
    run(run_two_ranks, mode, mps_rank)
