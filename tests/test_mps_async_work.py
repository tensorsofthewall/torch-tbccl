"""Phase 72: Work / Future / timeout / lifetime semantics with an MPS rank. The Phase 71 scenarios in tests/_worker_async.py run with rank 0 on MPS
(DEVICE_RANK0=mps); outstanding-Work counts, failure, abort and shutdown with MPS are the `mps` parameters of tests/test_async_work.py,
tests/test_failure_lifecycle.py and tests/test_lifecycle.py."""
import pytest
import torch

import torch_tbccl

pytestmark = [
    pytest.mark.multiprocess,
    pytest.mark.skipif(not (torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]), reason="needs MPS and an MPS-enabled build"),
]


@pytest.mark.parametrize("mode", ["inflight", "future_independent", "lifetime", "sequential", "timeout", "threads"])
def test_async_mode_with_an_mps_rank(run_two_ranks, mode):
    results = run_two_ranks("_worker_async.py", timeout=120, extra_env={"TEST_MODE": mode, "DEVICE_RANK0": "mps", "OMP_NUM_THREADS": "2"})
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


def test_peer_death_with_an_mps_survivor(run_two_ranks):
    (rc0, out0), (rc1, out1) = run_two_ranks("_worker_async.py", timeout=120, extra_env={"TEST_MODE": "peer_dies", "DEVICE_RANK0": "mps", "OMP_NUM_THREADS": "2"})
    assert rc0 == 0, out0
    assert rc1 == 7, out1
