import pytest


def run(run_two_ranks, mode, timeout=90):
    return run_two_ranks("_worker_async.py", timeout=timeout, extra_env={"TEST_MODE": mode})


@pytest.mark.parametrize(
    "mode", ["inflight", "future_independent", "lifetime", "sequential", "timeout", "threads"]
)
def test_async_mode(run_two_ranks, mode):
    results = run(run_two_ranks, mode)
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


def test_peer_death_propagates_failure_and_shuts_down_cleanly(run_two_ranks):
    (rc0, out0), (rc1, out1) = run(run_two_ranks, "peer_dies")
    assert rc0 == 0, out0
    assert rc1 == 7, out1
    assert out0.strip().splitlines()[-1] == "rank 0 ok"


# outstanding-Work counts, async variants of every asynchronous-capable operation, and resource growth (tests/_worker_async71.py).
import torch  # noqa: E402

import torch_tbccl  # noqa: E402

HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
HAS_MPS = torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]
DEVICES = ["", pytest.param("cuda", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")), pytest.param("mps", marks=pytest.mark.skipif(not HAS_MPS, reason="needs MPS"))]


def run71(run_two_ranks, mode, n=8, device="", timeout=180):
    results = run_two_ranks("_worker_async71.py", timeout=timeout, extra_env={"TEST_MODE": mode, "N": str(n), "DEVICE_RANK0": device})
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]
    return [out for _, out in results]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [2, 8, 32])
@pytest.mark.parametrize("mode", ["p2p_outstanding", "allreduce_outstanding"])
def test_outstanding_works(run_two_ranks, mode, n, device):
    run71(run_two_ranks, mode, n, device)


@pytest.mark.parametrize("device", DEVICES)
def test_async_collectives_return_before_completion(run_two_ranks, device):
    run71(run_two_ranks, "async_collectives", device=device)


def test_outstanding_works_do_not_grow_resources(run_two_ranks):
    run71(run_two_ranks, "growth", timeout=300)
