import pytest
import torch

import torch_tbccl

HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
LAYOUTS = [
    "",
    pytest.param("0", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")),
    pytest.param("1", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")),
    pytest.param("0,1", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")),
]


def run(run_two_ranks, mode, cuda_ranks=""):
    results = run_two_ranks("_worker_p2p.py", timeout=180, extra_env={"TEST_MODE": mode, "CUDA_RANKS": cuda_ranks})
    for rc, out in results:
        assert rc == 0, out
    assert all("ok" in out for _, out in results)


@pytest.mark.parametrize("cuda_ranks", LAYOUTS)
def test_p2p_matrix(run_two_ranks, cuda_ranks):
    run(run_two_ranks, "matrix", cuda_ranks)


@pytest.mark.parametrize("cuda_ranks", ["", pytest.param("0", marks=pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA"))])
def test_blocked_recv_abort(run_two_ranks, cuda_ranks):
    run(run_two_ranks, "blocked_abort", cuda_ranks)


def test_many_groups_auto_ports(run_two_ranks):
    # TBCCL_LOCAL_ENDPOINT=host:0: every communicator picks its own free port pair (world + 3 pair groups + 1-rank groups).
    results = run_two_ranks("_worker_groups.py", timeout=120, extra_env={"AUTO_PORT": "1"})
    for rc, out in results:
        assert rc == 0, out
        assert "ok" in out
