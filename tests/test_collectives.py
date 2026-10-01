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
    results = run_two_ranks(
        "_worker_collectives.py", timeout=180, extra_env={"TEST_MODE": mode, "CUDA_RANKS": cuda_ranks}
    )
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


@pytest.mark.parametrize("cuda_ranks", LAYOUTS)
@pytest.mark.parametrize("mode", ["broadcast", "all_gather", "mixed_order"])
def test_collectives(run_two_ranks, mode, cuda_ranks):
    run(run_two_ranks, mode, cuda_ranks)


def test_argument_errors(run_two_ranks):
    run(run_two_ranks, "errors")
