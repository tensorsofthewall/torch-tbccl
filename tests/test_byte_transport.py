import pytest
import torch

import torch_tbccl


def run(run_two_ranks, mode, cuda_ranks="", timeout=300):
    results = run_two_ranks("_worker_bytes.py", timeout=timeout, extra_env={"TEST_MODE": mode, "CUDA_RANKS": cuda_ranks})
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


MODES = ["p2p", "broadcast_gather", "bundle", "reduction_rejected"]


@pytest.mark.parametrize("mode", MODES)
def test_cpu_byte_transport(run_two_ranks, mode):
    run(run_two_ranks, mode)


_CUDA = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]),
    reason="needs a CUDA GPU and a CUDA-enabled torch-tbccl build",
)


@_CUDA
@pytest.mark.parametrize("cuda_ranks", ["0", "1", "0,1"])
@pytest.mark.parametrize("mode", MODES)
def test_cuda_byte_transport(run_two_ranks, mode, cuda_ranks):
    run(run_two_ranks, mode, cuda_ranks)
