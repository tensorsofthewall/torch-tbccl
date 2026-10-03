import pytest
import torch

import torch_tbccl


def run(run_two_ranks, mode, cuda_ranks="", timeout=300):
    results = run_two_ranks("_worker_lowprecision.py", timeout=timeout, extra_env={"TEST_MODE": mode, "CUDA_RANKS": cuda_ranks})
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


@pytest.mark.parametrize("mode", ["allreduce", "stream", "errors"])
def test_cpu_low_precision(run_two_ranks, mode):
    run(run_two_ranks, mode)


_CUDA = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]),
    reason="needs a CUDA GPU and a CUDA-enabled torch-tbccl build",
)


# "0" = CUDA rank 0 <-> CPU rank 1; "1" = the reverse orientation; "0,1" = both ranks on one GPU.
@_CUDA
@pytest.mark.parametrize("cuda_ranks", ["0", "1", "0,1"])
@pytest.mark.parametrize("mode", ["allreduce", "stream", "errors"])
def test_cuda_low_precision(run_two_ranks, mode, cuda_ranks):
    run(run_two_ranks, mode, cuda_ranks)


@_CUDA
@pytest.mark.parametrize("cuda_ranks", ["0", "0,1"])
def test_stream_negative_control_detects_missing_dependency(run_two_ranks, cuda_ranks):
    run(run_two_ranks, "stream_negative_control", cuda_ranks)
