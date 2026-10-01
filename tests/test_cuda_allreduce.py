import pytest
import torch

import torch_tbccl

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]),
    reason="needs a CUDA GPU and a CUDA-enabled torch-tbccl build",
)

# "0" = CUDA rank 0 <-> CPU rank 1 (the real-link shape); "0,1" = both ranks on one GPU
# (validates adapter/device-pointer semantics, not multi-GPU scaling).
LAYOUTS = ["0", "1", "0,1"]


def run(run_two_ranks, mode, cuda_ranks, timeout=120):
    results = run_two_ranks(
        "_worker_cuda.py", timeout=timeout, extra_env={"TEST_MODE": mode, "CUDA_RANKS": cuda_ranks}
    )
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]
    return results


@pytest.mark.parametrize("cuda_ranks", LAYOUTS)
@pytest.mark.parametrize("mode", ["basic", "stream", "consumer", "lifetime"])
def test_cuda(run_two_ranks, mode, cuda_ranks):
    run(run_two_ranks, mode, cuda_ranks)


@pytest.mark.parametrize("cuda_ranks", ["0", "0,1"])
def test_stream_negative_control_detects_missing_dependency(run_two_ranks, cuda_ranks):
    run(run_two_ranks, "stream_negative_control", cuda_ranks)


@pytest.mark.parametrize("cuda_ranks", ["0", "0,1"])
def test_overlap_submission_is_nonblocking(run_two_ranks, cuda_ranks):
    results = run(run_two_ranks, "overlap", cuda_ranks, timeout=300)
    for _, out in results:
        print([l for l in out.splitlines() if "comm=" in l])
