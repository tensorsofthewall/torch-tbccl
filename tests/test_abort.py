import pytest
import torch

import torch_tbccl


def run(run_two_ranks, mode):
    results = run_two_ranks("_worker_abort.py", timeout=90, extra_env={"TEST_MODE": mode})
    for rc, out in results:
        assert rc == 0, out
    assert "rank 1 ok" in results[1][1] and "rank 0 ok" in results[0][1]
    return results


@pytest.mark.parametrize("mode", ["abort", "destroy", "timeout_then_abort", "lifetime", "timeout_then_success"])
def test_abort_scenarios(run_two_ranks, mode):
    run(run_two_ranks, mode)


# Phase 49: the same silent-peer abort semantics for the new reduction dtypes (the Work and Future go terminal, the buffer is not touched after abort, teardown is bounded).
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int8", "uint8"])
@pytest.mark.parametrize("mode", ["abort", "destroy", "lifetime", "timeout_then_abort"])
def test_abort_scenarios_low_precision(run_two_ranks, mode, dtype):
    results = run_two_ranks("_worker_abort.py", timeout=90, extra_env={"TEST_MODE": mode, "ABORT_DTYPE": dtype})
    for rc, out in results:
        assert rc == 0, out
    assert "rank 1 ok" in results[1][1] and "rank 0 ok" in results[0][1]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]),
    reason="needs a CUDA GPU and a CUDA-enabled torch-tbccl build",
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("mode", ["abort", "lifetime"])
def test_abort_scenarios_low_precision_cuda(run_two_ranks, mode, dtype):
    results = run_two_ranks("_worker_abort.py", timeout=120, extra_env={"TEST_MODE": mode, "ABORT_DTYPE": dtype, "ABORT_DEVICE": "cuda"})
    for rc, out in results:
        assert rc == 0, out
    assert "rank 1 ok" in results[1][1] and "rank 0 ok" in results[0][1]
