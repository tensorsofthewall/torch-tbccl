import pytest


@pytest.mark.parametrize("trace", ["1", "0"])
def test_trace(run_two_ranks, trace):
    results = run_two_ranks("_worker_trace.py", extra_env={"TORCH_TBCCL_TRACE": trace})
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]
