import pytest


def run(run_two_ranks, mode):
    results = run_two_ranks("_worker_abort.py", timeout=90, extra_env={"TEST_MODE": mode})
    for rc, out in results:
        assert rc == 0, out
    assert "rank 1 ok" in results[1][1] and "rank 0 ok" in results[0][1]
    return results


@pytest.mark.parametrize("mode", ["abort", "destroy", "timeout_then_abort", "lifetime", "timeout_then_success"])
def test_abort_scenarios(run_two_ranks, mode):
    run(run_two_ranks, mode)
