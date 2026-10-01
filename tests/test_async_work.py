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
