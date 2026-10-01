def test_two_rank_cpu_allreduce(run_two_ranks):
    results = run_two_ranks("_worker_allreduce.py", timeout=120)
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


def test_invalid_calls_fail_promptly_and_leave_group_usable(run_two_ranks):
    results = run_two_ranks("_worker_errors.py", timeout=60)
    for rc, out in results:
        assert rc == 0, out
