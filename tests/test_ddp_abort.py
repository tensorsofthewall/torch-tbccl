import sys

SCRIPT = "../examples/ddp_silent_peer.py"


def test_ddp_silent_peer_abort(run_two_ranks):
    results = run_two_ranks(SCRIPT, timeout=120, extra_env={"OMP_NUM_THREADS": "4"})
    for rc, out in results:
        assert rc == 0, out
    out0 = results[0][1]
    assert "backward raised" in out0 and "abort" in out0 and "rank 0 ok" in out0
    assert "rank 1 ok" in results[1][1]


def test_peer_observes_remote_abort(run_two_ranks):
    results = run_two_ranks("_worker_abort.py", timeout=90, extra_env={"TEST_MODE": "abort", "OBSERVE_REMOTE": "1"})
    for rc, out in results:
        assert rc == 0, out
    assert "observed remote abort" in results[1][1]
