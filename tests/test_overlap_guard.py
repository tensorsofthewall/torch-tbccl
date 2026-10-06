"""Phase 71: collective and point-to-point operations in flight together on one group are refused, never silently corrupted (tests/_worker_overlap.py)."""
import pytest


@pytest.mark.multiprocess
def test_overlapping_collective_and_p2p_is_refused_and_the_group_survives(run_two_ranks):
    for rc, out in run_two_ranks("_worker_overlap.py", timeout=120, extra_env={"AUTO_PORT": "1", "OMP_NUM_THREADS": "2"}):
        assert rc == 0, out
        assert out.strip().splitlines()[-1].endswith("ok"), out
