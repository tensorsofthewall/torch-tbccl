"""Real DistributedDataParallel over torch-tbccl, CPU<->CPU on loopback (examples/ddp_train.py)."""
import pytest

SCRIPT = "../examples/ddp_train.py"


def run(run_two_ranks, *args, timeout=240, trace=False):
    results = run_two_ranks(
        SCRIPT, timeout=timeout, args=args, extra_env={"TORCH_TBCCL_TRACE": "1"} if trace else None
    )
    for rc, out in results:
        assert rc == 0, out
    return [out for _, out in results]


def test_train_matches_reference(run_two_ranks):
    outs = run(run_two_ranks, "--steps", "20")
    for out in outs:
        assert "constructor sync (parameters == rank 0's seed): OK" in out
        assert "20 steps" in out and out.strip().endswith("OK")


def test_multiple_buckets(run_two_ranks):
    # 0.05 MB cap on a ~0.1 MB gradient set => at least two buckets per backward.
    outs = run(run_two_ranks, "--steps", "6", "--bucket-cap-mb", "0.05", trace=True)
    for out in outs:
        line = [l for l in out.splitlines() if "collectives recorded" in l][0]
        n = int(line.split("'allreduce':")[1].split("}")[0].split(",")[0])
        # step 0 runs the initial (1 MiB-first-bucket) layout = 1 allreduce; after DDP rebuilds its
        # buckets from the observed gradient order, the 5 later steps use the 0.05 MB cap (>= 2 each)
        assert n >= 1 + 2 * 5, line


def test_buffer_broadcast(run_two_ranks):
    for out in run(run_two_ranks, "--mode", "buffers"):
        assert "forward buffer broadcast (rank 0's value everywhere): OK" in out


@pytest.mark.parametrize("mode", ["count-mismatch", "shape-mismatch"])
def test_mismatch_fails_instead_of_hanging(run_two_ranks, mode):
    outs = run(run_two_ranks, "--mode", mode, "--timeout", "20", timeout=60)
    assert "OK" in outs[1]
