"""Real DistributedDataParallel with an MPS rank next to a CPU rank on one Mac (examples/ddp_mlp.py, checked against the single-process reference).
Both ranks must stay within the script's tolerances; the optimizer step runs on each rank's own device, so CPU and MPS parameters may differ by an ulp (reported as
cross_rank_max_abs_diff)."""
import pytest
import torch

import torch_tbccl
from test_ddp import run_ddp

pytestmark = [
    pytest.mark.ddp,
    pytest.mark.multiprocess,
    pytest.mark.skipif(not (torch.backends.mps.is_available() and torch_tbccl.compiled_features()["mps"]), reason="needs MPS and an MPS-enabled build"),
]


@pytest.mark.parametrize("devices", ["cpu,mps", "mps,cpu"])
def test_ddp_with_an_mps_rank_matches_the_reference(run_ranks, devices):
    res = run_ddp(run_ranks, 2, "--steps", "10", "--devices", devices)
    assert sorted(r["device"] for r in res) == ["cpu", "mps:0"] or sorted(r["device"] for r in res) == ["cpu", "mps"], res
    assert all(r["last_loss"] < r["first_loss"] for r in res)
    assert all(r["cross_rank_max_abs_diff"] <= 1e-6 for r in res), res


def test_ddp_with_an_mps_rank_repeats_create_train_destroy(run_ranks):
    res = run_ddp(run_ranks, 2, "--steps", "6", "--devices", "cpu,mps", "--cycles", "3")
    assert len(res) == 6


def test_ddp_with_an_mps_rank_and_unused_parameters(run_ranks):
    run_ddp(run_ranks, 2, "--steps", "6", "--devices", "mps,cpu", "--unused")


def test_ddp_with_an_mps_rank_and_tiny_buckets(run_ranks):
    run_ddp(run_ranks, 2, "--steps", "6", "--devices", "cpu,mps", "--bucket-cap-mb", "0.0005")
