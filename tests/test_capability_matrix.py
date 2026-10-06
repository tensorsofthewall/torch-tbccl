"""The supported ProcessGroup surface is a tested contract. tools/capability_matrix.py drives every operation x dtype x payload shape x root/destination
through the public torch.distributed API on loopback (one launch per world size / device mode) and records what happened; tools/capability_matrix.expected_status holds
the documented surface. Every observed cell must equal the documented one, so neither a regression nor a silently newly-working cell passes.
The full matrix (W1-W4 x cpu/cuda0/cudaall, written to capability_matrix.json) is regenerated with the tool itself."""
import importlib.util
import os

import pytest
import torch

import torch_tbccl

pytestmark = pytest.mark.multiprocess
TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "capability_matrix.py")
spec = importlib.util.spec_from_file_location("capability_matrix", TOOL)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)

HAS_CUDA = torch.cuda.is_available() and torch_tbccl.compiled_features()["cuda"]
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
CONFIGS = [
    ("cpu", 1), ("cpu", 2), ("cpu", 3), ("cpu", 4),
    pytest.param("cuda0", 2, marks=needs_cuda), pytest.param("cuda0", 3, marks=needs_cuda), pytest.param("cudaall", 2, marks=needs_cuda),
]


@pytest.mark.parametrize("devmode,world", CONFIGS)
def test_observed_surface_equals_the_documented_surface(devmode, world):
    lines, hung, rcs = matrix.run_config(world, devmode, hang=120)
    records = matrix.aggregate(world, devmode, lines, hung, rcs)
    assert not hung, f"hang after {hung}"
    assert rcs == [0] * world, rcs
    failures = [r for r in records if r["status"] in ("WRONG", "ERROR", "HANG")]
    assert not failures, failures[:5]
    assert not matrix.mismatches(records), [(r["op"], r["dtype"], r["status"], r["detail"][:100]) for r in matrix.mismatches(records)][:8]
    assert len(records) >= 139, len(records)


def test_expected_surface_documents_the_known_restrictions():
    e = matrix.expected_status
    assert e("all_reduce SUM", "bfloat16", 2) == "PASS" and e("all_reduce SUM", "bfloat16", 3) == "REJECTED"
    assert e("all_reduce PRODUCT", "float32", 2) == "REJECTED" and e("all_reduce MAX", "int32", 2) == "REJECTED"
    assert e("gather", "float32", 2) == "PASS" and e("gather", "float32", 4) == "REJECTED"
    assert e("send/recv", "bfloat16", 1) == "NA" and e("broadcast", "float32", 1) == "REJECTED"
