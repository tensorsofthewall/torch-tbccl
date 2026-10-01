import torch_tbccl


def test_versions_are_distinct_strings():
    assert torch_tbccl.__version__
    assert torch_tbccl.runtime_version() not in ("", "unknown")
    assert torch_tbccl.TESTED_TORCH_VERSION


def test_compiled_features():
    f = torch_tbccl.compiled_features()
    assert f["cpu"] is True
    assert f["mps"] is False
    assert isinstance(f["cuda"], bool)
