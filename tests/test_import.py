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


def test_extension_matches_the_running_torch_series():
    import torch

    built = torch_tbccl.built_with_torch()
    assert torch_tbccl._series(built) == torch_tbccl._series(torch.__version__)
    assert torch_tbccl._series(torch.__version__) in torch_tbccl.TESTED_TORCH_SERIES


def test_torch_abi_mismatch_is_a_clear_import_error(monkeypatch):
    import pytest

    monkeypatch.setattr(torch_tbccl._C, "built_with_torch", lambda: "2.12.3", raising=False)
    with pytest.raises(ImportError, match=r"compiled against torch 2\.12\.3 .* Rebuild it"):
        torch_tbccl._check_torch_abi()
    monkeypatch.setenv("TORCH_TBCCL_ALLOW_TORCH_MISMATCH", "1")
    torch_tbccl._check_torch_abi()  # explicit opt-out
