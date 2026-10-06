import sys

import torch

import torch_tbccl


def test_versions_are_distinct_strings():
    assert torch_tbccl.__version__
    assert torch_tbccl.runtime_version() not in ("", "unknown")
    assert torch_tbccl.TESTED_TORCH_VERSION


def test_compiled_features():
    f = torch_tbccl.compiled_features()
    assert f["cpu"] is True
    assert f["mps"] is (sys.platform == "darwin" and torch.backends.mps.is_built())  # MPS only in the macOS build, never on Linux
    assert isinstance(f["cuda"], bool)
    assert not (f["cuda"] and f["mps"])
    assert torch_tbccl.supported_devices() == [d for d in ("cpu", "cuda", "mps") if f[d]]


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


def test_info_describes_every_layer():
    info = torch_tbccl.info()
    assert info["torch_tbccl"] == torch_tbccl.__version__ and info["libtbccl"] == torch_tbccl.runtime_version()
    assert info["c_abi"] == torch_tbccl.c_abi_version() and info["wire_protocol"] == torch_tbccl.wire_protocol_version()
    assert info["registered"] is True and "cpu" in info["devices"]


def test_compiled_against_a_supported_libtbccl_abi():
    assert torch_tbccl.c_abi_version() in torch_tbccl.SUPPORTED_C_ABI
    assert torch_tbccl.wire_protocol_version() in torch_tbccl.TESTED_WIRE_PROTOCOL


def test_libtbccl_c_abi_mismatch_is_a_clear_import_error(monkeypatch):
    import pytest

    monkeypatch.setattr(torch_tbccl._C, "c_abi_version", lambda: 2, raising=False)
    with pytest.raises(ImportError, match=r"supports libtbccl C ABI \[1\] but this build was compiled against C ABI 2 .*Rebuild torch-tbccl"):
        torch_tbccl._check_tbccl_abi()


def test_untested_wire_protocol_warns(monkeypatch):
    import pytest

    monkeypatch.setattr(torch_tbccl._C, "wire_protocol_version", lambda: 4, raising=False)
    with pytest.warns(UserWarning, match="wire protocol 4"):
        torch_tbccl._check_tbccl_abi()


def test_info_module_runs():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "torch_tbccl.info", "--json"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    import json

    assert json.loads(out.stdout)["c_abi"] == torch_tbccl.c_abi_version()
