"""Packaging discipline. A wheel is built from the source tree, inspected without importing it, installed into a brand-new venv, and exercised from an unrelated
directory with no PYTHONPATH. Editable installs hide packaging bugs (the vLLM 0.31 alignment work found one in vllm-tbccl); this cannot.

Needs TBCCL_ROOT (an installed TBCCL prefix) and the `build` module; the clean-venv part also needs `uv` and the package index/cache for torch (set TORCH_TBCCL_TEST_CLEAN_INSTALL=1).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from conftest import HERE

pytestmark = pytest.mark.packaging
ROOT = os.path.dirname(HERE)
TBCCL_ROOT = os.environ.get("TBCCL_ROOT")
UV = shutil.which("uv") or os.path.expanduser("~/mambaforge/bin/uv")
needs_prefix = pytest.mark.skipif(not TBCCL_ROOT, reason="TBCCL_ROOT is not set (needs an installed TBCCL prefix to build a wheel)")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("wheel")
    p = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", str(out), ROOT], capture_output=True, text=True, env=dict(os.environ, TBCCL_ROOT=TBCCL_ROOT or ""))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    (whl,) = glob.glob(str(out / "*.whl"))
    return whl


@needs_prefix
def test_wheel_contents_metadata_and_runtime_paths(wheel):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "package_inspect.py"), wheel, "--allow-local-platform"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads(p.stdout)
    assert "torch_tbccl/__init__.py" in rep["members"] and "torch_tbccl/info.py" in rep["members"]
    assert any(m.startswith("torch_tbccl/_C.") for m in rep["members"])
    assert rep["problems"] == []
    assert rep["metal_frameworks"] == (sys.platform == "darwin" and torch.backends.mps.is_built())  # Metal only in the macOS MPS wheel
    assert not [n for n in rep["needed"] if "tbccl" in os.path.basename(n)]


@needs_prefix
def test_inspector_rejects_an_unpublishable_platform_tag(wheel):
    if sys.platform == "darwin":
        pytest.skip("a locally built macOS wheel already carries a macosx tag")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "package_inspect.py"), wheel], capture_output=True, text=True)
    assert p.returncode == 1 and "must be manylinux_*" in p.stdout


@needs_prefix
def test_inspector_catches_a_missing_module(wheel, tmp_path):
    import zipfile

    broken = str(tmp_path / os.path.basename(wheel))
    with zipfile.ZipFile(wheel) as zin, zipfile.ZipFile(broken, "w") as zout:
        for item in zin.infolist():
            if item.filename != "torch_tbccl/_version.py":
                zout.writestr(item, zin.read(item.filename))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "package_inspect.py"), broken, "--allow-local-platform"], capture_output=True, text=True)
    assert p.returncode == 1 and "missing module torch_tbccl/_version.py" in p.stdout


@needs_prefix
def test_unsupported_c_abi_prefix_fails_the_build_clearly(tmp_path):
    prefix = tmp_path / "prefix"
    shutil.copytree(os.path.join(TBCCL_ROOT, "include"), prefix / "include")
    hdr = prefix / "include" / "tbccl" / "tbccl.h"
    hdr.write_text(hdr.read_text().replace("#define TBCCL_C_ABI_VERSION 1u", "#define TBCCL_C_ABI_VERSION 2u"))
    p = subprocess.run([sys.executable, "setup.py", "--name"], cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, TBCCL_ROOT=str(prefix)))
    assert p.returncode != 0
    assert "has C ABI 2" in p.stdout + p.stderr and "supports C ABI [1]" in p.stdout + p.stderr


@needs_prefix
@pytest.mark.skipif(os.environ.get("TORCH_TBCCL_TEST_CLEAN_INSTALL") != "1", reason="set TORCH_TBCCL_TEST_CLEAN_INSTALL=1 (creates a venv, installs torch from the index/cache)")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")), pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"))])
def test_clean_venv_install_and_real_w2_collective(wheel, tmp_path, device):
    if os.environ.get("TORCH_TBCCL_TEST_SCRATCH"):  # a torch+CUDA venv is ~6 GB: put it on a disk with room when /tmp is a small tmpfs
        import tempfile

        tmp_path = type(tmp_path)(tempfile.mkdtemp(dir=os.environ["TORCH_TBCCL_TEST_SCRATCH"]))
    venv = tmp_path / "venv"
    index = ["--index-url", "https://download.pytorch.org/whl/cu130", "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"]
    subprocess.run([UV, "venv", "--python", "3.13", str(venv)], check=True, capture_output=True)
    py = str(venv / "bin" / "python")
    inst = subprocess.run([UV, "pip", "install", "--python", py, f"torch=={torch.__version__}", wheel, *index], capture_output=True, text=True)
    assert inst.returncode == 0, inst.stderr[-2000:]
    work = tmp_path / "unrelated"
    work.mkdir()
    shutil.copy(os.path.join(ROOT, "tools", "clean_install_check.py"), work)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "TBCCL_ROOT")}
    p = subprocess.run([py, "clean_install_check.py", "--device", device], cwd=work, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
