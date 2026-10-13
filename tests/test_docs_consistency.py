"""Documentation contract checks: the documented environment variables, versions and quick-start example match the code and the manifest."""
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def _read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _source_env_vars():
    names = set()
    for base in ("csrc", "torch_tbccl"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
            for name in files:
                if name.endswith((".py", ".cpp", ".hpp", ".mm")):
                    with open(os.path.join(dirpath, name), errors="ignore") as f:
                        names.update(re.findall(r'"((?:TORCH_TBCCL|TBCCL)_[A-Z0-9_]+)"', f.read()))
    return names


def test_runtime_environment_variables_are_documented_and_exist():
    page = _read("docs", "reference", "configuration.md")
    documented = set(re.findall(r"`((?:TORCH_TBCCL|TBCCL)_[A-Z0-9_]+)`", page))
    in_source = _source_env_vars()
    undocumented = {v for v in in_source if v not in documented}
    assert not undocumented, f"environment variables read by the code but not documented: {sorted(undocumented)}"
    build_only = {"TBCCL_ROOT", "TORCH_TBCCL_SANITIZE", "TORCH_TBCCL_TEST_CLEAN_INSTALL", "TORCH_TBCCL_TEST_SCRATCH"}
    stale = {v for v in documented if v not in in_source and v not in build_only}
    assert not stale, f"documented environment variables that the code does not read: {sorted(stale)}"


def test_documented_versions_match_the_manifest():
    manifest = json.loads(_read("compatibility.json"))
    version = manifest["package"]["development_version"]
    compat = _read("docs", "reference", "compatibility.md")
    install = _read("docs", "getting-started", "install.md")
    assert version in compat and version in install
    for series in manifest["frameworks"]["torch"]["series"]:
        assert f"{series}" in compat
    for abi in manifest["requires_tbccl"]["c_abi"]:
        assert f"| {abi} " in compat or f"`SUPPORTED_C_ABI`" in compat
    for wire in manifest["requires_tbccl"]["wire_protocol"]:
        assert str(wire) in compat


def test_docs_do_not_claim_registry_publication():
    text = _read("docs", "getting-started", "install.md") + _read("docs", "reference", "compatibility.md")
    assert "not published on PyPI or TestPyPI" in text
    assert "pip install --pre" not in text and "test.pypi.org" not in text


@pytest.mark.multiprocess
@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(sys.executable), "torchrun")), reason="torchrun not installed next to the interpreter")
def test_quickstart_example_runs_under_torchrun(tmp_path):
    text = _read("docs", "getting-started", "quickstart.md")
    code = re.search(r"`allreduce.py`:\n\n```python\n(.*?)```", text, re.S).group(1)
    script = tmp_path / "allreduce.py"
    script.write_text(code)
    env = dict(os.environ, TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", OMP_NUM_THREADS="2")
    torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
    p = subprocess.run([torchrun, "--standalone", "--local-addr", "127.0.0.1", "--nproc-per-node", "2", str(script)], env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    assert p.stdout.count("tensor([3.])") == 2 and p.stdout.count("tbccl") == 2, p.stdout
