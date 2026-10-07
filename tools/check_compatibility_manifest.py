#!/usr/bin/env python3
"""Check that compatibility.json agrees with torch_tbccl/_version.py and pyproject.toml.

The sources stay the single definition of the compatibility ranges; this check fails when the manifest drifts from them.
The TBCCL C ABI and wire protocol values themselves are defined by TBCCL; this project only records which of them it supports.
The file is read statically: the package is never imported.
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "compatibility.json").read_text())

consts = {}
for node in ast.parse((ROOT / "torch_tbccl" / "_version.py").read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        consts[node.targets[0].id] = ast.literal_eval(node.value)

pyproject = (ROOT / "pyproject.toml").read_text()
requires_python = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject).group(1)

checks = {
    "package.development_version": (manifest["package"]["development_version"], consts["__version__"]),
    "frameworks.torch.series": (tuple(manifest["frameworks"]["torch"]["series"]), consts["TESTED_TORCH_SERIES"]),
    "frameworks.torch.tested_version": (manifest["frameworks"]["torch"]["tested_version"], consts["TESTED_TORCH_VERSION"]),
    "requires_tbccl.c_abi": (tuple(manifest["requires_tbccl"]["c_abi"]), consts["SUPPORTED_C_ABI"]),
    "requires_tbccl.wire_protocol": (tuple(manifest["requires_tbccl"]["wire_protocol"]), consts["TESTED_WIRE_PROTOCOL"]),
    "python.requires": (manifest["python"]["requires"], requires_python),
}
bad = [k for k, (declared, actual) in checks.items() if declared != actual]
for k in bad:
    print(f"MISMATCH {k}: manifest {checks[k][0]!r}, sources {checks[k][1]!r}")
if bad:
    sys.exit(1)
print("compatibility.json agrees with the sources")
