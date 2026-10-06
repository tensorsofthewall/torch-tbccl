"""Inspect a built torch-tbccl wheel WITHOUT importing it: required members, metadata, the compiled extension's dynamic dependencies and runtime search path.

    python tools/p71_package_inspect.py dist/torch_tbccl-*.whl [--json OUT]

Fails (exit 1) on: a missing module / compiled extension / metadata, a wheel tag that does not match the running platform, an extension that references the build
directory (absolute RUNPATH/RPATH entries outside system library directories, or an absolute install name), or metadata that does not pin the torch series.
What the wheel needs at run time is printed: the shared libraries the extension asks the loader for, and where it may look for them.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

REQUIRED = ["torch_tbccl/__init__.py", "torch_tbccl/_version.py", "torch_tbccl/info.py"]
SYSTEM_DIRS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/local/cuda", "/opt/cuda", "/usr/local/lib")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def dynamic_info(path):
    """(needed, search_paths, install_names) of a shared object, via readelf (Linux) or otool (macOS)."""
    if sys.platform == "darwin":
        out = run(["otool", "-L", path]) or ""
        needed = [l.split()[0] for l in out.splitlines()[1:] if l.strip()]
        lc = run(["otool", "-l", path]) or ""
        rpaths = re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)", lc)
        ident = re.findall(r"cmd LC_ID_DYLIB\n\s+cmdsize \d+\n\s+name (\S+)", lc)
        return needed, rpaths, ident
    out = run(["readelf", "-d", path]) or ""
    needed = re.findall(r"NEEDED\)\s+Shared library: \[([^\]]+)\]", out)
    rpaths = []
    for m in re.finditer(r"\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]*)\]", out):
        rpaths += [p for p in m.group(1).split(":") if p]
    return needed, rpaths, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    problems, report = [], {"wheel": os.path.basename(a.wheel)}
    z = zipfile.ZipFile(a.wheel)
    names = z.namelist()
    report["members"] = names
    for m in REQUIRED:
        if m not in names:
            problems.append(f"missing module {m}")
    exts = [n for n in names if re.match(r"torch_tbccl/_C\..*\.(so|pyd)$", n)]
    if len(exts) != 1:
        problems.append(f"expected exactly one compiled extension torch_tbccl/_C.*, found {exts}")
    meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    if meta is None:
        problems.append("missing METADATA")
        text = ""
    else:
        text = z.read(meta).decode()
    report["requires_dist"] = re.findall(r"^Requires-Dist: (.+)$", text, re.M)
    report["requires_python"] = (re.search(r"^Requires-Python: (.+)$", text, re.M) or [None, None])[1]
    if not any(r.startswith("torch<2.14") or r.startswith("torch>=2.13") for r in report["requires_dist"]):
        problems.append(f"metadata does not pin the torch series: {report['requires_dist']}")
    tag = re.search(r"-(cp\d+)-(cp\d+)-(\w+)\.whl$", a.wheel)
    report["tag"] = tag.groups() if tag else None
    want_py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not tag or tag.group(1) != want_py:
        problems.append(f"wheel python tag {tag.group(1) if tag else None} != running {want_py}")
    if not any(n.endswith(".dist-info/WHEEL") for n in names):
        problems.append("missing WHEEL file")

    if len(exts) == 1:
        with tempfile.TemporaryDirectory() as d:
            z.extract(exts[0], d)
            so = os.path.join(d, exts[0])
            needed, rpaths, ident = dynamic_info(so)
            report["needed"], report["search_paths"], report["install_names"] = needed, rpaths, ident
            for p in rpaths:
                if not (p.startswith("$ORIGIN") or p.startswith("@loader_path") or p.startswith("@executable_path") or p.startswith(SYSTEM_DIRS)):
                    problems.append(f"absolute runtime search path outside the system directories: {p}")
            for i in ident:
                if i.startswith("/") and not i.startswith(SYSTEM_DIRS):
                    problems.append(f"absolute install name {i}")
            blob = open(so, "rb").read()
            leaks = sorted({m.decode(errors="replace") for m in re.findall(rb"/(?:home|Users|mnt)/[\w.\-]+", blob)})
            report["embedded_home_paths"] = leaks
            if leaks:
                problems.append(f"the extension embeds build-machine paths: {leaks}")
    report["problems"] = problems
    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
