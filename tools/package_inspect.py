"""Inspect a built torch-tbccl wheel WITHOUT importing it: required members, metadata, the compiled extension's dynamic dependencies and runtime search path.

    python tools/package_inspect.py dist/torch_tbccl-*.whl [--json OUT] [--allow-local-platform]

Fails (exit 1) on: a missing module / compiled extension / metadata / licence, a wheel tag that is not valid for publication (Linux: manylinux_*, never a bare linux_*;
macOS: macosx_*_arm64) or does not match the running platform, a version that disagrees with torch_tbccl/_version.py, development files or build-machine paths in any member,
an extension that references the build
directory (absolute RUNPATH/RPATH entries outside system library directories, or an absolute install name), or metadata that does not pin the torch series.
It also fails on a run-time dependency on libtbccl, on Apple frameworks in a non-macOS wheel, and reports whether the macOS wheel links Metal (the MPS adapter).
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


def parse_tags(filename):
    """(python, abi, [platform, ...]) from a wheel file name; the platform part may be a dotted compressed set."""
    parts = os.path.basename(filename)[:-4].split("-")
    return parts[-3], parts[-2], parts[-1].split(".")


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
    ap.add_argument("--allow-local-platform", action="store_true", help="accept a bare linux_* platform tag (a locally built wheel; never for publication)")
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
    py_tag, abi_tag, plats = parse_tags(a.wheel)
    report["tag"] = [py_tag, abi_tag, plats]
    want_py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if py_tag != want_py or abi_tag != want_py:
        problems.append(f"wheel python/abi tag {py_tag}/{abi_tag} != running {want_py}")
    if sys.platform == "darwin":
        if not plats or not all(re.fullmatch(r"macosx_\d+_\d+_arm64", p) for p in plats):
            problems.append(f"macOS wheel platform tag must be macosx_<major>_<minor>_arm64, got {plats}")
    elif not a.allow_local_platform:
        if not plats or not all(p.startswith("manylinux_") for p in plats):
            problems.append(f"Linux wheel platform tag must be manylinux_*, got {plats} (a bare linux_x86_64 wheel cannot be published)")
    if not any(re.fullmatch(r".*\.dist-info/licenses/LICENSE", n) or re.fullmatch(r".*\.dist-info/LICENSE", n) for n in names):
        problems.append("the licence file is not in the wheel")
    ver_in_name = os.path.basename(a.wheel).split("-")[1]
    ver_src = re.search(r'__version__ = "([^"]+)"', z.read("torch_tbccl/_version.py").decode()) if "torch_tbccl/_version.py" in names else None
    if not ver_src or ver_src.group(1) != ver_in_name:
        problems.append(f"wheel version {ver_in_name} != torch_tbccl/_version.py {ver_src.group(1) if ver_src else None}")
    for n in names:
        if re.match(r"(tests|tools|docs|examples|csrc|\.github)/", n) or n.endswith((".pyc", ".o", ".a")):
            problems.append(f"development file in the wheel: {n}")
        if not n.endswith((".so", ".pyc")):
            private = re.findall(rb"/(?:home|Users|mnt|tmp)/[\w.\-]+", z.read(n))
            if private:
                problems.append(f"{n} contains private or build paths: {sorted(set(x.decode() for x in private))}")
    if not any(n.endswith(".dist-info/WHEEL") for n in names):
        problems.append("missing WHEEL file")

    if len(exts) == 1:
        with tempfile.TemporaryDirectory() as d:
            z.extract(exts[0], d)
            so = os.path.join(d, exts[0])
            needed, rpaths, ident = dynamic_info(so)
            report["needed"], report["search_paths"], report["install_names"] = needed, rpaths, ident
            # TBCCL is linked statically: no dependency on a libtbccl at run time. Metal/Foundation/libobjc (system frameworks) belong to the macOS MPS adapter only.
            tbccl_deps = [n for n in needed if "tbccl" in os.path.basename(n)]
            if tbccl_deps:
                problems.append(f"the extension depends on a shared libtbccl at run time: {tbccl_deps}")
            apple = [n for n in needed if re.search(r"/(Metal|Foundation)\.framework/|libobjc", n)]
            report["metal_frameworks"] = bool(apple)
            if apple and sys.platform != "darwin":
                problems.append(f"a non-macOS wheel depends on Apple frameworks: {apple}")
            for n in apple:
                if not n.startswith("/System/Library/") and not n.startswith("/usr/lib/"):
                    problems.append(f"Apple framework dependency outside the system directories: {n}")
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
