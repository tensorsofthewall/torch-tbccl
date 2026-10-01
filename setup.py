"""Build torch_tbccl._C against an INSTALLED TBCCL prefix (TBCCL_ROOT).

Never reaches into the TBCCL source tree: only <TBCCL_ROOT>/include/tbccl
and the installed library are used.
"""
import glob
import os
import re
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))


def fail(msg):
    sys.exit(f"torch-tbccl: {msg}")


def find_tbccl():
    root = os.environ.get("TBCCL_ROOT")
    if not root:
        fail("TBCCL_ROOT is not set. Point it at an installed TBCCL prefix, e.g.\n"
             "  TBCCL_ROOT=/path/to/tbccl-install uv pip install -e . --no-build-isolation")
    root = os.path.abspath(root)
    inc = os.path.join(root, "include")
    if not os.path.isfile(os.path.join(inc, "tbccl", "communicator.hpp")):
        fail(f"{inc}/tbccl/communicator.hpp not found; TBCCL_ROOT is not an installed TBCCL prefix")

    libs = []
    for libdir in ("lib", "lib64"):
        for name in ("libtbccl.so", "libtbccl.dylib", "libtbccl.a"):
            p = os.path.join(root, libdir, name)
            if os.path.isfile(p):
                libs.append(p)
        if libs:
            break
    if not libs:
        fail(f"no libtbccl.(so|dylib|a) under {root}/lib or lib64")

    version = "unknown"
    for f in glob.glob(os.path.join(root, "lib*", "cmake", "TBCCL", "TBCCLConfigVersion.cmake")):
        m = re.search(r'set\(PACKAGE_VERSION "([^"]+)"\)', open(f).read())
        if m:
            version = m.group(1)
    return root, inc, libs[0], version


root, inc, lib, tbccl_version = find_tbccl()
print(f"torch-tbccl: using TBCCL {tbccl_version} from {root} ({os.path.basename(lib)})")

ext = CppExtension(
    name="torch_tbccl._C",
    sources=sorted(glob.glob(os.path.join("csrc", "*.cpp"))),
    include_dirs=[inc],
    extra_objects=[lib],
    define_macros=[("TORCH_TBCCL_LINKED_TBCCL_VERSION", f'"{tbccl_version}"')],
    extra_compile_args=["-O2", "-Wall"],
    extra_link_args=["-pthread"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
