"""Build torch_tbccl._C against an INSTALLED TBCCL prefix (TBCCL_ROOT).

Never reaches into the TBCCL source tree: only <TBCCL_ROOT>/include/tbccl
and the installed library are used.
"""
import glob
import os
import re
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))


def fail(msg):
    sys.exit(f"torch-tbccl: {msg}")


def version_info():
    ns = {}
    exec(open(os.path.join(HERE, "torch_tbccl", "_version.py")).read(), ns)  # plain constants; importing the package would need the extension
    return ns


def check_compatibility(root, inc):
    """Refuse a TBCCL prefix whose C ABI this package does not support (the extension is linked statically, so the headers decide what runs)."""
    ns = version_info()
    hdr = os.path.join(inc, "tbccl", "tbccl.h")
    m = os.path.isfile(hdr) and re.search(r"#\s*define\s+TBCCL_C_ABI_VERSION\s+(\d+)", open(hdr).read())
    if not m:
        fail(f"{hdr} does not define TBCCL_C_ABI_VERSION; {root} is not a TBCCL prefix this package can build against (needs TBCCL >= 0.5.0)")
    abi = int(m.group(1))
    if abi not in ns["SUPPORTED_C_ABI"]:
        fail(f"the TBCCL prefix {root} has C ABI {abi}; torch-tbccl {ns['__version__']} supports C ABI {list(ns['SUPPORTED_C_ABI'])}. "
             "Point TBCCL_ROOT at a compatible prefix (or use a torch-tbccl release that supports this ABI).")
    w = re.search(r"kWireProtocolVersion\s*=\s*(\d+)", open(os.path.join(inc, "tbccl", "rank_directory.hpp")).read())
    if w and int(w.group(1)) not in ns["TESTED_WIRE_PROTOCOL"]:
        print(f"torch-tbccl: warning: wire protocol {w.group(1)} of this prefix has not been tested (tested: {list(ns['TESTED_WIRE_PROTOCOL'])})")
    return abi


def find_tbccl():
    root = os.environ.get("TBCCL_ROOT")
    if not root:
        fail("TBCCL_ROOT is not set. Point it at an installed TBCCL prefix, e.g.\n"
             "  TBCCL_ROOT=/path/to/tbccl-install uv pip install -e . --no-build-isolation")
    root = os.path.abspath(root)
    inc = os.path.join(root, "include")
    if not os.path.isfile(os.path.join(inc, "tbccl", "communicator.hpp")):
        fail(f"{inc}/tbccl/communicator.hpp not found; TBCCL_ROOT is not an installed TBCCL prefix")

    check_compatibility(root, inc)

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
    cuda_lib = None
    for libdir in ("lib", "lib64"):
        p = os.path.join(root, libdir, "libtbccl_cuda.a")
        if os.path.isfile(p):
            cuda_lib = p
    return root, inc, libs[0], version, cuda_lib


def find_cudart():
    """Headers from a CUDA toolkit (CUDA_HOME); the runtime library from the copy torch
    itself ships/loads when present, so there is exactly one cudart in the process."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if torch.version.cuda is None:
        return None
    site = os.path.dirname(os.path.dirname(torch.__file__))
    pip_inc = pip_lib = None
    for inc in sorted(glob.glob(os.path.join(site, "nvidia", "cu*", "include"))):
        libdir = os.path.join(os.path.dirname(inc), "lib")
        if glob.glob(os.path.join(libdir, "libcudart.so*")):
            pip_inc, pip_lib = inc, libdir
    inc = None
    for cand in (os.path.join(CUDA_HOME, "include") if CUDA_HOME else None, pip_inc):
        if cand and os.path.isfile(os.path.join(cand, "crt", "host_defines.h")):
            inc = cand
            break
    libdir = pip_lib or (os.path.join(CUDA_HOME, "lib64") if CUDA_HOME else None)
    return (inc, libdir) if inc and libdir else None


root, inc, lib, tbccl_version, cuda_lib = find_tbccl()
print(f"torch-tbccl: using TBCCL {tbccl_version} from {root} ({os.path.basename(lib)})")

include_dirs = [inc]
library_dirs = []
libraries = []
objects = [lib]
macros = [("TORCH_TBCCL_LINKED_TBCCL_VERSION", f'"{tbccl_version}"')]
link_args = ["-pthread"]

cudart = find_cudart() if cuda_lib else None
if cuda_lib and cudart:
    cuda_inc, cuda_libdir = cudart
    print(f"torch-tbccl: CUDA enabled (tbccl_cuda + cudart from {cuda_libdir})")
    include_dirs.append(cuda_inc)
    library_dirs.append(cuda_libdir)
    cudart_so = sorted(glob.glob(os.path.join(cuda_libdir, "libcudart.so.*")))[0]
    objects = [cuda_lib, lib]  # tbccl_cuda depends on tbccl
    libraries += ["c10_cuda", "torch_cuda"]
    # The runtime path must not name the build directory: relative to the installed package when cudart comes from the pip nvidia/ tree next to it
    # (the layout every torch wheel uses), else the absolute system CUDA directory. The extension is linked by distutils with an argument list, no shell, so `$ORIGIN` is literal.
    site = os.path.dirname(os.path.dirname(__import__("torch").__file__))
    if os.path.commonpath([cuda_libdir, site]) == site:
        rpath = "$ORIGIN/" + os.path.relpath(cuda_libdir, os.path.join(site, "torch_tbccl"))
    else:
        rpath = cuda_libdir
    link_args += [f"-Wl,-rpath,{rpath}", cudart_so]
    macros.append(("TORCH_TBCCL_WITH_CUDA", "1"))
else:
    print("torch-tbccl: CUDA disabled (needs TBCCL's tbccl_cuda component and a CUDA-enabled torch)")

sanitize = os.environ.get("TORCH_TBCCL_SANITIZE")  # e.g. "address,undefined" or "thread"
compile_args = ["-O1", "-g", "-fno-omit-frame-pointer", f"-fsanitize={sanitize}"] if sanitize else ["-O2"]
# Keep the build machine's directory layout out of the binary (assert/__FILE__ strings): sources, the torch headers and the TBCCL prefix get neutral prefixes.
_torch_dir = os.path.dirname(os.path.abspath(__import__("torch").__file__))
compile_args += [f"-ffile-prefix-map={HERE}=torch-tbccl", f"-ffile-prefix-map={_torch_dir}=torch", f"-ffile-prefix-map={root}=tbccl-prefix",
                 f"-ffile-prefix-map={__import__('sysconfig').get_paths()['include']}=python"]
if sanitize:
    link_args.append(f"-fsanitize={sanitize}")
    print(f"torch-tbccl: sanitizer build ({sanitize})")

sources = sorted(glob.glob(os.path.join("csrc", "*.cpp")))
if sys.platform == "darwin":
    link_args.append("-Wl,-S")  # no debug map: ld would record the absolute path of every object file in the build directory
    if torch.backends.mps.is_built():
        # PyTorch MPS tensors: the c10d MPS dispatch shim and the Objective-C++ tensor adapter (macOS only; Linux never compiles an .mm). The Metal framework is a system library.
        sources += sorted(glob.glob(os.path.join("csrc", "*.mm")))
        link_args += ["-framework", "Metal", "-framework", "Foundation"]
        macros.append(("TORCH_TBCCL_WITH_MPS", "1"))
        print("torch-tbccl: MPS enabled (Objective-C++ adapter + c10d MPS dispatch)")
    else:
        print("torch-tbccl: MPS disabled (this torch was not built with MPS)")

ext = CppExtension(
    name="torch_tbccl._C",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_objects=objects,
    define_macros=macros,
    extra_compile_args=compile_args + ["-Wall"],
    extra_link_args=link_args,
)

setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
