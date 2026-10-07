# License and third-party software

torch-tbccl is licensed under the Apache License, Version 2.0. The full text is in the `LICENSE` file at the repository root. Copyright 2026 Sandesh Bharadwaj.

Contributions are accepted under the same license (Apache-2.0, section 5).

## Static linking of libtbccl

The compiled extension `torch_tbccl._C` links the installed TBCCL static libraries (`libtbccl.a`, and `libtbccl_cuda.a` for CUDA builds) into itself. A torch-tbccl wheel therefore redistributes TBCCL code. TBCCL and torch-tbccl are both Apache-2.0 and have the same copyright holder, so the wheel's single `LICENSE` file covers both; no additional notice is required for the embedded TBCCL code. The build must use a TBCCL prefix whose license is Apache-2.0.

The wheel does not bundle PyTorch, the CUDA runtime or any other shared library: PyTorch is a dependency installed separately, and a CUDA build resolves `libcudart` at run time from the environment (the PyTorch wheel's NVIDIA packages, or the system CUDA directory).

## Source code

The extension includes PyTorch headers at build time. It contains no copied PyTorch source. The MPS dispatch shim re-implements the behavior of PyTorch's CPU and CUDA c10d kernels for the MPS dispatch key using PyTorch's public registration API, and checks the operator schemas at registration time.

## Dependencies

| Dependency | Role | License |
|---|---|---|
| PyTorch | runtime dependency, build-time headers | BSD-style (see PyTorch's `LICENSE`) |
| TBCCL | statically linked | Apache-2.0 |
| NumPy, pytest | tests only | not redistributed |

The documentation toolchain is used only to build the documentation and is not redistributed.

## Notices

No third-party `NOTICE` file or license bundle is required for the wheel as built: it contains no third-party code. PyTorch itself ships its own license and notice files with its own distribution. A `NOTICE` file is not provided.
