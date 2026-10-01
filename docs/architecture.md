# Architecture

```
PyTorch (torch.distributed, c10d)
   |
   v
torch-tbccl            adapter only
   |
   v
installed libtbccl     transport / collectives / device providers
```

Dependencies flow one way. libtbccl never depends on PyTorch.

torch-tbccl translates:

| PyTorch                     | TBCCL                          |
|-----------------------------|--------------------------------|
| `c10d::Store`               | `tbccl::CommunicatorOptions`   |
| rank / world_size           | `tbccl::Communicator`          |
| `at::Tensor`                | `tbccl::BufferView`            |
| torch dtype                 | `tbccl::DataType`              |
| c10d `ReduceOp`             | `tbccl::ReduceOp`              |
| current CUDA stream         | `tbccl::ExecutionContext`      |
| `tbccl::Work`               | `c10d::Work`                   |

No collective algorithm, TCP code, chunk scheduling, CUDA staging or local
reduction lives here. If one seems necessary, identify the missing libtbccl
API instead.

## Build rule
The extension links only against an **installed** TBCCL (`TBCCL_ROOT`):
`include/tbccl/*.hpp` plus the installed library. No TBCCL source-tree
include paths, no `add_subdirectory`, no TBCCL `.cpp` files compiled in.

The installed static library must be built position-independent
(`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`), since it is linked into a
shared Python extension.

## Deferred
- TBCCL's C ABI (not needed by a C++ extension; useful for Rust/Swift/MLX later).
- N>2, other collectives, DDP, MPS.
