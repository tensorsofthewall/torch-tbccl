# Development

This repository, **torch-tbccl**, is the PyTorch adapter. It maps `torch.distributed` onto an installed libtbccl and nothing more.

| Belongs here | Does not belong here |
|---|---|
| the c10d `ProcessGroup` / `Work` / `Future` mapping | communication algorithms, TCP code, chunk scheduling, CUDA staging, local reduction (these are TBCCL) |
| `c10d::Store` to TBCCL endpoint exchange | anything that requires another adapter (vllm-tbccl, exo-tbccl) |
| torch dtype, `ReduceOp` and CUDA-stream mapping | patches to PyTorch |
| the out-of-tree MPS dispatch registration | exposing TBCCL classes as a second Python API |

Start with [contributing](contributing.md) and [where should this change go?](where-should-this-change-go.md). The rules for the code base are in `AGENTS.md`.

```{toctree}
:maxdepth: 1

contributing
where-should-this-change-go
testing
building-docs
release-process
documentation-versions
```
