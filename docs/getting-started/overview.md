# Overview

torch-tbccl registers a `torch.distributed` backend named `"tbccl"`. After `import torch_tbccl`, `dist.init_process_group("tbccl")` creates process groups whose communication runs over [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/), a transport that works between machines with different operating systems and accelerators.

## Why use it

NCCL needs NVIDIA GPUs on every rank and Gloo moves CUDA tensors through the CPU one way or another. If your group mixes a Linux machine with a CUDA GPU and a Mac with Apple silicon (MPS), neither fits. torch-tbccl lets the same PyTorch program use CPU, CUDA and MPS tensors across such a group, for example to train with DDP across a CUDA machine and a Mac over a Thunderbolt cable.

## How it works

```text
PyTorch (torch.distributed, c10d)
   |   unmodified
torch-tbccl            a c10d backend: maps tensors, dtypes, ops and streams
   |
libtbccl (bundled)     transport, collectives, CUDA staging
   |
TCP / Thunderbolt 4
```

- PyTorch's own rendezvous (`torchrun`, `MASTER_ADDR`, a store) decides who the ranks are. torch-tbccl uses that store only to exchange TBCCL endpoints; the data then flows over TBCCL's own connections to the address in `TBCCL_LOCAL_ENDPOINT`.
- CUDA tensors are staged through pinned host memory by TBCCL; MPS tensors are read and written in place through their shared Metal buffers. There is no direct CUDA-to-MPS DMA.
- The wheel contains a statically linked libtbccl. You do not install TBCCL separately.

## What it is not

It is not NCCL: only a defined set of operations is supported, `SUM` is the only reduction, and there is no recovery after a failure ([supported operations](../reference/supported-operations.md), [known limitations](../project/limitations.md)). It does not patch PyTorch and it exposes no Python API of its own beyond version and diagnostic helpers.

## Next steps

1. [Install](install.md) the wheel that matches your platform.
2. Run the [quickstart](quickstart.md).
3. Check [how to know TBCCL is being used](../guides/verify-active.md).
