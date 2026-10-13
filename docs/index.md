# torch-tbccl

A PyTorch `torch.distributed` backend named `"tbccl"` for communicating between heterogeneous machines (for example a Linux host with an NVIDIA GPU and a Mac) over TCP, including a direct Thunderbolt 4 link. PyTorch is not modified: `import torch_tbccl` and use the normal distributed API.

```{note}
**Release candidate.** The current release candidate is 0.2.0rc1 (PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1 / wire protocol 4). It is a pre-release for validation, not for production use. The final 0.2.0 has not been released.
```

torch-tbccl is one of three independent adapters over the [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/) runtime; the others are vllm-tbccl (vLLM) and exo-tbccl (exo). It does not depend on them and they do not depend on it. See the [ecosystem overview](project/ecosystem.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

reference/supported-operations
guides/ddp
guides/mps
reference/configuration
```

```{toctree}
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/verify-active
```

```{toctree}
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:maxdepth: 2
:caption: Project

project/index
validation/index
```
