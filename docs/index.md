# torch-tbccl

:::{container} tb-lead
torch-tbccl is a PyTorch `torch.distributed` backend named `"tbccl"`. It lets one PyTorch program communicate between machines that differ in operating system and accelerator, for example a Linux host with an NVIDIA GPU and a Mac with Apple silicon, over TCP including a direct Thunderbolt 4 link. PyTorch is not modified: `import torch_tbccl` and use the normal distributed API.
:::

:::{container} tb-actions
[Get started](getting-started/quickstart.md) [What is supported](reference/supported-operations.md)
:::

```{note}
**Release.** The current release is 0.3.0 (PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1 / wire protocol 4), whose wheels bundle TBCCL 0.7.0. It is experimental.
```

<nav class="tb-eco" aria-label="TBCCL ecosystem: the four documentation sites">
<ul>
<li class="tb-tbccl"><a href="https://tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">tbccl</span><span class="tb-eco-role">native runtime</span></a></li>
<li class="tb-torch" aria-current="page"><span><span class="tb-eco-name">torch-tbccl</span><span class="tb-eco-role">PyTorch backend</span><span class="tb-eco-here">This site</span></span></li>
<li class="tb-vllm"><a href="https://vllm-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">vllm-tbccl</span><span class="tb-eco-role">vLLM integration</span></a></li>
<li class="tb-exo"><a href="https://exo-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">exo-tbccl</span><span class="tb-eco-role">exo integration</span></a></li>
</ul>
</nav>

## What you get

::::{container} tb-cards
:::{container} tb-card tb-torch

<h3>The normal distributed API</h3>

`dist.init_process_group("tbccl")` with `torchrun`, process groups and `DistributedDataParallel`. Your program keeps using `torch.distributed`.

[Overview](getting-started/overview.md)
:::

:::{container} tb-card tb-torch

<h3>Collectives and devices</h3>

Point-to-point, `broadcast`, `all_gather`, `all_reduce` (`SUM` only) and `barrier` on CPU, CUDA (Linux) and MPS (macOS) tensors.

[Supported operations](reference/supported-operations.md)
:::

:::{container} tb-card tb-torch

<h3>DDP across unlike machines</h3>

Train with `DistributedDataParallel` on CPU, CUDA and, next to a CPU or CUDA rank, MPS ranks.

[DDP guide](guides/ddp.md) · [MPS guide](guides/mps.md)
:::

:::{container} tb-card tb-torch

<h3>Built on TBCCL</h3>

The wheel contains a statically linked libtbccl, so you do not install TBCCL separately.

[TBCCL documentation](https://tbccl.tensorsofthewall.com/en/latest/) · [GitHub](https://github.com/tensorsofthewall/torch-tbccl)
:::
::::

## Get started

::::{container} tb-cards
:::{container} tb-card 

<h3>Install</h3>

A platform wheel; install the matching PyTorch first.

[Installing torch-tbccl](getting-started/install.md)
:::

:::{container} tb-card 

<h3>Run a two-host all-reduce</h3>

The shortest path to a working group across two machines.

[Quickstart](getting-started/quickstart.md)
:::

:::{container} tb-card 

<h3>Train with DDP</h3>

Gradient reduction across CUDA, CPU and MPS ranks.

[DDP guide](guides/ddp.md)
:::

:::{container} tb-card 

<h3>Use a Mac GPU</h3>

What works with MPS tensors and what does not.

[MPS guide](guides/mps.md)
:::

:::{container} tb-card 

<h3>Check it is active</h3>

Confirm the job really runs on the `tbccl` backend.

[Verify the backend](guides/verify-active.md)
:::
::::

## What is supported

Only configurations that were actually exercised are listed as validated. Everything else is not supported or not claimed.

::::{container} tb-cols
:::{container}
**Validated**

- PyTorch 2.13.x and CPython 3.13 on Linux x86-64 (CPU, and CUDA with the CUDA 13 build of PyTorch) and macOS arm64 (CPU and MPS).
- A Linux CUDA host and a Mac with MPS over a direct Thunderbolt 4 link: point-to-point, `all_reduce`, `broadcast`, `all_gather` and `barrier`, in both rank orientations.
- World sizes 1 to 4, and DDP, on loopback and (on the development tree) between a CUDA and an MPS rank.
:::
:::{container}
**Not supported or not claimed**

- `SUM` is the only reduction; there is no `reduce`, `scatter`, `reduce_scatter` or `all_to_all`, and `gather` works for two-rank groups only.
- No recovery after an abort or a peer failure; recreate the process group.
- FSDP and world sizes 5 to 8 are not validated; two Linux hosts over a physical network are not part of the recorded validation.
- Other PyTorch and CPython versions are not claimed.
:::
::::

Authoritative pages: [compatibility](reference/compatibility.md), [validated configurations](project/validated-configurations.md) and [known limitations](project/limitations.md).

## Where it fits

torch-tbccl is one of three independent adapters over the [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/) runtime; the others are vllm-tbccl (vLLM) and exo-tbccl (exo). It does not depend on them and they do not depend on it. See the [ecosystem overview](project/ecosystem.md).

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/quickstart
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

reference/supported-operations
guides/ddp
guides/mps
reference/configuration
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/verify-active
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Project

project/index
validation/index
```
