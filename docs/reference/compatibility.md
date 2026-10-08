# Compatibility

| | Value |
|---|---|
| Package version | 0.2.0rc1 (release candidate; no final release has been published) |
| PyTorch | 2.13.x (`torch>=2.13,<2.14`), validated with 2.13.0; the extension is tied to the minor series it was built with |
| CPython | 3.13 only (`>=3.13,<3.14`) |
| TBCCL C ABI | 1 (`SUPPORTED_C_ABI`) |
| TBCCL wire protocol | 4 (`TESTED_WIRE_PROTOCOL`); a build against another wire version warns at import |
| Platforms | Linux x86-64 (CPU, CUDA 13), macOS arm64 (CPU, MPS) |

## Wheels

| Wheel | Platform | PyTorch it needs |
|---|---|---|
| `torch_tbccl-0.2.0rc1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | Linux x86-64 | the CUDA 13 build of PyTorch 2.13 (`+cu130`) |
| `torch_tbccl-0.2.0rc1-cp313-cp313-macosx_14_0_arm64.whl` | macOS 14 or newer, Apple silicon | PyTorch 2.13 from PyPI (MPS-enabled) |

Both contain libtbccl 0.6.0rc1 statically linked. See [installing](../getting-started/install.md).

## Relationship to the other projects

torch-tbccl depends only on PyTorch (at run time) and on libtbccl (bundled at build time). It does not depend on vllm-tbccl or exo-tbccl. Core transport compatibility (platforms, CUDA, world sizes, the C ABI and wire protocol) is defined by the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html); this page records what torch-tbccl adds. The machine-readable form is `compatibility.json` at the repository root; `tools/check_compatibility_manifest.py` verifies it against `torch_tbccl/_version.py` and `pyproject.toml`.

The package version, the linked TBCCL runtime version, the C ABI version and the wire protocol version are four different things. `torch_tbccl.info()` reports all of them. Ranks with different wire protocols cannot connect ([TBCCL versioning](https://tbccl.tensorsofthewall.com/en/latest/reference/versioning.html)).
