# Compatibility

| | Value |
|---|---|
| Package version | 0.2.0rc1 (release candidate; no final release has been published) |
| PyTorch | 2.13.x (`torch>=2.13,<2.14`), validated with 2.13.0; the extension is tied to the minor series it was built with |
| CPython | 3.13 only (`>=3.13,<3.14`) |
| TBCCL C ABI | 1 (`SUPPORTED_C_ABI`) |
| TBCCL wire protocol | 4 (`TESTED_WIRE_PROTOCOL`); a build against another wire version warns at import |
| Platforms | Linux x86-64 (CPU, CUDA 13) and macOS arm64 (CPU, MPS) |

The machine-readable form is `compatibility.json` at the repository root; `tools/check_compatibility_manifest.py` verifies it against `torch_tbccl/_version.py` and `pyproject.toml`. libtbccl owns the C ABI and wire protocol definitions ([TBCCL versioning](https://tbccl.tensorsofthewall.com/en/stable/reference/versioning.html)); this project records which of them it supports.

The package version, the linked TBCCL runtime version, the C ABI version and the wire protocol version are four different things. `torch_tbccl.info()` reports all of them.
