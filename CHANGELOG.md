# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. 0.2.0 is the first release; it is experimental.

## 0.3.1

Security-maintenance patch release. There is no change to the adapter source (`csrc/`, `torch_tbccl/`), the supported PyTorch 2.13.x line, the TBCCL C ABI (1) or the wire protocol (4); the version string is the only source difference. It is experimental.

This is **not** a fix for a vulnerability in torch-tbccl or in 0.3.0. In October 2026 the emSign root affected by the eMudhra intermediate-key compromise was removed from the CA bundles of Python tooling (certifi 2026.10.19, pip 26.2.2, uv 0.13.4). None of the published 0.3.0 wheels contains a CA bundle or TLS code (all of them were searched). What this release changes is the environment the wheels are built in and the TBCCL they bundle: they are rebuilt in a patched manylinux image, with pip 26.2.2 and uv 0.13.5 and with the revoked root blocked in the build container's system bundle, against TBCCL 0.7.1. The 0.3.0 wheels stay as published and are not unsafe to use.

### Installation

- Two CI-built wheels per platform set, named `torch_tbccl-0.3.1-...`, attached to the GitHub release with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. They are not published on PyPI or TestPyPI. The wheels statically link TBCCL 0.7.1.

### Security

- The documentation build pins certifi 2026.10.19; the release and documentation jobs install pip 26.2.2 from its hash-pinned wheel before their first pip use; the manylinux_2_28 image moves to a digest with pip 26.2.2, certifi 2026.10.19 and uv 0.13.4 (compilers unchanged) and the clean-install test image to Python 3.13.17; `tools/check_trust_dependencies.py` rejects affected pins and TLS-verification bypasses.
- Use pip 26.2.2 or later, uv 0.13.4 or later and certifi 2026.10.19 or later in the environments where you install the adapter; upgrading the `certifi` package does not patch an older pip. See `SECURITY.md`.

## 0.3.0

The first minor release after 0.2, promoted from release candidate 0.3.0rc1 with no code change. There is no change to the backend source (`csrc/`, `torch_tbccl/`), the `ProcessGroup` behaviour, PyTorch support (2.13.x, CPython 3.13), the TBCCL C ABI (1) or the wire protocol (4).

### Installation

- Two CI-built wheels, named `torch_tbccl-0.3.0-...`, attached to the GitHub release with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. They are not published on PyPI or TestPyPI.
- The wheels are built against TBCCL 0.7.0 and statically link it. A 0.2.x wheel keeps the TBCCL it was built with; to use the changes below, install a 0.3.x wheel.

### Changed

- **Two-rank `all_gather` is faster by default.** The bundled TBCCL 0.7.0 runs both directions of a two-rank AllGather at once. The adapter needs no setting. Measured on a direct Thunderbolt 4 link between a Linux/NVIDIA host and an Apple-silicon Mac, completed latency fell by roughly 25% to 66% in the sampled sizes.

### Added

- **Opt-in AllReduce pipeline for DDP.** `TBCCL_ALLREDUCE_PIPELINE_BYTES=131072` on both ranks, set before `init_process_group`, pipelines two-rank all-reduce in chunks. In a DistributedDataParallel step with 25 MB buckets over Thunderbolt 4 the step time fell by roughly a third to a half; with the setting unset the step time matched the published 0.2.1 wheel within about 1%. The setting is not negotiated, so set it on both ranks; see the DDP guide for when not to enable it.
- Documentation of the setting in the DDP guide and the configuration reference.

### Compatibility

- PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1, wire protocol 4, TBCCL 0.6.0 or newer as a build-time minimum. Platforms unchanged.

### Known limitations

- The measurements above come from one Linux CUDA / Apple Metal machine pair. PCIe AER reporting was unavailable on the Linux host, so no statement about AER errors is made.

## 0.2.1

Patch release. It adds the public PyTorch benchmark provider and its documentation. There is no change to the backend, the `ProcessGroup` behaviour, PyTorch support (2.13.x, CPython 3.13), the TBCCL C ABI (1) or the wire protocol (4), and nothing in this release is a performance improvement. The wheels are built against TBCCL 0.6.1 (a patch release of 0.6.0 without runtime change); the minimum TBCCL package stays 0.6.0.

### Installation

- The same two CI-built wheels as 0.2.0, named `torch_tbccl-0.2.1-...`, attached to the GitHub release with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation.
- The benchmark provider is **not** in the wheel. It is the `benchmarks/provider` directory of the source of this release (the source archive of the tag `v0.2.1`); the shared runner from the `tbccl` repository runs it with your PyTorch environment's interpreter.

### Added

- A benchmark provider (provider protocol 1) for the shared `tbccl` benchmark suite: point-to-point, collectives, asynchronous workloads and `DistributedDataParallel` steps over the `tbccl` backend, with `describe` and `validate` steps and a standalone entry point.
- Benchmark documentation: the workloads, how they are timed, the measured reference results with their hardware, software and limits, and a quickstart.

### Compatibility

- PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1, wire protocol 4, TBCCL 0.6.0 or newer. Platforms unchanged.

### Known limitations

- The reference benchmark results were measured with a development build of 0.2.0 on one Linux CUDA / Apple Metal machine pair and are labelled as such.

## 0.2.0

First release. Promoted from release candidate 0.2.0rc1 with no change to the backend; it is built against TBCCL 0.6.0 (C ABI 1, wire protocol 4). The MPS peer-failure test was made robust to the failure surfacing at submission or at wait (test only).

### Installation

- CI-built wheels (Linux x86-64 manylinux_2_28 for the CUDA-13 PyTorch; macOS arm64 14.0+) attached to the GitHub release (not published on PyPI or TestPyPI), with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The wheel links TBCCL 0.6.0 statically.

### Added

- A `torch.distributed` backend named `"tbccl"` over an installed TBCCL runtime: `send`/`recv`, `broadcast`, `all_gather`, `barrier` and `SUM` `all_reduce` on CPU, CUDA and (macOS) MPS tensors, at world sizes 1 to 4.
- Real `DistributedDataParallel` training at world sizes 2 to 4.
- Heterogeneous groups, such as CUDA on Linux with MPS on a Mac.
- MPS support registers the needed c10d kernels out of tree.

### Changed

- Point-to-point and collective calls may be in flight together on one process group (requires TBCCL wire protocol 4).

### Fixed

- None.

### Compatibility

- PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1 and wire protocol 4.
- Linux x86-64 (CPU, CUDA 13) and macOS arm64 (CPU, MPS).
- The compiled extension is tied to the PyTorch minor series it was built with.

### Known limitations

- Operations outside the documented list raise a clear error; it does not aim for NCCL feature parity.
- Float16 and BFloat16 `all_reduce` only at world size 2; `gather` at world size 2 only.
- No recovery after a failure: the communicator is aborted.
- Inherits TBCCL's security model: no authentication or encryption.

## 0.2.0rc1 (release candidate)

First release candidate of the first planned release, 0.2.0. Expect an rc2 if a blocker is found.

### Installation

- CI-built wheels (Linux x86-64 manylinux_2_28 for the CUDA-13 PyTorch; macOS arm64 14.0+) attached to the GitHub pre-release (not published on PyPI or TestPyPI), with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The wheel links TBCCL 0.6.0rc1 statically; no TBCCL install is needed. Requires CPython 3.13 and `torch>=2.13,<2.14`.

### Added

- A `torch.distributed` backend named `"tbccl"` over an installed TBCCL runtime: `send`/`recv`, `broadcast`, `all_gather`, `barrier` and `SUM` `all_reduce` on CPU, CUDA and (macOS) MPS tensors, at world sizes 1 to 4.
- Real `DistributedDataParallel` training at world sizes 2 to 4.
- Heterogeneous groups, such as CUDA on Linux with MPS on a Mac.
- MPS support registers the needed c10d kernels out of tree.

### Changed

- Point-to-point and collective calls may be in flight together on one process group (requires TBCCL wire protocol 4).

### Fixed

- None.

### Compatibility

- PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1 and wire protocol 4.
- Linux x86-64 (CPU, CUDA 13) and macOS arm64 (CPU, MPS).
- The compiled extension is tied to the PyTorch minor series it was built with.

### Known limitations

- Operations outside the documented list raise a clear error; it does not aim for NCCL feature parity.
- Float16 and BFloat16 `all_reduce` only at world size 2; `gather` at world size 2 only.
- No recovery after a failure: the communicator is aborted.
- Inherits TBCCL's security model: no authentication or encryption.
