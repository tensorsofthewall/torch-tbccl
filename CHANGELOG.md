# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. torch-tbccl has had no final release. 0.2.0rc1 is a release candidate (pre-release), not production-ready.

## 0.2.0rc1 (release candidate)

First release candidate of the first planned release, 0.2.0. Expect an rc2 if a blocker is found.

### Installation

- CI-built wheels (Linux x86-64 manylinux_2_28 for the CUDA-13 PyTorch; macOS arm64 14.0+) attached to the GitHub pre-release and uploaded to TestPyPI, with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The wheel links TBCCL 0.6.0rc1 statically; no TBCCL install is needed. Requires CPython 3.13 and `torch>=2.13,<2.14`.

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
