# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. torch-tbccl has not been released: everything below is unreleased.

## Unreleased

Planned for 0.2.0. This section describes the first planned release and changes until it is published.

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
