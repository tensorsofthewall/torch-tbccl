# AGENTS.md

Technical guidance for contributors and coding agents working in this repository. User documentation is in `README.md`; design notes are in `docs/architecture.md` and `docs/pytorch_api_audit.md`. Contribution workflow is in `CONTRIBUTING.md`.

## Purpose

`torch-tbccl` is an out-of-tree PyTorch `torch.distributed` backend (`"tbccl"`) that adapts c10d onto an installed TBCCL runtime. It is an adapter and nothing more.

```
PyTorch (c10d)  ->  torch-tbccl  ->  installed libtbccl
```

Dependencies flow one way: libtbccl never depends on PyTorch.

## Layout

| Path | Role |
|---|---|
| `csrc/` | Native adapter: bindings, `ProcessGroupTBCCL`, `WorkTBCCL`, completion worker, tensor adapter, bootstrap, error mapping; `.mm` files for macOS MPS |
| `torch_tbccl/` | Python package: registration, version checks, diagnostics |
| `tests/` | pytest suite; multi-rank tests spawn worker scripts (`tests/_worker_*.py`) |
| `tools/`, `examples/` | Capability matrix, packaging checks, physical-link harnesses, DDP examples |

## Architecture boundaries

- No communication algorithm, TCP code, chunk scheduling, device staging or reduction logic lives here. If the adapter seems to need one, identify the missing libtbccl API instead.
- The adapter only maps: `c10d::Store` to TBCCL bootstrap options, rank and world size to a `Communicator`, `at::Tensor` to `BufferView`, torch dtypes and reduce ops to TBCCL enums, the current CUDA stream to an execution context, and `tbccl::Work` to `c10d::Work`.
- Build only against an installed TBCCL prefix (`TBCCL_ROOT`). Never include TBCCL sources or compile TBCCL files into the extension. Never patch TBCCL behavior from here; fix generic defects in the TBCCL repository.
- Do not patch PyTorch and do not expose TBCCL classes to Python. Collectives go through `torch.distributed`.

## Critical invariants

- `BufferView` is non-owning. `WorkTBCCL` keeps strong references to every tensor until the TBCCL operation completes. Never add an implicit `.contiguous()` or any hidden copy.
- Never destroy tensors (a `WorkState`) on the completion thread. A tensor's last reference can hold the GIL when it is dropped, and interpreter shutdown joins that thread while holding the GIL. The worker retires finished states for callers to destroy.
- libtbccl never sees the c10d store. `csrc/bootstrap.cpp` publishes each rank's actual endpoint and one communicator id through the store and hands libtbccl a rank directory.
- No mutable global rank, world-size, endpoint or communicator state.
- Collectives and point-to-point operations may overlap on one group. libtbccl wire protocol 4 gives each peer pair separate collective and P2P connections, so their relative order is free. Build only against a prefix with wire protocol >= 4 (`TESTED_WIRE_PROTOCOL` in `torch_tbccl/_version.py`).
- Unsupported operations raise a clear error. Do not add kernels for operators the backend does not support.
- MPS (macOS): PyTorch 2.13 registers no c10d kernels for the MPS key, so `csrc/c10d_mps_dispatch.cpp` registers the supported ones out of tree (version- and schema-gated). An MPS tensor's `data_ptr()` is a Metal handle, never a CPU address; `csrc/mps_tensor_adapter.mm` converts it after a device-wide MPS synchronize. TBCCL operations on MPS tensors must come from one thread at a time (PyTorch's own `torch.mps.synchronize()` is not safe to call concurrently).
- Error messages are prefixed `torch-tbccl:` and categorized; preserve TBCCL's original message. TBCCL tagged errors are mapped in `csrc/errors.hpp`.
- The extension is tied to the torch minor series it was built with.

## Build and test

```sh
uv venv --python 3.13 .venv
TBCCL_ROOT=<installed tbccl prefix> uv pip install --python .venv/bin/python --no-build-isolation -e .
.venv/bin/python -m pytest -q
```

- Set `TBCCL_ROOT` on every build. A failed rebuild leaves the old extension in place, so read the build output, not only the test result.
- Rebuild with `--no-build-isolation --no-deps`; do not use `--reinstall`, which rewrites torch.
- Several ranks on one machine need `OMP_NUM_THREADS` set (for example 2).
- Multi-rank behavior is tested on loopback. Physical Thunderbolt and GPU runs are opt-in harnesses under `tools/` and `examples/`, kept out of ordinary pytest.
- Validate packaging with a built wheel in a clean environment, not with an editable install.

## Code ownership expectations

Changes to `csrc/process_group_tbccl.*`, `csrc/bootstrap.cpp`, the MPS adapter and the supported-surface contract (`tools/capability_matrix.py`, `tests/test_capability_matrix.py`) need maintainer review.

## Conventions

- Match the surrounding code. Comment only non-obvious constraints.
- Validate at real boundaries (user tensors, environment variables, store contents, TBCCL errors).
- Add explicit paths when staging files.
