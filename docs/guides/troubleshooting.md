# Troubleshooting

Find the symptom, check the likely cause with the diagnostic, apply the fix. Start with `python -m torch_tbccl.info` on every machine ([how do I know TBCCL is being used?](verify-active.md)).

## Installing and importing

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `ImportError` mentioning `libtorch_cuda.so` or `libc10_cuda.so` on Linux | a CPU-only PyTorch; the Linux wheel needs the CUDA 13 build | `python -c "import torch; print(torch.__version__)"` (expect `2.13.x+cu130`) | install `torch==2.13.0` from `https://download.pytorch.org/whl/cu130` ([installing](../getting-started/install.md)) |
| `ImportError` mentioning a torch series on `import torch_tbccl` | the wheel was built against a different torch minor series | `python -m torch_tbccl.info` (`torch_running` vs `torch_built_with`) | install PyTorch 2.13.x, or rebuild against the installed torch |
| no matching distribution for the wheel | Python is not 3.13, or the platform is not Linux x86-64 / macOS arm64 | `python --version` | use Python 3.13 on a supported platform |
| `ImportError` mentioning the C ABI | built from source against a TBCCL prefix with an unsupported C ABI | `tbccl-info` of the prefix | rebuild against a compatible prefix |
| a warning that the wire protocol is untested | built against a TBCCL wire protocol this release was not tested with | `python -m torch_tbccl.info` | use the released wheel; ranks with different wire versions cannot connect |
| installing the wheel changed my torch | the wheel was installed without `--no-deps` | `pip list \| grep torch` | reinstall the intended torch and use `--no-deps`; never use `--reinstall` |
| `UserWarning: Failed to initialize NumPy: No module named 'numpy'` at import | the environment has no NumPy; PyTorch warns once | `python -c "import numpy"` | harmless for torch-tbccl (the validated installs have no NumPy and run `all_reduce` on CPU, CUDA and MPS); `uv pip install --python .venv/bin/python numpy` silences it |

## Process-group creation

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `ValueError: torch-tbccl: invalid argument: TBCCL_LOCAL_ENDPOINT is not set` | the endpoint variable is missing on a rank of a group with two or more ranks | `echo $TBCCL_LOCAL_ENDPOINT` on every host | set `TBCCL_LOCAL_ENDPOINT=<host>:0` per rank, with a host that the peers can reach |
| `protocol_mismatch ... wire protocol` at `init_process_group` | ranks run torch-tbccl wheels built against different TBCCL wire protocols | `python -m torch_tbccl.info` on each host | install the same torch-tbccl version everywhere |
| `timeout` at `init_process_group` | a rank never started, never published its endpoint, or `TBCCL_LOCAL_ENDPOINT` is not reachable from the peers | `ping` the peer address; check that every rank started | start every rank; use an address reachable from the peers (not `127.0.0.1` across hosts) |
| the second `init_process_group` after a failed first one fails | retrying over the same store after a failed bootstrap is not supported | | create a fresh store (restart `torchrun`, or use a new `TCPStore`/`HashStore`) |
| hostname does not resolve under `torchrun` on one machine | the default local address is the hostname | | add `--local-addr 127.0.0.1` |
| a Mac rank hangs at start while Linux is fine | macOS firewall or Local Network prompt not yet answered, or a background job without a live session | look for a dialog on the Mac's screen | answer the prompt once from a local session; run the rank in a live terminal or SSH session |

## Operations

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `NotImplementedError: torch-tbccl: unsupported operation ...` | an operation outside the [supported surface](../reference/supported-operations.md) (for example `reduce`, `MAX`) | the message names it | use a supported operation or another backend for it |
| a one-rank group rejects a collective | collectives need at least two ranks | | do not use DDP at world size 1 |
| `ValueError: ... tensor must be contiguous (no implicit copy is made)` | the adapter never makes a hidden copy | `tensor.is_contiguous()` | pass `tensor.contiguous()` |
| float16/bfloat16 `all_reduce` rejected | those reductions work at world size 2 only | | use float32 or run two ranks |
| MPS: a Metal assertion or a crash with several threads | `torch.mps.synchronize()` is not safe from two threads | | submit operations on MPS tensors from one thread; use CPU tensors for side traffic ([MPS](mps.md)) |

## Failure and lifecycle

Expectations are explicit: **there is no recovery.** After a failure the process group is not usable and must be destroyed and recreated.

- **Peer loss.** If a peer exits or dies, the surviving ranks' operations fail within milliseconds with a `torch-tbccl:` error.
- **Abort.** PyTorch's `torch.distributed.distributed_c10d._abort_process_group()` (a private PyTorch function) aborts the communicator through the backend; pending work fails. Use it to unblock a process whose peer is gone.
- **Destroy with pending work.** `dist.destroy_process_group()` with un-waited work aborts the group instead of hanging; the work fails with an error. Wait on your `Work` handles before destroying.
- **Waiting.** `Work.wait(timeout)` raises on expiry but does not cancel the operation.
- **Interpreter shutdown** with a live group does not hang (the completion thread never destroys tensors; see [work lifetime](../concepts/work-lifetime.md)).
- **Mixed-version ranks** cannot connect: `protocol_mismatch ... wire protocol` is raised at bootstrap, not later.

Peer-failure behavior is validated on loopback; do not kill a physical peer to test it.
