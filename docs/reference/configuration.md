# Environment variables

Read from the environment of each process. Only `TBCCL_LOCAL_ENDPOINT` is needed in normal use; the others are diagnostics or escape hatches.

## Run time

| Variable | Default | Accepted values | Purpose | When to use | Scope |
|---|---|---|---|---|---|
| `TBCCL_LOCAL_ENDPOINT` | unset | `<host>:<port>`; `host` an IPv4 address or resolvable name (no IPv6), `port` 0 to 65535 | where this rank listens for TBCCL connections. The control port is `port`, the data port is `port + 1000`; `<host>:0` lets the kernel choose free ports | every rank of a group with two or more ranks; use an address the peers can reach | per process, read when a process group is created |
| `TORCH_TBCCL_TRACE` | off | `1` (any value other than empty or `0` enables) | records one in-memory timeline entry per collective; read with `torch_tbccl.trace_events()` | to confirm TBCCL is used or to time operations | per process, read at import |
| `TORCH_TBCCL_FORCE_SYNC_ALLREDUCE` | off | `1` | makes `all_reduce` wait for completion before returning | diagnostic baseline only, never a production mode | per process group, read at creation |
| `TORCH_TBCCL_ALLOW_TORCH_MISMATCH` | unset | `1` | skips the torch-series check at import | only if you accept an unvalidated, possibly crashing combination | per process, read at import |

`OMP_NUM_THREADS` (a PyTorch/OpenMP setting) should be set, for example to 2, when several ranks share one machine, or they starve each other.

## Build time (building from source only)

| Variable | Purpose |
|---|---|
| `TBCCL_ROOT` | the installed TBCCL prefix; required on every build |
| `TORCH_TBCCL_SANITIZE` | builds the extension with sanitizer flags, for example `address,undefined` or `thread` (development) |

Test-only variables (for example `TORCH_TBCCL_TEST_CLEAN_INSTALL` and `TORCH_TBCCL_TEST_SCRATCH`) are described in [building and testing](../development/testing.md) and are not user configuration.
