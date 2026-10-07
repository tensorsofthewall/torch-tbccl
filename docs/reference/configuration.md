# Configuration

## Run time

| Variable | Effect |
|---|---|
| `TBCCL_LOCAL_ENDPOINT=<host>:<port>` | where this rank listens for TBCCL connections; required on every rank; `<host>:0` picks free ports; the data port is `port + 1000` |
| `TORCH_TBCCL_TRACE=1` | records per-collective timestamps in memory; read with `torch_tbccl.trace_events()` |
| `TORCH_TBCCL_FORCE_SYNC_ALLREDUCE=1` | makes all_reduce wait for completion before returning; diagnostic baseline only, never a production mode |
| `TORCH_TBCCL_ALLOW_TORCH_MISMATCH=1` | skips the torch-series check at import, at your own risk |

## Build time

| Variable | Effect |
|---|---|
| `TBCCL_ROOT` | the installed TBCCL prefix; required on every build |
| `TORCH_TBCCL_SANITIZE` | builds the extension with sanitizer flags (development) |

`OMP_NUM_THREADS` should be set (for example to 2) when several ranks share one machine, or they starve each other.
