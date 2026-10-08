# How do I know TBCCL is being used?

A silent fallback to another backend would invalidate a benchmark or a debugging session. These checks confirm that your process group runs on TBCCL.

## 1. The backend name

```python
import torch.distributed as dist
dist.init_process_group("tbccl")
print(dist.get_backend())          # tbccl
```

If you pass another backend (`"gloo"`, `"nccl"`) or a device-to-backend map, `get_backend()` reports it. Requesting `"tbccl"` explicitly is the only way to get TBCCL; torch-tbccl never replaces another backend.

## 2. The installation

```sh
python -m torch_tbccl.info          # registered True, backend tbccl, the TBCCL versions
```

or, in Python, `torch_tbccl.info()` returns the same fields as a dict (`registered`, `libtbccl`, `c_abi`, `wire_protocol`, `devices`, ...). `registered False` or an `ImportError` on `import torch_tbccl` means the backend is not available in that process.

## 3. Evidence that operations went through TBCCL

Set `TORCH_TBCCL_TRACE=1` and read the in-memory timeline after some collectives:

```python
import torch_tbccl
events = torch_tbccl.trace_events()
print(events[0])
# {'seq': 0, 'op': 'allreduce', 'bytes': 4, 'device': 'cpu', ..., 'complete_ns': ..., 'error': False}
```

One record per collective: sequence number, operation, bytes, device and the entry, submit, TBCCL-return, completion and wait timestamps of this process. Records appearing for your operations prove they ran through the TBCCL process group. Timestamps are from this process's monotonic clock; never compare them across machines. `torch_tbccl.trace_events()` returns an empty list when tracing is off.

## 4. Errors name the backend

Failures raised by the backend start with `torch-tbccl:` (for example `torch-tbccl: unsupported operation: reduce is not implemented by the tbccl backend`). An operation that the backend does not support is rejected with that message; it does not fall back to another backend.

## For benchmarking

Record the output of `python -m torch_tbccl.info` on every host together with your results, and keep `TORCH_TBCCL_FORCE_SYNC_ALLREDUCE` unset (it is a diagnostic baseline, never a production mode).
