# PyTorch c10d API audit (Phase 42)

Audited against the headers actually installed in the dev environment:
**torch 2.14.1+cu130**, CPython 3.13, `_GLIBCXX_USE_CXX11_ABI=1`.
Headers: `torch/include/torch/csrc/distributed/c10d/`.

## Backend (`Backend.hpp`)
- Base class is `c10d::Backend`; constructor `explicit Backend(int rank, int size)`.
- `allreduce(std::vector<at::Tensor>&, const AllreduceOptions& = {})` returns
  `c10::intrusive_ptr<Work>`; the default implementation throws
  "does not support allreduce" (so unimplemented collectives fail loudly
  without us overriding them).
- Other virtuals worth overriding: `getBackendName()`, `setTimeout()`,
  `shutdown()`, `abort()`, `waitForPendingWorks()`.
- `AllreduceOptions`: `reduceOp` (default SUM), `timeout` (`kUnsetTimeout`),
  `asyncOp`, optional `sparseIndices`.

## Work (`Work.hpp`)
- `Work(int rank = -1, OpType = UNKNOWN, const char* profilingTitle = nullptr,
  const optional<vector<Tensor>>& inputTensors = nullopt)`; derives from
  `torch::CustomClassHolder` (intrusive_ptr ownership).
- Virtuals: `isCompleted()`, `isSuccess() const`, `exception() const`,
  `wait(std::chrono::milliseconds timeout = kNoTimeout)` (`kNoTimeout` = 0 ms;
  throws if the work failed; returns false only if aborted),
  `synchronize()`, `blockCurrentStream()`, `abort()`, `getFuture()`,
  `getFutureResult()`, `result()`, `getDuration()`.
- `getFuture()` is documented as NCCL-only; the plain manual `all_reduce`
  path does not need it. Decision deferred to the async-Work commit.

## Store (`Store.hpp`)
- `set(key, vector<uint8_t>)`, `get(key)`, `check(keys)`,
  `wait(keys)` / `wait(keys, timeout)` (bounded wait available),
  `getTimeout()`, `add`, `deleteKey`, `multiGet`/`multiSet`.

## Python registration (`distributed_c10d.py`)
```python
Backend.register_backend(name, func, extended_api=False, devices=None, *, _backend_type=None)
```
- With `extended_api=False`, `func(store, rank, world_size, timeout)` is called.
- `devices` is a list such as `["cpu", "cuda"]`; if omitted, PyTorch warns
  and assumes cpu + current accelerator, so we always pass it explicitly.
- Registering also makes `default_device_backend_map[device]` point at the
  backend for devices that have no backend yet.

## Still to audit before the corresponding commits
- `ProcessGroupGloo.hpp` / `ProcessGroupNCCL.hpp` as reference for the
  Work subclass pattern and stream handling.
- Current-CUDA-stream API (`c10/cuda/CUDAStream.h`) when the CUDA commit lands.
