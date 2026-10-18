# Code map

Where to start reading, by topic. It is a map to the directories and the first files, not a walk through the source; the rules for the code base are in `AGENTS.md` and the design is in the [architecture](../concepts/architecture.md) page. Every path below is checked by a test, so a rename that leaves this page stale fails the documentation checks.

```{table}
:class: tb-stack tb-stack-map

| Topic | Start at | Read with |
|---|---|---|
| Backend registration (what `import torch_tbccl` does) | `torch_tbccl/__init__.py`, `csrc/bindings.cpp` | [Verify the backend](../guides/verify-active.md) |
| Version and compatibility checks at import | `torch_tbccl/_version.py`, `compatibility.json` | [Compatibility](../reference/compatibility.md) |
| The native extension: the process group | `csrc/process_group_tbccl.cpp`, `csrc/process_group_tbccl.hpp` | [Architecture](../concepts/architecture.md) |
| Tensor dispatch (tensor to buffer, dtype, stream) | `csrc/tensor_adapter.cpp`, `csrc/mps_tensor_adapter.mm`, `csrc/c10d_mps_dispatch.cpp` | [MPS guide](../guides/mps.md) |
| Distributed collectives, `Work` and completion | `csrc/work_tbccl.cpp`, `csrc/completion_worker.cpp` | [Work lifetime](../concepts/work-lifetime.md) |
| Rendezvous (store to TBCCL endpoints) | `csrc/bootstrap.cpp` | [Rendezvous](../concepts/rendezvous.md) |
| DDP | `examples/ddp_mlp.py`, `tests/test_ddp.py` | [DDP guide](../guides/ddp.md) |
| The supported surface as a tested contract | `tests/test_capability_matrix.py`, `tools/capability_matrix.py` | [Supported operations](../reference/supported-operations.md) |
| Tracing and diagnostics | `csrc/trace.cpp`, `torch_tbccl/info.py`, `tools/collect_diagnostics.py` | [Verify the backend](../guides/verify-active.md) |
| Packaging and release checks | `tools/package_inspect.py`, `tools/build_release_wheel.sh`, `setup.py` | [Testing](testing.md) |
```

The runtime itself (transport, algorithms, protocol, the C ABI) is not in this repository; its [code map](https://tbccl.tensorsofthewall.com/en/latest/development/code-map.html) lives in TBCCL. An adapter change that needs one of those belongs there ([where should this change go?](where-should-this-change-go.md)).
