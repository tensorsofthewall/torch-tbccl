# Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `ImportError` mentioning a torch series on `import torch_tbccl` | the wheel was built against a different torch minor series | rebuild against the installed torch ([installing](../getting-started/install.md)) |
| `ImportError` mentioning the C ABI | built against a TBCCL prefix with an unsupported C ABI | rebuild against a compatible prefix |
| a warning that the wire protocol is untested | the wheel was built against a TBCCL with a wire protocol version this release has not been tested with | rebuild all ranks against the same, tested prefix; ranks with different wire versions cannot connect |
| `protocol_mismatch ... wire protocol` at `init_process_group` | ranks were built against TBCCL prefixes with different wire protocol versions | rebuild every rank against the same prefix |
| `timeout` at `init_process_group` | a rank never connected, or `TBCCL_LOCAL_ENDPOINT` is not reachable from the peers | check the address each rank advertises |
| `NotImplementedError: torch-tbccl: unsupported operation` | an operation outside the [supported surface](../reference/supported-operations.md) | use a supported operation or another backend for it |
| a one-rank group rejects a collective | collectives need at least two ranks | do not use DDP at world size 1 |

After a failure the process group is not recoverable: a peer exit fails the survivors within milliseconds and `destroy_process_group` with pending work aborts it. `Work.wait(timeout)` raises on expiry but does not cancel the operation. Retrying `init_process_group` over the same store after a failed bootstrap is not supported; use a fresh store.
