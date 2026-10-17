# Communication benchmarks

`comm_bench.py` times one `torch.distributed` operation between two ranks, float32, one tensor per iteration. All times are microseconds on one process's own `perf_counter_ns`; no clocks of different hosts are ever subtracted.

| Operation | What is timed | Reported value |
|---|---|---|
| `all_reduce`, `broadcast`, `all_gather` | the blocking `dist.*` call, from just before it until it returns on that rank; the barrier and the source refill are outside the timer | per-iteration maximum over the two ranks |
| `p2p_ack` | on the source: from before `dist.send` until a 1-byte CPU acknowledgement from the destination was received | the source rank's time. It **includes the acknowledgement's return trip**; it is not a one-way latency |
| `p2p_rtt` | on the source: `dist.send` plus `dist.recv` of the echo | the source rank's time: a **round trip** |
| `all_reduce_async` | `dist.all_reduce(async_op=True)` returning a `Work` (submission), the time inside `Work.wait()` (wait), and submission start until `wait()` returned (lifetime) | per-iteration maximum over the two ranks, for each of the three |
| `isend_irecv_ack` | like `p2p_ack`, with `isend`/`irecv` and `Work.wait()` | the source rank's time |

Submission is not completion: an asynchronous call returns long before the data has moved, and a result that reported submission time as transfer time would be wrong. The three asynchronous series are kept apart for that reason. Round-trip times are never halved into a one-way figure.

For a device tensor (`cuda`, `mps`) the script also records a measurement-only device synchronisation after the call (`dev_sync_us`) and the sum (`device_ready_us`). That synchronisation is not part of the framework call; it only shows when the device side is ready.

## Correctness

After the timed loop, one untimed transfer is compared bit for bit on the receiving rank, with the device preserved. A rank that fails the comparison reports `verify_ok: false`, and the runner and `standalone.py` then report no measurement. Correctness is never checked inside a timed loop.

## Backends and devices

`--backend tbccl` uses this package's `"tbccl"` process group. `--backend gloo` is a CPU baseline on CPU tensors only (the script binds Gloo to the address you give it, because an interface name can resolve to an IPv6 link-local address). Gloo is not claimed for CUDA or MPS pairings, and across a Linux and a macOS PyTorch build the two Gloo stores do not rendezvous, so there is no cross-host Gloo baseline.

`--devices a,b` gives the device of rank 0 and rank 1 (`cpu`, `cuda`, `mps`). A pairing the package does not support is rejected before anything runs; nothing is silently moved to the CPU.

## Output

Each rank prints one `TBCCL_TORCH_BENCH {json}` line: the operation, size, devices, warm-up and iteration counts, the raw per-iteration samples for every timed series, the verification result, CPU seconds, the interpreter, PyTorch, torch-tbccl and libtbccl versions and the TCP connections the process held. Logs go to stderr.
