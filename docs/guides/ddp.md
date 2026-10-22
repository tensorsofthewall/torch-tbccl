# Distributed data parallel

`torch.nn.parallel.DistributedDataParallel` works over the `tbccl` backend at world sizes 2 to 4, on CPU, with CUDA and CPU ranks together, with `find_unused_parameters`, and with any bucket size. It does **not** work at world size 1, because collectives on a one-rank group are rejected.

DDP uses three c10d operations, all supported:

| c10d call | Role in DDP |
|---|---|
| `allgather` (one input, `world_size` outputs) | parameter-count verification at construction |
| `broadcast` (one tensor, root 0) | shape metadata and initial parameter and buffer synchronization; per-forward buffer sync |
| `allreduce` (`SUM`, Float32, one tensor) | one call per gradient bucket; the returned future is consumed by DDP's reducer |

All three go through one ordered executor in TBCCL, so the order of collectives is the issue order on each rank.

## Mixing point-to-point traffic with DDP

An application may exchange point-to-point messages on the same process group while DDP runs, from another thread, in any relative order on each rank. The only constraints are the usual ones: every rank issues collectives in the same order and P2P messages match in FIFO order per peer. `examples/ddp_mlp.py --side-p2p N` demonstrates this. On an MPS rank, use CPU tensors for such side traffic (see [MPS](mps.md)).

## Faster gradient AllReduce between two hosts (opt-in)

With two ranks on two hosts, the core runtime can overlap the two directions of the link inside each `allreduce`. It is off by default and is set with an environment variable that torch-tbccl passes through unchanged:

```sh
export TBCCL_ALLREDUCE_PIPELINE_BYTES=131072   # on both ranks, before init_process_group
```

* **Set it on both ranks.** The value is not exchanged with the peer. One rank on its own gains little and can be slower (see the [TBCCL pipeline notes](https://tbccl.tensorsofthewall.com/en/latest/development/n2-all-reduce-pipeline.html)).
* **It needs a TBCCL that has the pipeline.** torch-tbccl links the core runtime statically, so the variable only has an effect in a torch-tbccl build made against a TBCCL that includes the pipeline; a build against an earlier core ignores it.
* **Payloads under 256 KiB are unchanged**, and it does not apply to world sizes other than 2. Do not enable it for ranks on one machine: on loopback it was slower at small and medium gradient sizes.

Measured on one Linux and Mac mini pair joined by Thunderbolt 4, with the synthetic MLP of the benchmark suite (about 33.6 MB of float32 gradients), batch 64 per rank, five launches per setting in each rank orientation, setting `0` against `131072` on both ranks:

| Configuration | Step time without / with the setting |
|---|---|
| CUDA and MPS, 25 MB bucket cap | 55.8 ms to 26.7 ms and 55.2 ms to 24.1 ms (52% and 56% lower) |
| CUDA and MPS, 1 MB bucket cap (8 buckets) | 85.5 ms to 26.9 ms and 100.7 ms to 27.2 ms (69% and 73% lower) |
| CUDA and MPS, batch 256 per rank | 62.5 ms to 32.8 ms and 55.6 ms to 30.3 ms (48% and 45% lower) |
| CPU and CPU, 25 MB bucket cap | 62.6 ms to 38.0 ms and 51.7 ms to 33.8 ms (39% and 35% lower) |

With the setting on, the step time barely depended on the bucket cap. Limitations: one machine pair, one link, one synthetic model whose step is dominated by communication, float32, two ranks. A model with much more computation per step gains proportionally less, because only the gradient exchange gets faster, and other hardware or links may differ. Results were identical between the two settings in the benchmark's correctness check. Measure your own workload with `benchmarks/provider/ddp_bench.py` ([DDP benchmarks](../benchmarks/ddp.md)).

### Thread-pool contention on the Mac rank

PyTorch's intra-op thread pool (OpenMP) can compete with TBCCL's communication threads on a small machine. On the Mac rank, `OMP_NUM_THREADS=1` removed about 2 ms of the roughly 3 ms of extra latency a 16 MiB AllReduce showed inside PyTorch compared with the native benchmark; the effect was the same with and without the pipeline. This is a tuning suggestion, not a requirement: it also limits the CPU computation of that process, so try it on your workload and keep it only if the step time improves.

## Example

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --standalone --nproc-per-node 2 examples/ddp_mlp.py
```
