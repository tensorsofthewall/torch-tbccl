# DDP benchmarks

`ddp_bench.py` trains a small synthetic network under `DistributedDataParallel` on two ranks and times the training step.

* **Model.** `--layers` blocks of `Linear(width, width)` and `tanh`, an MSE loss against a fixed target, SGD without momentum, float32. Inputs and initial weights are deterministic from fixed seeds. Model M in the [results](results.md) is 8 layers, width 1024 (33.6 MB of gradients), batch 64 per rank.
* **Bucket size.** `--bucket-cap-mb` is DDP's gradient bucket cap. Smaller buckets start communication earlier but send more messages.
* **Phases** (each timed on the rank's own clock):
  1. *correctness* (not timed): a few steps compared with a CPU reference that never touches the communication backend (gradients, parameters, finite values, agreement between the ranks);
  2. *free*: `--steps` steps with one device synchronisation at the end of each, the realistic step time and the basis of steps per second;
  3. *phased*: the same steps with a measurement synchronisation after forward, backward and the optimiser step;
  4. *no_sync*: the same work under `DistributedDataParallel.no_sync()`, which has no communication, so it is this rank's compute-only time.
* **Reported step time.** The per-step maximum over the two ranks (the step ends when the slower rank finishes), then the median over steps.

"Exposed communication" (phased backward minus no-sync backward) is an inference from two loops of one process, not a measured interval. On the faster rank it also contains the time spent waiting for the slower rank. With a large batch the rank with less compute throughput sets the step, whatever the network does.

```sh
# one machine, CPU (a smoke test, not a performance claim)
RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=<PORT> TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 \
  python benchmarks/provider/ddp_bench.py --backend tbccl --devices cpu,cpu --layers 2 --width 256 --batch 16 --steps 8 &
RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=<PORT> TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 \
  python benchmarks/provider/ddp_bench.py --backend tbccl --devices cpu,cpu --layers 2 --width 256 --batch 16 --steps 8
wait
```

For two machines, start one rank on each as in [getting started](getting-started.md), with the link addresses; the runner automates the launch, the checks and the cleanup. Each rank prints one `TBCCL_TORCH_DDP {json}` line, including the raw per-step times.
