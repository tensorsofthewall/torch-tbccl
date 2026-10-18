# Benchmarks

torch-tbccl ships its PyTorch benchmark workloads in `benchmarks/provider/`. They measure `torch.distributed` over torch-tbccl (and over Gloo as a CPU baseline) for point-to-point transfers, blocking and asynchronous collectives, and `DistributedDataParallel`, on **your** hardware.

```{toctree}
:maxdepth: 1

getting-started
communication
ddp
results
methodology
```

| I want to... | Read |
|---|---|
| see reference measurements with charts | [Results](results.md) |
| understand what a number does and does not mean | [Methodology and limitations](methodology.md) |
| run a benchmark on my machine in a minute | [Getting started](getting-started.md) |
| know what each communication benchmark times | [Communication benchmarks](communication.md) |
| measure a DDP training step across buckets | [DDP benchmarks](ddp.md) |
| generate a report from my run, or compare two runs | the TBCCL runner: [report generation](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/reports.html) and [comparing runs](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/comparing.html) |
| find out why a run failed | [Getting started](getting-started.md) for the script, [runner troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/troubleshooting.html) for the runner |

Two ways to run the same workloads:

* **Standalone.** Clone this repository, install torch-tbccl, and run `benchmarks/provider/standalone.py` (two ranks on one machine) or start `comm_bench.py` / `ddp_bench.py` on each host yourself. Nothing else needs to be installed: no vLLM, no exo, no TBCCL benchmark suite.
* **Through the TBCCL benchmark runner.** The [TBCCL benchmark runner](https://tbccl.tensorsofthewall.com/en/latest/development/benchmark-suite.html) drives the same scripts on several hosts, supervises and cleans up their processes, checks the data path, stores results in one schema and compares runs. This repository is a *provider* for that runner: `python3 -m benchmarks.suite providers CONFIG` lists it with the versions it found.

You never need the author's measurements to use either path, and your numbers must not be compared with the reference numbers unless your hardware, software and workload match (see [Methodology and limitations](methodology.md)).
