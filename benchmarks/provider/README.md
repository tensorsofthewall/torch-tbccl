# PyTorch benchmark provider

Benchmark workloads for `torch.distributed` over torch-tbccl (and Gloo as a CPU baseline): `comm_bench.py` (collectives, point-to-point, asynchronous work), `ddp_bench.py` (DistributedDataParallel), `standalone.py` (two ranks on one machine) and `provider.py` (the entry point of the TBCCL benchmark provider protocol 1).

```sh
python benchmarks/provider/provider.py describe                       # what this environment can run (no framework import)
python benchmarks/provider/standalone.py --op all_reduce --bytes 4096  # a verified loopback measurement
python -m unittest discover -s benchmarks/provider/tests -p 'test_*.py'
```

Documentation: [docs/benchmarks](../../docs/benchmarks/index.md). These scripts need only this package and PyTorch; they import no other TBCCL adapter.
