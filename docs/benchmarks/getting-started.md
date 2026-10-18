# Getting started

These commands run from a clone of this repository, in the Python environment that has torch-tbccl installed ([install](../getting-started/install.md)). The supported tuple is in the [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html): CPython 3.13, PyTorch 2.13.x, Linux x86-64 (CPU, CUDA 13) or macOS arm64 (CPU, MPS).

## 1. Check what this environment can run

```sh
python benchmarks/provider/provider.py describe
```

This prints one JSON object and runs nothing else: it loads no model, initialises no device and opens no connection. Look at `availability`: `available`, `environment_missing` (it lists the missing package) or `unsupported` (an unsupported platform). The versions in `availability.detected` come from package metadata, not from importing PyTorch.

## 2. A first measurement on one machine

```sh
python benchmarks/provider/standalone.py --op all_reduce --bytes 4096 --backend tbccl
python benchmarks/provider/standalone.py --op p2p_ack --bytes 1048576 --backend gloo --json
```

Two ranks start on the loopback interface, run a warm-up and the timed iterations, verify the received data on every rank, and print a median and a 95th percentile. If any rank fails or the data does not verify, no measurement is printed and the command exits with status 1; both ranks are always ended. Operations: `all_reduce`, `broadcast`, `all_gather`, `p2p_ack`, `p2p_rtt`, `all_reduce_async`, `isend_irecv_ack`. Devices default to `cpu,cpu`; `--devices cuda,cpu` or `cpu,mps` need the matching hardware.

A loopback number measures the software path on one machine. It says nothing about a network or a Thunderbolt link.

## 3. Two machines

Start one rank on each machine at about the same time. Use the address of the link both machines share; `<RANK0_ADDRESS>` is rank 0's, `<MY_ADDRESS>` is the address of the machine you are on, and `<PORT>` is any free port on rank 0.

```sh
# machine 0
RANK=0 WORLD_SIZE=2 MASTER_ADDR=<RANK0_ADDRESS> MASTER_PORT=<PORT> TBCCL_LOCAL_ENDPOINT=<RANK0_ADDRESS>:0 \
  python benchmarks/provider/comm_bench.py --backend tbccl --devices cpu,cpu --op p2p_ack --bytes 4096 \
    --warmup 20 --iterations 100 --expect-addresses <RANK0_ADDRESS>,<RANK1_ADDRESS>

# machine 1
RANK=1 WORLD_SIZE=2 MASTER_ADDR=<RANK0_ADDRESS> MASTER_PORT=<PORT> TBCCL_LOCAL_ENDPOINT=<RANK1_ADDRESS>:0 \
  python benchmarks/provider/comm_bench.py --backend tbccl --devices cpu,cpu --op p2p_ack --bytes 4096 \
    --warmup 20 --iterations 100 --expect-addresses <RANK0_ADDRESS>,<RANK1_ADDRESS>
```

Each rank prints one line, `TBCCL_TORCH_BENCH {json}`, with its raw per-iteration samples, whether the data verified, the interpreter and library versions, and the TCP connections it used. `--expect-addresses` makes a rank report which of its peers were outside the two addresses you named, so a run that silently used another network can be recognised. Collect both lines; for point-to-point operations the time is in the *source* rank's line (`--source`, default 0), for collectives it is the per-iteration maximum over the two lines.

These two commands do not supervise the remote process, clean up after a crash, check link health or write a report. For anything beyond a quick look, use the runner.

## 4. Through the TBCCL benchmark runner

In a TBCCL checkout, add this provider to a host of a configuration (all paths and addresses are yours):

```yaml
hosts:
  host-a:
    executables:
      torch_python: <PATH_TO_PYTHON_WITH_TORCH_TBCCL>
      torch_bench_dir: <PATH_TO_THIS_REPOSITORY>/benchmarks/provider
```

Then:

<!-- doccheck: skip (runs from a TBCCL checkout; the runner is not in this repository) -->
```sh
python3 -m benchmarks.suite providers my-config.yaml          # lists torch-tbccl with its version and state, per host
python3 -m benchmarks.suite run my-config.yaml --dry-run      # prints the commands; launches nothing
python3 -m benchmarks.suite run my-config.yaml                # runs; add --allow-physical for a real link
python3 -m benchmarks.suite report <RUN_ID>                   # Markdown, CSV and plots from your own run
python3 -m benchmarks.suite compare <RUN_A> <RUN_B>           # compares only equivalent experiments
```

The runner records, in every result, which provider directory and interpreter it used and the framework versions the provider reported. A missing provider is reported as unavailable with a reason and does not stop native benchmarks.

## Tests

```sh
python -m unittest discover -s benchmarks/provider/tests -p 'test_*.py'
```

The provider tests run offline. With PyTorch installed they also run two real CPU ranks on loopback.
