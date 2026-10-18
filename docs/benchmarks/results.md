# Results

These are reference measurements from one pair of machines. They show what the benchmarks produce and how large the effects were there; they are not a statement about other hardware, other link types, other PyTorch builds or other workloads. Run the benchmarks on your own systems for numbers that apply to them.

## Performance at a glance

| What was measured | Result on this pair of machines |
|---|---|
| DDP training step, 1 MB against 100 MB gradient buckets | 82.2 ms down to 54.4 ms (model M, batch 64 per rank) |
| 16 MiB send with acknowledgement, PyTorch against the native benchmark | 10,740 us against 8,697 us: an observed difference, classified *indeterminate* |
| `all_reduce`, 1 MiB, blocking | 1,796 us (range 1,763-1,850) |
| `all_reduce(async_op=True)` | submission returned in 10 to 91 us; that is submission, not completion |

These numbers come from one pair of machines (a Linux CUDA host and a Mac) joined by one Thunderbolt 4 cable and one PyTorch build. Each chart below is drawn from the table next to it; the table is the record and the chart only makes its shape easier to see. The charts need no scripts and follow the light and dark theme.

## What was measured

| | Linux host | macOS host |
|---|---|---|
| machine | Intel Core i7-12800HX laptop, NVIDIA RTX 3070 Ti Laptop GPU (8 GiB), CUDA 13 | Apple M4 Mac mini, 16 GiB unified memory |
| operating system | EndeavourOS, Linux 6.18 LTS | macOS 26.5.1 |
| link | one direct Thunderbolt 4 cable between the two machines, IP over Thunderbolt, MTU 9000, TCP/IPv4 | |
| software | PyTorch 2.13.0, torch-tbccl 0.2.0.dev0 (the development build that became 0.2.0), TBCCL 0.6.0 (C ABI 1, wire protocol 4), CPython 3.13 | same |

Rank 0 ran on the Linux host and rank 1 on the Mac. "A to B" means the Linux host is the source.

**Hardware health.** The Linux host was booted with `pci=noaer`, so PCIe Advanced Error Reporting was disabled for the whole campaign. Do not read these results as "zero PCIe errors". The statement that holds is: *AER unavailable; no detected link-state or interface-health regression under the available checks.* Link state, interface counters and thermal state were checked before and between runs.

## Point-to-point and blocking collectives, CPU tensors

Timing scope: `p2p_ack` is the source's time from before `send` until the destination's acknowledgement arrived; `p2p_rtt` is a round trip; the collective is the per-iteration maximum over the ranks. Each cell is the median over independent process-pair repetitions of that repetition's median (six pairs for the comparison rows, in three interleaved native/torch-tbccl blocks). Correctness: every transfer was verified bit for bit.

| Workload (cpu/cpu, float32) | torch-tbccl | native TBCCL, same scope | Notes |
|---|---|---|---|
| send/recv + ACK, 4 KiB, A to B | 242 us | two modes: 124, 125, 127, 227, 233, 263 us | at this size the process pairs fall in two latency groups (about 125 us or 230-260 us), so a single "difference" does not exist |
| send/recv + ACK, 16 MiB, A to B | 10,740 us | 8,697 us | +22.3% in the paired comparison (exact signed-rank p = 0.031); the native baseline itself varied 26% between process pairs |
| ping-pong round trip, 4 KiB | 247 us | not measured in the same scope | a round trip; not halved into a one-way latency |
| blocking `all_reduce`, 1 MiB | 1,796 us (range 1,763-1,850) | not comparable | the PyTorch call returning on the slowest rank is a different scope from the native benchmark's |

**How to read this.** The 16 MiB difference is an *observed difference between two measurements* made in the same way. It is not an isolated "adapter overhead": it includes the Python and PyTorch dispatch path, the way the benchmark builds and verifies tensors, and anything else that differs between the two programs, and this campaign did not separate those. Under the strict comparison rule used by the TBCCL benchmark runner, with a native baseline that itself varied by 26%, the classification for both rows is *indeterminate*, not *regression*. Treat 4 KiB as latency- and variability-bound.

```{raw} html
:file: charts/p2p-native-comparison.html
```

The chart is drawn so that the classification is part of it. Two bars that differ by 22% look like a finding, and the evidence does not support one as an adapter cost: the two programs differ in more than the adapter, and the native baseline moved by about as much between process pairs. Read it as "the PyTorch path took longer in this comparison", not as an overhead or a regression.

## Asynchronous work

`all_reduce(async_op=True)` returned in 10 to 91 us, and its lifetime (submission to the end of `wait()`) was 0.8 to 1.1 times the blocking call for the same size. These figures come from the campaign's original analysis and were not recomputed afterwards. Submission time is not completion time; compare lifetimes with lifetimes.

## DDP training step, CUDA and MPS

Linux CUDA rank and macOS MPS rank, model M (8 layers, width 1024, 33.6 MB of gradients), batch 64 per rank, float32, tbccl backend. The value is the median step time of the slower rank, then the median of three independent repetitions of 20 steps each.

| bucket cap | step time (median) | range over 3 repetitions |
|---|---|---|
| 1 MB | 82.2 ms | 80.8-85.5 ms |
| 5 MB | 62.2 ms | 61.8-64.8 ms |
| 25 MB | 57.4 ms | 55.9-57.7 ms |
| 100 MB | 54.4 ms | 53.8-55.4 ms |

Larger buckets were faster here (82 ms down to 54 ms). The step is communication-bound for this model on this link: the gradient exchange continues after the last gradient is ready, and the faster accelerator waits for the slower rank. With much larger batches the compute imbalance between the accelerator and the Mac CPU decides the step instead. Three repetitions are enough to see the size of the effect, not to rank bucket sizes that differ by less than the spread.

```{raw} html
:file: charts/ddp-bucket-cap.html
```

Larger buckets were faster in this workload, but 100 MB is not shown to be a best size in general: it is the largest value that was tried, 25 MB and 100 MB are separated by less than their combined spread, and a model with more or fewer gradients, a different link or a different number of ranks can move the optimum. The measurement says what happened here; it is not a tuning rule.

## Using these results

What was tested is limited to what the tables above show. Anything else is a reason to measure, not a finding.

| Question | Tested here | Where to look |
|---|---|---|
| Does the DDP bucket cap matter? | Yes, for one model, one link and two ranks: step time fell from 82.2 ms to 54.4 ms as the cap grew from 1 MB to 100 MB | [DDP benchmarks](ddp.md) (`--bucket-cap-mb`); [DDP guide](../guides/ddp.md) |
| Does overlapping computation with communication help? | Not measured as a separate experiment. DDP overlaps the gradient exchange with the backward pass by design; the benchmark reports exposed communication as an inference from two loops | [DDP benchmarks](ddp.md) |
| Does the number of ranks change the result? | No. Only two ranks were measured | [Methodology](methodology.md) |
| Is traffic really using the transport and the device path you expect? | Each rank records its TCP connections, and the runner rejects a run whose traffic left the configured addresses | [Methodology](methodology.md); [verify the active transport](../guides/verify-active.md) |

## Limitations of these results

* One pair of machines, one cable, one PyTorch build. Nothing here applies to other Thunderbolt generations, other operating systems or other GPU families.
* A development build of torch-tbccl (0.2.0.dev0) was measured; re-measure on the release before quoting.
* Three to six independent repetitions per cell: enough for effects of tens of percent, not for small ones.
* Round trips are reported as round trips. Differences between PyTorch and the native benchmark are observations, not adapter costs.
* AER was unavailable (see above).

## How the charts are made

The chart inputs are in `docs/benchmarks/chart-data.json`, copied by hand from the tables on this page. They are published aggregates, not raw records, and carry the exact text of each cell. `python3 tools/benchmark_charts.py` regenerates the figures in `docs/benchmarks/charts/` deterministically and `--check` fails when they are out of date; a test fails when a chart input no longer appears in the table it cites. See [methodology](methodology.md) for what each value means.
