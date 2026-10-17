# Methodology and limitations

* **Independent unit.** The replication unit is an independent process-pair repetition (a fresh pair of processes and a fresh process group), never an iteration. Iterations inside a repetition are samples of that repetition; they are not independent of each other. Statistics that need independence use the repetition medians.
* **Warm-up and iterations.** Each run discards warm-up iterations and keeps raw per-iteration samples. Percentiles use linear interpolation. With 100 iterations or fewer the 95th and 99th percentiles are not meaningful tail estimates and are labelled so.
* **Correctness outside the timer.** Data are verified once, after the timed loop, on the receiving rank.
* **Completion.** A result says what completion means for it: the call returned, the data was received and acknowledged, or the work handle was waited on. Asynchronous submission is never reported as completion.
* **Clocks.** Every interval is measured with one process's monotonic clock. Timestamps from different machines are never subtracted.
* **Data path.** Each rank records its TCP connections. The runner rejects a run whose traffic left the two addresses you configured (for example one that fell back to a Wi-Fi or LAN path).
* **Matching.** Compare runs only when device pairing, direction, size, backend, completion semantics, PyTorch and torch-tbccl versions and hardware match. The runner's `compare` command refuses to compare experiments that differ in these, and a different machine is a different benchmark environment.
* **Observed difference versus overhead.** A difference between PyTorch-over-torch-tbccl and the native TBCCL benchmark is an observation. Calling it adapter overhead needs an experiment that isolates the adapter.
* **Not covered.** Training convergence, model quality, multi-node scaling beyond two ranks, and any claim about the Thunderbolt link rate (these benchmarks measure TCP goodput through two kernels, not the physical link).
