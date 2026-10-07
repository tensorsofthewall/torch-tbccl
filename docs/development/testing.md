# Testing

```sh
python -m pytest -q                       # unit tests, loopback multi-process tests, DDP; CUDA tests skip without a GPU
python tools/capability_matrix.py run --worlds 1,2,3,4 --devmodes cpu,cuda0,cudaall --out matrix.json   # the full operation x dtype x world-size matrix
TBCCL_ROOT=... P71_CLEAN_INSTALL=1 python -m pytest tests/test_clean_install.py   # wheel build, inspection, a fresh virtual environment, a real collective
```

Markers: `multiprocess`, `cuda`, `ddp`, `packaging`, `physical`. Anything that touches a second host or the Thunderbolt link is a script under `tools/` and never part of a generic run. Multi-rank tests run on loopback first; set `OMP_NUM_THREADS` when running several ranks on one machine.

The packaging checks build a real wheel and install it into a clean environment; validate packaging that way, never with an editable install. Set `P71_SCRATCH` to a disk with about 6 GB free when `/tmp` is a small tmpfs.
