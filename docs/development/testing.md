# Building and testing

Copy-pasteable commands. Run them from the repository root, in an environment with the matching PyTorch and a **built** torch-tbccl (see [installing](../getting-started/install.md); use `TBCCL_ROOT=<prefix>` on every build).

## Python tests

```sh
python -m pytest -q                       # unit tests, loopback multi-process tests, DDP; CUDA tests skip without a GPU, MPS tests skip off macOS
python -m pytest -q tests/test_readme_example.py   # the README/quickstart example, run for real on two ranks
python tools/capability_matrix.py run --worlds 1,2,3,4 --devmodes cpu,cuda0,cudaall --out matrix.json   # the full operation x dtype x world-size matrix
```

Markers: `multiprocess`, `cuda`, `ddp`, `packaging`, `physical`. Multi-rank tests run on loopback first; set `OMP_NUM_THREADS=2` when running several ranks on one machine. Anything that touches a second host or the Thunderbolt link is a script under `tools/` and never part of a generic run.

## Wheel build and inspection

Validate packaging with a real wheel installed into a clean environment, never with an editable install:

```sh
TBCCL_ROOT=<prefix> python -m build --wheel --no-isolation -o dist .
python tools/package_inspect.py dist/torch_tbccl-*.whl --allow-local-platform     # members, metadata, no embedded machine paths, expected libraries (a local Linux wheel has a plain linux tag)
TBCCL_ROOT=<prefix> TORCH_TBCCL_TEST_CLEAN_INSTALL=1 python -m pytest tests/test_clean_install.py    # build, inspect, fresh venv, a real collective
```

A publishable wheel is built in a manylinux_2_28 environment (Linux) or with `MACOSX_DEPLOYMENT_TARGET` set (macOS) by `TBCCL_ROOT=<prefix> tools/build_release_wheel.sh <out-dir>`, which repairs and then inspects it; CI does this for releases. Set `TORCH_TBCCL_TEST_SCRATCH` to a disk with about 6 GB free when `/tmp` is a small tmpfs.

## Documentation and manifest checks

```sh
make docs              # builds docs/_build/html with warnings as errors (creates .venv-docs)
make docs-linkcheck    # checks local links and anchors
python tools/check_compatibility_manifest.py    # compatibility.json against torch_tbccl/_version.py and pyproject.toml
```

## Release artifacts

CI (`.github/workflows/release-artifacts.yml`, run by a maintainer on a tag) builds the wheels, inspects them, installs them into clean environments and runs a collective and the test suite. Maintainers verify published wheels with `SHA256SUMS`, the SPDX SBOM and `gh attestation verify` as described in `RELEASE.md`.
