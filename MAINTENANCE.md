# Maintenance

The shared maintenance policy of the TBCCL ecosystem (branches, tag immutability, pinning, release cadence, security and correctness fixes, upstream updates, what needs physical requalification, automation and its failure categories) is [MAINTENANCE.md in tbccl](https://github.com/tensorsofthewall/tbccl/blob/main/MAINTENANCE.md); the versioning rules are in [VERSIONING.md](https://github.com/tensorsofthewall/tbccl/blob/main/VERSIONING.md). This file records only what is specific to torch-tbccl.

## Supported release

torch-tbccl 0.2.1: PyTorch 2.13.x, CPython 3.13, TBCCL 0.6.1 bundled (minimum 0.6.0; C ABI 1, wire protocol 4), Linux x86-64 (CUDA 13) and macOS arm64. The current ecosystem status (all four projects and their pins) is in the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html), built from [`release/ecosystem.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/ecosystem.json). Documentation built from a release tag describes that release only and is not updated when other projects release.

## Pins

This adapter has no framework fork pin. It follows the PyTorch 2.13 series (`TESTED_TORCH_SERIES` in `torch_tbccl/_version.py`); the extension refuses to import under another series.

## What needs physical requalification here

Runtime sources of this repository: `csrc/` and `torch_tbccl/` (everything that is not documentation, tests, tools or CI). A change to them, to the framework pin or to the supported TBCCL line needs physical requalification: `tools/physical_ops.py` (CUDA, CPU and MPS operations) and `tools/hetero_ops.py` (CUDA to MPS) over the Thunderbolt link, run from the public wheels. Documentation, tests, CI, workflows, release metadata and GitHub Release edits do not. The full matrix is in the shared policy.

## Automation in this repository

`docs` workflow (every pull request, offline): release state, release documentation (README, homepage and installation commands against the package version), tooling tests, internal links. `maintenance` workflow (weekly and by hand): release state, stable release surface, links, documentation hosts and the documentation build. `release-preflight` workflow (by hand, with the intended tag; it checks the release documentation against that tag before anything is tagged). The `guard` job of `release-artifacts` refuses a build whose tag, version, manifest, pin or TBCCL line disagree. `tools/verify_public_release.py` verifies a published release from the outside, and `tools/collect_diagnostics.py` prints the environment report requested by the issue template.
