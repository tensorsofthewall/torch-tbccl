# Contributing to torch-tbccl

Thank you for contributing. This document describes how to set up, test and submit changes. Technical rules for the code base are in `AGENTS.md`.

## Development setup

You need Python 3.13, PyTorch 2.13.x (the tested tuple), a C++17 toolchain and an installed TBCCL package with wire protocol >= 4 (`TBCCL_ROOT`). Use `uv` for environments.

## Building and testing

```sh
uv venv --python 3.13 .venv
TBCCL_ROOT=<installed tbccl prefix> uv pip install --python .venv/bin/python --no-build-isolation -e .
.venv/bin/python -m pytest -q
```

Set `TBCCL_ROOT` on every build and read the build output (a failed rebuild leaves the old extension in place). Do not use `uv pip install --reinstall`, which rewrites torch. Several ranks on one machine need `OMP_NUM_THREADS` set.

Run the full pytest suite for any change. Multi-rank tests run on loopback. GPU, MPS and Thunderbolt runs are opt-in harnesses; say in the pull request what you ran. Validate packaging changes with a built wheel in a clean environment, not an editable install.

## Contribution workflow

1. Open an issue for anything beyond a small fix, so the approach can be agreed first.
2. Fork or branch from `main` and keep each pull request focused on one change.
3. Make sure the build and tests pass locally and add tests for new behavior and for bug fixes.
4. Open a pull request using the template and describe what changed, why, and what you ran.
5. Maintainers review every pull request. Address review comments with follow-up commits; maintainers may ask you to squash before merging.

Project policy: changes to `main` land through pull requests, and `main` is never force-pushed. Repository settings may or may not enforce this; the policy applies either way.

## Commit messages

Use short, descriptive subjects in the form `area: summary` (imperative, no trailing period), with a body that explains why when it is not obvious. Internal tracking numbers are not required in subjects.

```text
transport: add collective data connection
protocol: validate connection roles
test: cover mixed-domain ordering
docs: document wire compatibility
ci: add documentation checks
```

## Pull request expectations

- The change builds and the relevant tests pass; state which platforms and hardware you tested on.
- Behavior changes come with tests; documentation is updated alongside code.
- No unrelated formatting or refactoring in the same pull request.
- Do not commit machine-specific paths, host names, credentials or model weights.

## Documentation

User-visible changes update the relevant documentation (`README.md` and `docs/`) in the same pull request. Document behavior that exists, not behavior you intend to add.

## Compatibility requirements

- **Tested tuple:** one PyTorch minor series, one CPython version and one libtbccl C ABI / wire protocol combination (see `README.md` and `torch_tbccl/_version.py`). Extending it needs validation evidence.
- **Supported surface:** the supported operations are a tested contract (`tests/test_capability_matrix.py`). Unsupported operations must keep raising a clear error.
- **TBCCL changes:** generic defects are fixed in the TBCCL repository, never worked around here.

## AI-assisted contributions

AI-assisted contributions are permitted. Contributors remain responsible for the correctness, licensing, testing, and review of their submissions. Material AI assistance should be disclosed according to the project's contribution guidelines: add trailers to the commit message, for example

```text
AI-Assisted-By: <tool or assistant>
AI-Assistance: documentation | tests | benchmark tooling | build automation | mechanical | implementation
```

and use `AI-Validated-By: <tool>` only when the tool itself ran and recorded the validation the commit reports. Do not list an AI tool as an author, co-author, signer or reviewer.

## Reporting problems

Open an issue with the version, platform, how to reproduce, and the observed and expected behavior. Please do not include credentials or private network details.
