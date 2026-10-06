# torch-tbccl

> **Experimental.** An out-of-tree PyTorch distributed backend (`"tbccl"`) over an installed TBCCL runtime (libtbccl C ABI 1, wire protocol 4). Not NCCL feature parity.
> Validated tuple: torch-tbccl 0.2.0.dev0, **PyTorch 2.13.x**, **CPython 3.13**, Linux x86_64 (CPU + CUDA 13) and macOS arm64 (CPU + **MPS**, Phase 72). Anything else is untested.

```
PyTorch (torch.distributed) -> torch-tbccl -> libtbccl (linked statically into the extension) -> transports / collectives / device providers
```

torch-tbccl is an adapter and nothing more: all transport, algorithm and staging logic lives in libtbccl.

## Quick start

```python
import torch
import torch.distributed as dist
import torch_tbccl                      # registers the "tbccl" backend (required once per process)

dist.init_process_group("tbccl")

x = torch.tensor([dist.get_rank() + 1.0])
dist.all_reduce(x)                      # SUM
print(x)                                # tensor([3.]) on both ranks of a 2-rank group

dist.destroy_process_group()
```

Every rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>`: the address where this rank listens for TBCCL connections (**separate** from PyTorch's rendezvous store). `<host>:0` picks free ports, which is what you want
(many groups can coexist in one process). With `torchrun` on one machine:

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --standalone --nproc-per-node 2 your_script.py
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 torchrun --standalone --nproc-per-node 2 examples/ddp_mlp.py   # DDP, synthetic data, checked against a single-process reference
```

(torchrun registers the host's FQDN with its rendezvous; if the machine's hostname does not resolve, add `--local-addr 127.0.0.1` for single-machine runs.)

Across two hosts use PyTorch's normal rendezvous and give each host its own address in `TBCCL_LOCAL_ENDPOINT`:

```sh
# host A (rank 0)
TBCCL_LOCAL_ENDPOINT=<A address>:0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank 0 --master-addr <A address> --master-port 29500 train.py
# host B (rank 1)
TBCCL_LOCAL_ENDPOINT=<B address>:0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank 1 --master-addr <A address> --master-port 29500 train.py
```

`python -m torch_tbccl.info` prints what is installed (package, torch, libtbccl, C ABI, wire protocol, devices).

## Install

libtbccl is a **build-time** input: an installed TBCCL prefix (static libraries built position-independent). It is linked into the extension, so the wheel needs no TBCCL at run time.

```sh
# 1. an environment with the matching torch (the build imports it)
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install torch==2.13.0 setuptools wheel build            # Linux CUDA: add --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
# 2. build the wheel against the prefix and install it
TBCCL_ROOT=/path/to/tbccl-install python -m build --wheel --no-isolation -o dist .
uv pip install dist/torch_tbccl-*.whl
```

| Build | Needs |
|---|---|
| Linux x86_64 + CUDA | a prefix with `libtbccl_cuda.a` and a CUDA-enabled torch (its pip `nvidia/` runtime is used; the extension's RUNPATH is `$ORIGIN`-relative) |
| Linux host-only | a prefix without the CUDA component (or a CPU-only torch): CPU tensors only |
| macOS arm64 | TBCCL's host-only prefix and a torch built with MPS (the PyTorch macOS wheel): CPU and MPS tensors (adds an Objective-C++ file; needs the Xcode command-line tools to build) |

The build refuses a prefix with an unsupported C ABI. Never use `--reinstall` on an environment that holds torch (it rewrites torch); the extension is tied to torch's minor series and `import torch_tbccl`
refuses another one with a rebuild hint (`TORCH_TBCCL_ALLOW_TORCH_MISMATCH=1` overrides, at your own risk). For development: `TBCCL_ROOT=... uv pip install --no-build-isolation --no-deps -e .`.
Nothing here requires editing `site-packages`.

## Supported surface (measured; details in `docs/phase71_capability_audit.md` and `docs/phase71_capability_matrix.md`)

| Operation | Support |
|---|---|
| `init_process_group`, `new_group`, `destroy_process_group`, `torchrun` | world sizes 1-4 validated (5-8 accepted, untested); every group has its own communicator and ports |
| `send` / `recv` / `isend` / `irecv`, `batch_isend_irecv` | any dense contiguous dtype (bytes: bool, int8..int64, fp16/bf16/fp32/fp64, complex64, fp8), W2-W4; matched by FIFO per peer (tags ignored) |
| `broadcast`, `all_gather` | any dense contiguous dtype, W>=2, one tensor, equal sizes |
| `all_reduce` | **SUM only**: int8, uint8, int32, int64, float32, float64; float16/bfloat16 **at world size 2 only**; one tensor; async supported |
| `gather` | 2-rank groups only (so `gather_object` too); `all_gather_object` and `broadcast_object_list` work at W>=2 |
| `barrier` | W1-W4 |
| abort / failure | `_abort_process_group()` aborts the communicator; a peer exit fails the survivors in milliseconds; destroy with pending Work aborts it; no recovery |
| DDP | works at W2-W4 on CPU, with CUDA + CPU ranks, `find_unused_parameters`, any bucket size; **not at world size 1** |
| CPU, CUDA and MPS tensors | CUDA uses PyTorch's current stream at submission; mixed CUDA/CPU ranks work. **MPS (macOS, Phase 72)**: send/recv, isend/irecv, broadcast, all_gather, 2-rank gather, barrier and SUM all_reduce on MPS tensors (float32/float16/bfloat16/int32/int64/int8/uint8 reduce; any dtype PyTorch can create on MPS moves as bytes; no float64/float8 on MPS at all), W2 validated, DDP with an MPS rank next to a CPU rank. Same restrictions as CPU/CUDA, plus: every MPS operation first runs a device-wide MPS synchronize, only shared-storage buffers are supported, and ops outside the list above fail in PyTorch's own dispatcher. Details: `docs/phase72_mps_capability_matrix.md` |

Rejected with a clear `NotImplementedError` / `ValueError` before anything is communicated: `PRODUCT`/`MIN`/`MAX`/`AVG`/... reductions, dtypes without reduction arithmetic (int16, bool, complex, fp8),
`reduce`, `scatter`, `reduce_scatter(_tensor)`, `all_gather_into_tensor`, `all_to_all(_single)`, `scatter_object_list`, non-contiguous tensors (no hidden copy is ever made), sparse tensors, collectives on a one-rank group.
Not supported at all: FSDP (not validated), MPS tensors (PyTorch rejects them), fault recovery.

**Rules that are not enforceable by the type system**
* Collectives and point-to-point operations may be in flight on the same process group at the same time, in any relative order on each rank (libtbccl wire protocol 4 carries them on separate connections). The two contracts still hold inside each domain:
  every rank issues collectives in the same order, and P2P messages match FIFO per peer and direction. Needs libtbccl >= the wire-4 runtime; a wheel built against an older prefix keeps the Phase 71 rule (do not overlap them).
* `Work.wait(timeout)` raises on expiry but does not cancel the operation.
* Rendezvous stays with PyTorch; retrying `init_process_group` over the same store after a failed bootstrap is not supported (use a fresh store).

## Rendezvous

PyTorch rank rendezvous (`MASTER_ADDR`/`MASTER_PORT`, `TCPStore`, torchrun) and TBCCL's endpoint exchange are separate: through the group's store each rank publishes its ACTUAL control and data endpoints and rank 0 one communicator id under
`torch_tbccl/v3/g<generation>/{endpoint/<rank>,communicator_id}` (a per-group generation keeps repeated init over a persistent store apart); libtbccl itself never sees the store. With `TBCCL_LOCAL_ENDPOINT=<host>:<port>` the data
port is `port + 1000`; `<host>:0` picks free ports. Two machines: use each one's reachable address (e.g. the Thunderbolt link's); only the last rank needs no listener.

## Opt-in diagnostics

`TORCH_TBCCL_TRACE=1` (per-collective timeline via `torch_tbccl.trace_events()`), `TORCH_TBCCL_FORCE_SYNC_ALLREDUCE=1` (diagnostic baseline only, never a production mode).

## Tests

```sh
python -m pytest -q                                   # unit + loopback multi-process + DDP (CUDA tests skip without a GPU)
python tools/p71_capability_matrix.py run --worlds 1,2,3,4 --devmodes cpu,cuda0,cudaall --out matrix.json   # the full operation x dtype x world-size matrix
TBCCL_ROOT=... P71_CLEAN_INSTALL=1 python -m pytest tests/test_clean_install.py                              # wheel build, inspection, fresh venv, real collective
```

Markers: `multiprocess`, `cuda`, `ddp`, `packaging`, `physical` (anything that touches a second host or the Thunderbolt link is a script under `tools/`, never part of a generic run).
Multi-rank tests run on loopback first; set `OMP_NUM_THREADS` (e.g. 2) when running several ranks on one machine or they starve each other.

History and evidence: `docs/` (phase reports and results; Phase 71 starts at `docs/phase71_results.md`).
