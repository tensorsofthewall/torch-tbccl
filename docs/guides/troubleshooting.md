# Troubleshooting

Find the symptom, check the likely cause with the diagnostic, apply the fix. Start with `python -m torch_tbccl.info` on every machine ([how do I know TBCCL is being used?](verify-active.md)).

## Start here

### Work through it in this order

Each step is read-only. Stop at the first step that fails: later steps depend on it.

| Step | Question | Check | If it fails |
|---|---|---|---|
| 1 | Which versions are installed? | `.venv/bin/python -m torch_tbccl.info` on **every** machine (same wire protocol, `registered True`) | [Installing and importing](#installing-and-importing) |
| 2 | Is the device you need available? | `.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"` | [Installing and importing](#installing-and-importing); [MPS](mps.md) |
| 3 | Is the link ready? | `python3 thunderbolt_link_doctor.py --peer <peer-link-address>` (read-only; [how to get it](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html)) | [Thunderbolt link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| 4 | Did `torchrun` find every rank? | the `torchrun` output; rank 0's `--master-addr` and `--master-port` reachable from the other machine | [ping works but the ranks time out](#ping-works-but-the-ranks-time-out) |
| 5 | Did the TBCCL transport come up? | `TBCCL_LOCAL_ENDPOINT` set on every rank; `init_process_group` returns | [Process-group creation](#process-group-creation) |
| 6 | Does the collective run? | a one-element `all_reduce`; `TORCH_TBCCL_TRACE=1` ([verify](verify-active.md)) | [Operations](#operations) |
| 7 | Did a failure leave something behind? | `lsof -iTCP:<master port> -sTCP:LISTEN` on the rank-0 machine | [Failure and lifecycle](#failure-and-lifecycle) |
| 8 | Is it only slow? | the [benchmark methodology](../benchmarks/methodology.md) | [Results](../benchmarks/results.md) |

### Find your symptom

| You see | Go to |
|---|---|
| `ImportError` on `import torch_tbccl`, or no matching wheel | [Installing and importing](#installing-and-importing) |
| the backend is not registered or `"tbccl"` is unknown | `python -m torch_tbccl.info` must print `registered True`; [verify](verify-active.md) |
| `TBCCL_LOCAL_ENDPOINT is not set` | [Process-group creation](#process-group-creation) |
| `protocol_mismatch ... wire protocol` | [Process-group creation](#process-group-creation); install the same wheel everywhere |
| `torchrun` prints nothing for minutes, or `timeout` at `init_process_group` | [Ping works but the ranks time out](#ping-works-but-the-ranks-time-out) |
| `NotImplementedError: torch-tbccl: unsupported operation` | [Operations](#operations) |
| an MPS assertion or crash | [Operations](#operations), [MPS](mps.md) |
| processes do not exit after a failure | [Failure and lifecycle](#failure-and-lifecycle) |

### Which layer owns the error

| Layer | What it looks like | What to do |
|---|---|---|
| torch-tbccl (this adapter) | a message that starts with `torch-tbccl:` | this guide; report with `python -m torch_tbccl.info` |
| Native TBCCL | a result tag inside the message (`protocol_mismatch`, `timeout`); the adapter surfaces TBCCL errors | [TBCCL troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/troubleshooting.html) |
| PyTorch | `torchrun` rendezvous waits, c10d store errors, `torch.mps.synchronize()` threading, which CUDA build of PyTorch you installed | PyTorch's own documentation; this guide names the known cases |
| Operating system | a macOS Local Network or firewall prompt, a port in use | [link troubleshooting entries 9 and 10](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| Network configuration | wrong subnet, traffic leaving by Wi-Fi, mismatched MTU | [link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| External dependencies | the PyTorch wheel index, `uv`, the CUDA runtime that PyTorch brings | their own notes; `python -m torch_tbccl.info` shows what is loaded |

## Installing and importing

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `ImportError` mentioning `libtorch_cuda.so` or `libc10_cuda.so` on Linux | a CPU-only PyTorch; the Linux wheel needs the CUDA 13 build | `python -c "import torch; print(torch.__version__)"` (expect `2.13.x+cu130`) | install `torch==2.13.0` from `https://download.pytorch.org/whl/cu130` ([installing](../getting-started/install.md)) |
| `ImportError` mentioning a torch series on `import torch_tbccl` | the wheel was built against a different torch minor series | `python -m torch_tbccl.info` (`torch_running` vs `torch_built_with`) | install PyTorch 2.13.x, or rebuild against the installed torch |
| no matching distribution for the wheel | Python is not 3.13, or the platform is not Linux x86-64 / macOS arm64 | `python --version` | use Python 3.13 on a supported platform |
| `ImportError` mentioning the C ABI | built from source against a TBCCL prefix with an unsupported C ABI | `tbccl-info` of the prefix | rebuild against a compatible prefix |
| a warning that the wire protocol is untested | built against a TBCCL wire protocol this release was not tested with | `python -m torch_tbccl.info` | use the released wheel; ranks with different wire versions cannot connect |
| installing the wheel changed my torch | the wheel was installed without `--no-deps` | `pip list \| grep torch` | reinstall the intended torch and use `--no-deps`; never use `--reinstall` |
| `UserWarning: Failed to initialize NumPy: No module named 'numpy'` at import | the environment has no NumPy; PyTorch warns once | `python -c "import numpy"` | harmless for torch-tbccl (the validated installs have no NumPy and run `all_reduce` on CPU, CUDA and MPS); `uv pip install --python .venv/bin/python numpy` silences it |

## Process-group creation

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `ValueError: torch-tbccl: invalid argument: TBCCL_LOCAL_ENDPOINT is not set` | the endpoint variable is missing on a rank of a group with two or more ranks | `echo $TBCCL_LOCAL_ENDPOINT` on every host | set `TBCCL_LOCAL_ENDPOINT=<host>:0` per rank, with a host that the peers can reach |
| `protocol_mismatch ... wire protocol` at `init_process_group` | ranks run torch-tbccl wheels built against different TBCCL wire protocols | `python -m torch_tbccl.info` on each host | install the same torch-tbccl version everywhere |
| `timeout` at `init_process_group` | a rank never started, never published its endpoint, or `TBCCL_LOCAL_ENDPOINT` is not reachable from the peers | `ping` the peer address; check that every rank started | start every rank; use an address reachable from the peers (not `127.0.0.1` across hosts) |
| `torchrun` shows no output for many minutes when the other machine has not started | `torchrun`'s static rendezvous waits up to 15 minutes for every node unless told otherwise | `ping` and TCP-check the peer with the [link doctor](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html); check the other machine's command ran | start the other rank; add `--rdzv-conf timeout=60` so the wait is bounded (the two-host quickstart commands do) |
| the second `init_process_group` after a failed first one fails | retrying over the same store after a failed bootstrap is not supported | | create a fresh store (restart `torchrun`, or use a new `TCPStore`/`HashStore`) |
| hostname does not resolve under `torchrun` on one machine | the default local address is the hostname | | add `--local-addr 127.0.0.1` |
| a Mac rank hangs at start while Linux is fine | macOS firewall or Local Network prompt not yet answered, or a background job without a live session | look for a dialog on the Mac's screen | answer the prompt once from a local session; run the rank in a live terminal or SSH session |

## Operations

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `NotImplementedError: torch-tbccl: unsupported operation ...` | an operation outside the [supported surface](../reference/supported-operations.md) (for example `reduce`, `MAX`) | the message names it | use a supported operation or another backend for it |
| a one-rank group rejects a collective | collectives need at least two ranks | | do not use DDP at world size 1 |
| `ValueError: ... tensor must be contiguous (no implicit copy is made)` | the adapter never makes a hidden copy | `tensor.is_contiguous()` | pass `tensor.contiguous()` |
| float16/bfloat16 `all_reduce` rejected | those reductions work at world size 2 only | | use float32 or run two ranks |
| MPS: a Metal assertion or a crash with several threads | `torch.mps.synchronize()` is not safe from two threads | | submit operations on MPS tensors from one thread; use CPU tensors for side traffic ([MPS](mps.md)) |

## Ping works but the ranks time out

**Symptom.** `ping` answers both ways, yet `torchrun` waits with no progress, or `init_process_group` raises `timeout`.

**Two different waits.** `torchrun` first needs every node to reach rank 0's rendezvous port (`--master-port`, PyTorch's own store). Only after that does TBCCL connect the ranks' own ports (`TBCCL_LOCAL_ENDPOINT`). A hang before any `torch-tbccl:` message is the first; `timeout` raised by `init_process_group` is usually the second.

**Checks (read-only).** Run them while rank 0 is waiting.

```sh
# 1. From the other machine: can it open a TCP connection to rank 0's rendezvous port?
python3 -c "import socket; socket.create_connection(('<rank-0-link-address>', <master-port>), timeout=3).close(); print('TCP connect ok')"
# 2. Does traffic to rank 0 leave by the link you intend?
ip route get <rank-0-link-address>          # Linux
route -n get <rank-0-link-address>          # macOS
echo $TBCCL_LOCAL_ENDPOINT                  # on every rank: this machine's link address, not a hostname or 127.0.0.1
```

- `TCP connect ok` on the rendezvous port but a `timeout` later: the TBCCL ports are the problem. The `:0` port is chosen at start, so test with the doctor's TCP mode ([link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html), entry 8) and check `TBCCL_LOCAL_ENDPOINT` on every rank.
- `timed out`: a firewall, a wrong address, or a route through another interface. The second command shows which interface is used.
- `Connection refused`: rank 0 is not running, or listens on another address.

On macOS check the Local Network prompt and that the rank runs in a live session. **Do not** change firewall rules, MTU or interface settings as a first step.

## Failure and lifecycle

Expectations are explicit: **there is no recovery.** After a failure the process group is not usable and must be destroyed and recreated.

- **Peer loss.** If a peer exits or dies, the surviving ranks' operations fail within milliseconds with a `torch-tbccl:` error.
- **Abort.** PyTorch's `torch.distributed.distributed_c10d._abort_process_group()` (a private PyTorch function) aborts the communicator through the backend; pending work fails. Use it to unblock a process whose peer is gone.
- **Destroy with pending work.** `dist.destroy_process_group()` with un-waited work aborts the group instead of hanging; the work fails with an error. Wait on your `Work` handles before destroying.
- **Waiting.** `Work.wait(timeout)` raises on expiry but does not cancel the operation.
- **Interpreter shutdown** with a live group does not hang (the completion thread never destroys tensors; see [work lifetime](../concepts/work-lifetime.md)).
- **Mixed-version ranks** cannot connect: `protocol_mismatch ... wire protocol` is raised at bootstrap, not later.

Peer-failure behavior is validated on loopback; do not kill a physical peer to test it.

## When to ask for help

Escalate when the checks isolate the layer and it is torch-tbccl or TBCCL, or when you cannot isolate it. Include `python -m torch_tbccl.info` from every machine, the exact error, `python tools/collect_diagnostics.py` output (versions and interface names only; nothing is uploaded), and the commands you ran. See [getting help](https://tbccl.tensorsofthewall.com/en/latest/project/getting-help.html). Security problems follow the [security policy](../project/security-policy.md).
