# Mac access and Thunderbolt 4 link reference (torch-tbccl)

torch-tbccl is developed on the same Linux/Mac pair as TBCCL. The machine
topology, link health checks, AER/GPU safety rules and firewall notes are
documented in full in TBCCL's `docs/mac_thunderbolt_access.md` (sibling repo
`../tbccl`); **read that first and treat it as authoritative**. This file
records only what is specific to torch-tbccl and links back for the rest.

## Machines

| | Linux | Mac |
|---|---|---|
| Role | CUDA tensors (rank 0 or 1) | CPU tensors (rank 1 or 0) |
| torch-tbccl repo | sibling of `tbccl` under the projects directory | `~/projects/torch-tbccl` |
| TBCCL repo | `../tbccl` | `~/projects/tbccl` |
| Login user | `svb` | `ragnarok` |
| SSH | — | `ssh tbccl-mac` |
| Python env | `.venv` (uv, CPython 3.13, torch 2.13.0 since Phase 50, matching vLLM) | `.venv` (uv 0.12 via `pip --user`, CPython 3.13.15, torch 2.13.0 since Phase 50) |
| Installed TBCCL prefix | `TBCCL_ROOT` (see `.local/phase42_notes.md`) | `~/projects/tbccl-install` (host-only, built in `~/projects/tbccl-build-install`) |

Non-interactive SSH to the Mac uses zsh with a minimal PATH: use
`export PATH=/opt/homebrew/bin:$PATH` first (cmake, ctest, brew tools), and
there is no GNU `timeout`.

Thunderbolt 4 link: Linux `thunderbolt0` = `192.168.3.2`, Mac `bridge0` =
`192.168.3.1`, MTU 9000, healthy RTT ~0.3-0.5 ms. SSH uses the normal
network, not the TB4 cable.

## Mac setup (done in Phase 42)
`uv` was installed with `python3 -m pip install --user uv` (then `export PATH=$HOME/Library/Python/3.9/bin:$PATH`),
the venv created with `uv venv --python 3.13 .venv`, torch installed from PyPI, TBCCL built host-only out of
tree and installed to `~/projects/tbccl-install`, and torch-tbccl installed with
`TBCCL_ROOT=$HOME/projects/tbccl-install uv pip install --python .venv/bin/python -e . --no-build-isolation`.
Code reaches the Mac only via `git pull` after a user-approved push.

## Real two-host runs

- PyTorch rendezvous (the c10d Store) and TBCCL tensor traffic are
  separate: `MASTER_ADDR`/`MASTER_PORT` for the Store, and each rank's own
  `TBCCL_LOCAL_ENDPOINT=<tb4 ip>:<port>` (Linux `192.168.3.2:PORT`, Mac
  `192.168.3.1:PORT`). TBCCL's data connection uses the rank-0 endpoint
  port + 1000, so keep that port free too.
- Run both orientations (Linux rank 0 / Mac rank 1, then reversed). `MASTER_ADDR` must be the **rank 0**
  host's address, since rank 0 hosts the c10d Store.
- Before and after each real-TB4 group: link ping, AER snapshot
  (`scripts/tb4_health_snapshot.py` in `../tbccl`), GPU thermal state; stop
  on any condition listed in `../tbccl/AGENTS.md`. Keep real-link testing
  focused (the staged 4 KiB / 1 MiB / 16 MiB plan), do exhaustive sweeps on
  loopback.
- Confirmed in Phase 42: the first time the Mac's venv Python (a uv-managed CPython) listens for inbound
  connections, macOS shows a firewall prompt in the GUI session and the run **hangs silently until someone
  clicks Allow** (init timeouts do not fire). After that the app is permitted and runs normally. Do not try to
  work around it; ask the user to click Allow.
- A Python process is the TCP listener on the Mac, so the macOS Application
  Firewall may prompt or block it. If that happens, STOP and ask the user;
  never change firewall settings automatically.

## Syncing

Same rules as TBCCL: develop and test on Linux, then sync only after
explicit approval; check `git status` on the Mac before pulling (directly
`scp`'d files leave conflicting local modifications); avoid multi-source
`rsync` into one directory and never `git stash drop` an untracked-inclusive
stash. See the TBCCL doc for details.
