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
| Python env | `.venv` (uv, CPython 3.13, torch 2.14.1+cu130) | **not set up yet** (system Python 3.9.6 only, no uv, no torch) |
| Installed TBCCL prefix | `TBCCL_ROOT` (see `.local/phase42_notes.md`) | **not built yet** |

Non-interactive SSH to the Mac uses zsh with a minimal PATH: use
`export PATH=/opt/homebrew/bin:$PATH` first (cmake, ctest, brew tools), and
there is no GNU `timeout`.

Thunderbolt 4 link: Linux `thunderbolt0` = `192.168.3.2`, Mac `bridge0` =
`192.168.3.1`, MTU 9000, healthy RTT ~0.3-0.5 ms. SSH uses the normal
network, not the TB4 cable.

## Mac setup checklist (needs user approval per step; none done yet)

1. Install `uv` and a modern CPython (>= 3.10; Linux uses 3.13) and a
   macOS arm64 PyTorch build in a Mac-side `.venv` (git-ignored).
2. Bring `../tbccl` on the Mac to the pinned revision (see
   `.local/phase42_notes.md`; needs the PIC target property) via git pull
   after the user approves the push, build **host-only**
   (`-DCMAKE_BUILD_TYPE=Release`, no CUDA/Metal) and `cmake --install` to a
   prefix; run the TBCCL test suite there.
3. Clone state of `~/projects/torch-tbccl`: empty `main` tracking
   `origin` (https://github.com/tensorsofthewall/torch-tbccl); nothing is
   pushed from Linux yet, so code arrives only after a user-approved push
   (or per-file `scp`).
4. `TBCCL_ROOT=<prefix> uv pip install -e . --no-build-isolation`, then
   `pytest` (CPU loopback tests run on the Mac alone).

## Real two-host runs

- PyTorch rendezvous (the c10d Store) and TBCCL tensor traffic are
  separate: `MASTER_ADDR`/`MASTER_PORT` for the Store, and each rank's own
  `TBCCL_LOCAL_ENDPOINT=<tb4 ip>:<port>` (Linux `192.168.3.2:PORT`, Mac
  `192.168.3.1:PORT`). TBCCL's data connection uses the rank-0 endpoint
  port + 1000, so keep that port free too.
- Run both orientations (Linux rank 0 / Mac rank 1, then reversed).
- Before and after each real-TB4 group: link ping, AER snapshot
  (`scripts/tb4_health_snapshot.py` in `../tbccl`), GPU thermal state; stop
  on any condition listed in `../tbccl/AGENTS.md`. Keep real-link testing
  focused (the staged 4 KiB / 1 MiB / 16 MiB plan), do exhaustive sweeps on
  loopback.
- A Python process is the TCP listener on the Mac, so the macOS Application
  Firewall may prompt or block it. If that happens, STOP and ask the user;
  never change firewall settings automatically.

## Syncing

Same rules as TBCCL: develop and test on Linux, then sync only after
explicit approval; check `git status` on the Mac before pulling (directly
`scp`'d files leave conflicting local modifications); avoid multi-source
`rsync` into one directory and never `git stash drop` an untracked-inclusive
stash. See the TBCCL doc for details.
