# Quickstart

A two-rank all_reduce on one machine, then the same across two hosts. Install first ([installing](install.md)).

## Two ranks on one machine

`allreduce.py`:

```python
import torch
import torch.distributed as dist
import torch_tbccl                      # registers the "tbccl" backend (required once per process)

dist.init_process_group("tbccl")

x = torch.tensor([dist.get_rank() + 1.0])
dist.all_reduce(x)                      # SUM
print(dist.get_rank(), dist.get_backend(), x)

dist.destroy_process_group()
```

```sh
OMP_NUM_THREADS=2 TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 .venv/bin/torchrun --standalone --nproc-per-node 2 allreduce.py
```

Each rank prints its rank, the backend name `tbccl` and `tensor([3.])`; the two lines may interleave. Set `OMP_NUM_THREADS` (for example to 2) when several ranks share one machine, or they starve each other.

Every rank needs `TBCCL_LOCAL_ENDPOINT=<host>:<port>`: the address where this rank listens for TBCCL connections. It is **separate** from PyTorch's rendezvous store. `<host>:0` picks free ports, which is what you normally want; many groups can coexist in one process. If it is unset in a group of two or more ranks, `init_process_group` raises `ValueError: torch-tbccl: invalid argument: TBCCL_LOCAL_ENDPOINT is not set`. If the machine's hostname does not resolve, add `--local-addr 127.0.0.1` to `torchrun` for single-machine runs.

## Across two hosts

This is the first run across two real machines. Before it, connect them and verify the link with [Two machines over Thunderbolt](https://tbccl.tensorsofthewall.com/en/latest/getting-started/two-machine-thunderbolt.html): both computers need an address in one private subnet, and ping **and** a TCP connection must work in both directions. The examples below use the addresses from that page, **Mac `192.168.77.1`** and **Linux `192.168.77.2`**; use your own.

### 1. Install on both machines

Follow [installing](install.md) on each machine: PyTorch first, then the `torch-tbccl` wheel from the GitHub release. The versions must match on both. On **Linux** the PyTorch build must be the CUDA 13 one even if you only use CPU tensors, and no separate TBCCL install is needed (the wheel carries the runtime). Check each machine:

```sh
.venv/bin/python -m torch_tbccl.info     # registered True; the same wire_protocol on both
```

Copy [`examples/two_host_allreduce.py`](https://github.com/tensorsofthewall/torch-tbccl/blob/main/examples/two_host_allreduce.py) to both machines (the same file; `curl -fsSLO` from the raw URL works).

### 2. What the pieces are

| Piece | What it is | In the commands below |
|---|---|---|
| **Ranks** | one process per machine here; rank 0 and rank 1 | `--node-rank 0` / `1` |
| **PyTorch rendezvous** | rank 0 hosts a small key-value store; every rank connects to it once at startup to learn who else is in the job | `--master-addr 192.168.77.1 --master-port 29500` (rank 0's link address and a free TCP port) |
| **TBCCL endpoint** | where *this* rank listens for TBCCL's own data connections; `:0` lets the system choose free ports | `TBCCL_LOCAL_ENDPOINT=<this machine's link address>:0` |
| **Local address** | the address `torchrun` itself uses for this node, so it does not depend on a hostname resolving | `--local-addr <this machine's link address>` |
| **Routing** | the operating system sends traffic to `192.168.77.x` out of the Thunderbolt interface because that subnet exists only there | confirmed in the link check (`ip route get` / `route -n get`) |

Every address in these commands is a link address, so the rendezvous and the data both use the cable. If you used a different address anywhere (for example your Wi-Fi address as `--master-addr`), the job would still work but the traffic would not cross Thunderbolt.

### 3. Run it

Start rank 0 first, then rank 1 within about a minute: `--rdzv-conf timeout=60` makes `torchrun` give up after 60 seconds if the other machine has not joined (without it `torchrun` waits 15 minutes), and the script's own `--timeout` (60 seconds by default) bounds the rendezvous store and the collective, so start both commands close together. Here the Mac is rank 0. (The Mac as rank 0 is a choice; the Linux machine can be rank 0 if you swap the addresses and rank numbers consistently.)

```sh
# Mac (rank 0), in a live Terminal or SSH session
TBCCL_LOCAL_ENDPOINT=192.168.77.1:0 .venv/bin/torchrun --nnodes 2 --nproc-per-node 1 --node-rank 0 \
    --local-addr 192.168.77.1 --master-addr 192.168.77.1 --master-port 29500 --rdzv-conf timeout=60 two_host_allreduce.py
```

```sh
# Linux (rank 1)
TBCCL_LOCAL_ENDPOINT=192.168.77.2:0 .venv/bin/torchrun --nnodes 2 --nproc-per-node 1 --node-rank 1 \
    --local-addr 192.168.77.2 --master-addr 192.168.77.1 --master-port 29500 --rdzv-conf timeout=60 two_host_allreduce.py
```

Expected output (each machine prints its own rank's lines; the hostname and versions differ):

```text
rank 0: host=<hostname> world=2 backend=tbccl torch_tbccl=0.2.0 tbccl_runtime=0.6.0
rank 0: backend=tbccl result=3.0 PASS
```

and `rank 1: ... result=3.0 PASS` on the other machine. Both ranks contribute `rank + 1`, so `1 + 2 = 3`; the script exits 0 only when the backend is `tbccl` and every element is 3. It gives up after 60 seconds (`--timeout`).

On the Mac, the first run may show a macOS prompt about local network access or incoming connections for Python: allow it. A background job without a live session cannot reach the network (see [troubleshooting](../guides/troubleshooting.md)).

If it hangs or fails, run `python3 thunderbolt_link_doctor.py --peer <the other machine's link address>` (from the [link troubleshooting page](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html)) on both machines before changing anything. A common cause is `torchrun` resolving the machine's hostname to a different address; `--local-addr` above prevents it.

### 4. Confirm the traffic crossed the cable

Read the link's byte counters before and after a run with a larger tensor (`--elements 1048576` is 4 MiB per rank), on both machines:

```sh
ip -s link show thunderbolt0      # Linux: RX and TX bytes
netstat -ib -I bridge0            # macOS: Ibytes and Obytes (the Thunderbolt Bridge device name from your Mac)
```

The counters of the Thunderbolt interface should grow by megabytes during the run, not kilobytes; counters of Wi-Fi or Ethernet should not. This is the check that the collective used the cable, as opposed to some other network. (This is the expected behaviour, not an observed result: this counter check has not yet been run on a real link, and the two-host example itself has so far run only on loopback.) To confirm that TBCCL, not another backend, handled the operations, see [how do I know TBCCL is being used?](../guides/verify-active.md).

### 5. Clean up

Both processes exit on their own; no daemon is left behind and the Thunderbolt link needs no teardown. If a rank hangs, press Ctrl-C in its own terminal (do not kill processes by name; other work may be running). Check that nothing is still using the rendezvous port, `lsof -iTCP:29500 -sTCP:LISTEN` on the rank-0 machine, and that the TBCCL ports are released. Leave the network settings as they are.

### 6. GPU and MPS tensors

Only after the CPU run passes, try the device: `two_host_allreduce.py --device cuda` on Linux (needs a CUDA GPU) and `--device mps` on the Mac, using the same commands. Mixed CUDA-and-MPS groups need no change.

## Devices

Put tensors on `cpu`, `cuda` (Linux) or `mps` (macOS) as usual. A group may mix them, for example a CUDA rank with an MPS rank. See [supported operations](../reference/supported-operations.md) and the [MPS guide](../guides/mps.md).

## DDP

`examples/ddp_mlp.py` is a small distributed-data-parallel training run on synthetic data, checked against a single-process reference:

```sh
TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 .venv/bin/torchrun --standalone --nproc-per-node 2 examples/ddp_mlp.py
```

See the [DDP guide](../guides/ddp.md). To confirm a job really uses TBCCL see [how do I know TBCCL is being used?](../guides/verify-active.md).

## Next steps

Once the two-host all-reduce passes on CPU tensors, the same link can carry the other integrations; each has its own prerequisites and none is needed for the all-reduce:

- **vLLM** pipeline across a CUDA node and a Mac: the [vllm-tbccl quickstart](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/quickstart.html) (needs a local copy of a small model on both machines).
- **Exo** across a Mac and a Linux CUDA host: the [exo-tbccl two-host guide](https://exo-tbccl.tensorsofthewall.com/en/latest/guides/two-host.html). A passing all-reduce does not prove Exo discovery works: Exo also needs IPv6 link-local multicast on the link.
