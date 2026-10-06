import datetime
import socket

import pytest
import torch.distributed as dist

import torch_tbccl
from conftest import free_ports

T = datetime.timedelta(seconds=2)
EP = "torch_tbccl/v3/g0/endpoint/"
ID = "torch_tbccl/v3/g0/communicator_id"
SOME_ID = b"00112233445566778899aabbccddeeff"


def create(store, rank, ws, timeout=T):
    return torch_tbccl._C.create_backend(store, rank, ws, timeout)


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", "127.0.0.1:%d" % free_ports(1)[0])


def test_two_rank_rendezvous(run_two_ranks):
    results = run_two_ranks("_worker_init.py")
    for rc, out in results:
        assert rc == 0, out
    assert [out.strip().splitlines()[-1] for _, out in results] == ["rank 0 ok", "rank 1 ok"]


def test_two_rank_rendezvous_repeated(run_two_ranks):
    # Repeat to catch teardown races / stale state.
    for _ in range(3):
        for rc, out in run_two_ranks("_worker_init.py"):
            assert rc == 0, out


@pytest.mark.parametrize("ws", [9, 16])
def test_world_size_limit(ws, endpoint):
    with pytest.raises(NotImplementedError, match="world_size 1 to 8"):
        create(dist.HashStore(), 0, ws)


def test_missing_endpoint(monkeypatch):
    monkeypatch.delenv("TBCCL_LOCAL_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="TBCCL_LOCAL_ENDPOINT is not set"):
        create(dist.HashStore(), 0, 2)


@pytest.mark.parametrize(
    "bad", ["", "nocolon", ":1234", "host:", "host:abc", "host:70000", "::1:5000", "h:-1"]
)
def test_malformed_endpoint(bad, monkeypatch):
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", bad)
    with pytest.raises(ValueError):
        create(dist.HashStore(), 0, 2)


def test_malformed_endpoint_does_not_publish(monkeypatch):
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", "bad")
    store = dist.HashStore()
    with pytest.raises(ValueError):
        create(store, 0, 2)
    assert not store.check([EP + "0"])


def test_peer_never_publishes_times_out_boundedly(endpoint):
    import time

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="bootstrap timeout"):
        create(dist.HashStore(), 0, 2, datetime.timedelta(milliseconds=500))
    assert time.monotonic() - t0 < 10


def test_conflicting_endpoints_rejected(monkeypatch):
    # A rank whose control and data endpoints coincide cannot be told apart: libtbccl validates the directory before any network work.
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", "127.0.0.1:0")
    store = dist.HashStore()
    store.set(ID, SOME_ID)
    store.set(EP + "0", b"127.0.0.1:5555,127.0.0.1:5555")
    with pytest.raises(RuntimeError, match="already used by another endpoint"):
        create(store, 1, 2)


def test_malformed_published_record_rejected(monkeypatch):
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", "127.0.0.1:0")
    store = dist.HashStore()
    store.set(ID, SOME_ID)
    store.set(EP + "0", b"not-an-endpoint-record")
    with pytest.raises(ValueError, match="malformed endpoint record"):
        create(store, 1, 2)


def test_unreachable_peer_fails_boundedly(endpoint):
    import time

    dead, dead_data = free_ports(2)  # nobody listens here
    store = dist.HashStore()
    store.set(ID, SOME_ID)
    store.set(EP + "0", f"127.0.0.1:{dead},127.0.0.1:{dead_data}".encode())
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="communicator failure"):
        create(store, 1, 2, datetime.timedelta(seconds=2))
    assert time.monotonic() - t0 < 15


def test_publishes_actual_ports_and_one_shared_id(monkeypatch):
    # host:0 asks for free ports; the records carry the ACTUAL control and data endpoints, never a port derived from another.
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", "127.0.0.1:0")
    store = dist.HashStore()
    with pytest.raises(RuntimeError, match="bootstrap timeout"):
        create(store, 0, 3, datetime.timedelta(milliseconds=300))
    control, data = store.get(EP + "0").decode().split(",")
    (ch, cp), (dh, dp) = control.rsplit(":", 1), data.rsplit(":", 1)
    assert ch == dh == "127.0.0.1" and int(cp) > 0 and int(dp) > 0 and int(cp) != int(dp)
    assert len(store.get(ID)) == 32


def test_single_rank_group_needs_no_endpoint(monkeypatch):
    monkeypatch.delenv("TBCCL_LOCAL_ENDPOINT", raising=False)
    pg = create(dist.HashStore(), 0, 1)
    assert pg is not None


@pytest.mark.parametrize("world", [2, 3])
def test_reinit_over_a_persistent_store_never_reads_stale_records(run_ranks, world):
    # Regression (the packaging and capability-audit work): with torchrun the store outlives the process group; v2's fixed keys made cycle 2
    # read cycle 1's communicator id.
    for rank, (rc, out) in enumerate(run_ranks("_worker_reinit.py", world, extra_env={"CYCLES": "5"})):
        assert rc == 0, f"rank {rank}: {out}"
        assert out.strip().endswith(f"rank {rank} ok"), out
