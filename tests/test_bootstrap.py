import datetime
import socket

import pytest
import torch.distributed as dist

import torch_tbccl
from conftest import free_ports

T = datetime.timedelta(seconds=2)
EP = "torch_tbccl/v1/endpoint/"


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


@pytest.mark.parametrize("ws", [3, 4])
def test_world_size_must_be_two(ws, endpoint):
    with pytest.raises(NotImplementedError, match="world_size=2 only"):
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


def test_duplicate_endpoints_rejected(monkeypatch):
    monkeypatch.setenv("TBCCL_LOCAL_ENDPOINT", "127.0.0.1:5555")
    store = dist.HashStore()
    store.set(EP + "0", b"127.0.0.1:5555")
    with pytest.raises(ValueError, match="same TBCCL endpoint"):
        create(store, 1, 2)


def test_unreachable_peer_fails_boundedly(endpoint):
    import time

    dead = free_ports(1)[0]  # nobody listens here
    store = dist.HashStore()
    store.set(EP + "0", f"127.0.0.1:{dead}".encode())
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="communicator failure"):
        create(store, 1, 2, datetime.timedelta(seconds=2))
    assert time.monotonic() - t0 < 15


def test_single_rank_group_needs_no_endpoint(monkeypatch):
    monkeypatch.delenv("TBCCL_LOCAL_ENDPOINT", raising=False)
    pg = create(dist.HashStore(), 0, 1)
    assert pg is not None
