"""torch.distributed over the N-rank TBCCL communicator: world_size 3 and 4, CPU, loopback, dynamically allocated ports (TBCCL_LOCAL_ENDPOINT host:0)."""
import pytest
import torch


@pytest.mark.parametrize("world", [3, 4])
def test_nrank_cpu(run_ranks, world):
    results = run_ranks("_worker_nrank.py", world)
    for rank, (rc, out) in enumerate(results):
        assert rc == 0, f"rank {rank}: {out}"
        assert out.strip().splitlines()[-1] == f"rank {rank} ok"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("world,cuda_rank", [(3, 0), (3, 2), (4, 0)])
def test_nrank_one_cuda_rank(run_ranks, world, cuda_rank):
    results = run_ranks("_worker_nrank.py", world, extra_env={"DEVICE_RANK": str(cuda_rank)})
    for rank, (rc, out) in enumerate(results):
        assert rc == 0, f"rank {rank}: {out}"


def test_nrank_init_repeated(run_ranks):
    # create/destroy repeatedly to catch teardown races and stale ports
    for _ in range(3):
        for rank, (rc, out) in enumerate(run_ranks("_worker_init_n.py", 4)):
            assert rc == 0, f"rank {rank}: {out}"


@pytest.mark.parametrize("world", [3, 4])
def test_nrank_ddp_smoke(run_ranks, world):
    # Stretch goal: every collective DDP needs already works at N>2, so one optimizer step runs unchanged.
    for rank, (rc, out) in enumerate(run_ranks("_worker_ddp_n.py", world, timeout=120)):
        assert rc == 0, f"rank {rank}: {out}"


@pytest.mark.parametrize("world", [3, 4])
def test_subgroups_of_the_world(run_ranks, world):
    # Phase 71: new_group over subsets of ranks (collectives, broadcast, all_gather, P2P addressed by global rank, barrier) next to the world group.
    for rank, (rc, out) in enumerate(run_ranks("_worker_subgroups.py", world, timeout=120)):
        assert rc == 0, f"rank {rank}: {out}"
        assert out.strip().splitlines()[-1] == f"rank {rank} ok"
