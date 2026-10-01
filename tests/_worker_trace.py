"""One rank: TORCH_TBCCL_TRACE records a coherent per-collective timeline."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_tbccl

want = os.environ.get("TORCH_TBCCL_TRACE") == "1"
assert torch_tbccl.trace_enabled() == want
dist.init_process_group("tbccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
torch_tbccl.trace_reset()

x = torch.ones(1000)
w = dist.all_reduce(x, async_op=True)
w.wait()
b = torch.full((10,), 3, dtype=torch.int64)
dist.broadcast(b, src=0)
outs = [torch.zeros(10, dtype=torch.int64) for _ in range(2)]
dist.all_gather(outs, b)

ev = torch_tbccl.trace_events()
if not want:
    assert ev == [], ev
else:
    assert [e["op"] for e in ev] == ["allreduce", "broadcast", "allgather"], ev
    assert [e["seq"] for e in ev] == [0, 1, 2]
    assert [e["bytes"] for e in ev] == [4000, 80, 80]
    for e in ev:
        assert 0 < e["entry_ns"] <= e["before_submit_ns"] <= e["return_ns"] <= e["complete_ns"], e
        assert e["wait_entry_ns"] > 0 and e["wait_exit_ns"] >= e["wait_entry_ns"], e
        assert not e["error"] and e["device"] == "cpu"
    assert ev[0]["return_ns"] <= ev[1]["entry_ns"]
print(f"rank {rank} ok")
dist.destroy_process_group()
