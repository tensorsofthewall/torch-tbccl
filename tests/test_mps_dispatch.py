"""The c10d MPS dispatch: the operators exist for the MPS key after `import torch_tbccl`, the shim is version gated,
and an MPS tensor reaches ProcessGroupTBCCL instead of failing in the dispatcher."""
import subprocess
import sys

import pytest
import torch

import torch_tbccl

pytestmark = pytest.mark.skipif(not torch_tbccl.compiled_features()["mps"], reason="the MPS shim is compiled on macOS only")
OPS = ["send", "recv_", "broadcast_", "allreduce_", "allgather_", "gather_", "barrier"]


def test_kernels_registered_for_mps_after_import():
    for op in OPS:
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"c10d::{op}", "MPS"), op


def test_unsupported_operators_still_have_no_mps_kernel():
    # the shim registers only what the backend supports; everything else keeps failing in the dispatcher
    for op in ("reduce_", "scatter_", "reduce_scatter_", "_allgather_base_", "alltoall_base_", "allreduce_coalesced_"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(f"c10d::{op}", "MPS"), op


def test_registration_is_idempotent_and_reports_state():
    assert torch_tbccl._C.register_mps_dispatch() == ""


def test_devices_reported():
    assert torch_tbccl.supported_devices() == ["cpu", "mps"]
    assert torch_tbccl.info()["devices"] == ["cpu", "mps"]
    assert torch_tbccl.compiled_features() == {"cpu": True, "cuda": False, "mps": True}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_without_the_shim_the_dispatcher_rejects_mps():
    # the same call in a process that never imports torch_tbccl: the failure described in docs/adr/0002-out-of-tree-mps-dispatch.md
    code = (
        "import torch, torch.distributed as d, os\n"
        "os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29755')\n"
        "d.init_process_group('gloo', rank=0, world_size=1)\n"
        "try:\n d.all_reduce(torch.ones(4, device='mps'))\n"
        "except NotImplementedError as e:\n print('REJECTED', 'c10d::allreduce_' in str(e))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout
    assert "REJECTED True" in out, out
