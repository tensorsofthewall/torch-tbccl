import importlib

import torch.distributed as dist

import torch_tbccl


def test_backend_recognized():
    assert "tbccl" in dist.Backend.backend_list
    assert dist.Backend.backend_capability["tbccl"] == torch_tbccl.supported_devices()
    assert dist.Backend("tbccl") == "tbccl"


def test_registration_idempotent():
    importlib.reload(torch_tbccl)
    torch_tbccl.register_backend()
    torch_tbccl.register_backend()
    assert dist.Backend.backend_list.count("tbccl") == 1
