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


# Backend selection through the normal torch.distributed API, never a silent substitute.
import subprocess
import sys

import pytest


def python(code, **env):
    import os

    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ, **env))


def test_backend_is_unknown_until_the_package_is_imported():
    # Documented: one explicit `import torch_tbccl` registers the backend (there is no autoload entry point).
    p = python("import torch.distributed as dist\ntry:\n    dist.init_process_group('tbccl', rank=0, world_size=1, store=dist.HashStore())\nexcept Exception as e:\n    print('REFUSED', type(e).__name__, e)\n")
    assert "REFUSED" in p.stdout and "tbccl" in p.stdout.lower(), p.stdout + p.stderr  # PyTorch itself: AssertionError "Unknown backend type tbccl"


def test_missing_endpoint_fails_loudly_instead_of_falling_back():
    p = python(
        "import torch.distributed as dist, torch_tbccl, datetime\n"
        "try:\n    dist.init_process_group('tbccl', rank=0, world_size=2, store=dist.HashStore(), timeout=datetime.timedelta(seconds=5))\n"
        "except ValueError as e:\n    print('REFUSED', e)\n",
        TBCCL_LOCAL_ENDPOINT="",
    )
    assert "REFUSED" in p.stdout and "TBCCL_LOCAL_ENDPOINT" in p.stdout, p.stdout + p.stderr


@pytest.mark.parametrize("spec", ["tbccl", "cpu:tbccl"])
def test_selected_backend_is_the_tbccl_process_group(run_two_ranks, spec):
    for rc, out in run_two_ranks("_worker_registration.py", args=(spec,), extra_env={"AUTO_PORT": "1"}):
        assert rc == 0, out
