"""torch-tbccl: out-of-tree PyTorch distributed backend over an installed TBCCL."""
import torch  # noqa: F401  (must load libtorch before the native module)
import torch.distributed as _dist

from . import _C
from ._version import TESTED_TORCH_VERSION, __version__

BACKEND_NAME = "tbccl"


def runtime_version() -> str:
    """Version of the TBCCL runtime linked into this build."""
    return _C.runtime_version()


def compiled_features() -> dict:
    """Devices this build can serve, e.g. {"cpu": True, "cuda": False, "mps": False}."""
    return dict(_C.compiled_features())


def supported_devices() -> list:
    return [d for d in ("cpu", "cuda") if compiled_features().get(d)]


def register_backend() -> None:
    """Register "tbccl" with torch.distributed. Idempotent."""
    if BACKEND_NAME in _dist.Backend.backend_list:
        return
    _dist.Backend.register_backend(BACKEND_NAME, _C.create_backend, devices=supported_devices())


register_backend()

__all__ = [
    "__version__",
    "TESTED_TORCH_VERSION",
    "BACKEND_NAME",
    "runtime_version",
    "compiled_features",
    "supported_devices",
    "register_backend",
]
