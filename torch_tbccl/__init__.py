"""torch-tbccl: out-of-tree PyTorch distributed backend over an installed TBCCL."""
import torch  # noqa: F401  (must load libtorch before the native module)

from . import _C
from ._version import TESTED_TORCH_VERSION, __version__


def runtime_version() -> str:
    """Version of the TBCCL runtime linked into this build."""
    return _C.runtime_version()


def compiled_features() -> dict:
    """Devices this build can serve, e.g. {"cpu": True, "cuda": False, "mps": False}."""
    return dict(_C.compiled_features())


__all__ = ["__version__", "TESTED_TORCH_VERSION", "runtime_version", "compiled_features"]
