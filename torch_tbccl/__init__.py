"""torch-tbccl: out-of-tree PyTorch distributed backend over an installed TBCCL."""
import torch  # noqa: F401  (must load libtorch before the native module)
import torch.distributed as _dist

import os as _os
import re as _re

from . import _C
from ._version import SUPPORTED_C_ABI, TESTED_TORCH_SERIES, TESTED_TORCH_VERSION, TESTED_WIRE_PROTOCOL, __version__


def _series(version: str) -> str:
    m = _re.match(r"(\d+)\.(\d+)", version)
    return f"{m.group(1)}.{m.group(2)}" if m else version


def built_with_torch() -> str:
    """The torch version (from its headers) this extension was compiled against."""
    return _C.built_with_torch()


def _check_torch_abi() -> None:
    built, running = _series(_C.built_with_torch()), _series(torch.__version__)
    if built != running and _os.environ.get("TORCH_TBCCL_ALLOW_TORCH_MISMATCH") != "1":
        raise ImportError(
            f"torch-tbccl was compiled against torch {_C.built_with_torch()} but torch {torch.__version__} is installed. The extension "
            f"links against torch's C++ ABI, which is only stable within one minor series ({built} != {running}). Rebuild it in this environment: "
            "TBCCL_ROOT=<installed tbccl prefix> uv pip install --no-build-isolation --no-deps -e . "
            "(set TORCH_TBCCL_ALLOW_TORCH_MISMATCH=1 to skip this check at your own risk)"
        )


def _check_tbccl_abi() -> None:
    abi, wire = _C.c_abi_version(), _C.wire_protocol_version()
    if abi not in SUPPORTED_C_ABI:
        raise ImportError(
            f"torch-tbccl {__version__} supports libtbccl C ABI {list(SUPPORTED_C_ABI)} but this build was compiled against C ABI {abi} "
            f"(libtbccl {_C.runtime_version()}). Rebuild torch-tbccl against a compatible TBCCL prefix: TBCCL_ROOT=<prefix> pip install --no-build-isolation ."
        )
    if wire not in TESTED_WIRE_PROTOCOL:
        import warnings

        warnings.warn(
            f"torch-tbccl was built against libtbccl wire protocol {wire}; only {list(TESTED_WIRE_PROTOCOL)} has been tested. Ranks must all use the same wire protocol.",
            stacklevel=2,
        )


_check_torch_abi()
_check_tbccl_abi()

BACKEND_NAME = "tbccl"


def runtime_version() -> str:
    """Version of the TBCCL runtime linked into this build."""
    return _C.runtime_version()


def c_abi_version() -> int:
    """libtbccl C ABI version this build was compiled against."""
    return _C.c_abi_version()


def wire_protocol_version() -> int:
    """libtbccl Communicator wire-protocol version this build speaks (all ranks of a group must agree)."""
    return _C.wire_protocol_version()


def info() -> dict:
    """One dict describing this installation (versions of every layer, devices, where it was loaded from); `python -m torch_tbccl.info` prints it."""
    import platform

    return {
        "torch_tbccl": __version__,
        "location": _os.path.dirname(_os.path.abspath(__file__)),
        "torch_running": torch.__version__,
        "torch_built_with": built_with_torch(),
        "torch_tested": list(TESTED_TORCH_SERIES),
        "libtbccl": runtime_version(),
        "c_abi": c_abi_version(),
        "wire_protocol": wire_protocol_version(),
        "devices": supported_devices(),
        "cuda_runtime_torch": torch.version.cuda,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "backend": BACKEND_NAME,
        "registered": BACKEND_NAME in _dist.Backend.backend_list,
    }


def compiled_features() -> dict:
    """Devices this build can serve, e.g. {"cpu": True, "cuda": False, "mps": False}."""
    return dict(_C.compiled_features())


def supported_devices() -> list:
    return [d for d in ("cpu", "cuda") if compiled_features().get(d)]


def trace_enabled() -> bool:
    """True when per-collective timeline recording is on (TORCH_TBCCL_TRACE=1 or trace_set_enabled)."""
    return _C.trace_enabled()


def trace_set_enabled(on: bool) -> None:
    _C.trace_set_enabled(bool(on))


def trace_reset() -> None:
    """Drop recorded events. Call only with no collectives in flight."""
    _C.trace_reset()


def trace_now_ns() -> int:
    """The clock trace events are stamped with, for correlating application-side events."""
    return _C.trace_now_ns()


def trace_events() -> list:
    """Per-collective events (dicts) with this process's monotonic nanosecond stamps; never compare
    stamps across machines. complete_ns/wait_* are 0 until they happen."""
    return list(_C.trace_events())


def register_backend() -> None:
    """Register "tbccl" with torch.distributed. Idempotent."""
    if BACKEND_NAME in _dist.Backend.backend_list:
        return
    _dist.Backend.register_backend(BACKEND_NAME, _C.create_backend, devices=supported_devices())


register_backend()

__all__ = [
    "__version__",
    "SUPPORTED_C_ABI",
    "TESTED_WIRE_PROTOCOL",
    "TESTED_TORCH_VERSION",
    "TESTED_TORCH_SERIES",
    "built_with_torch",
    "BACKEND_NAME",
    "runtime_version",
    "c_abi_version",
    "wire_protocol_version",
    "info",
    "compiled_features",
    "supported_devices",
    "register_backend",
    "trace_enabled",
    "trace_set_enabled",
    "trace_reset",
    "trace_events",
    "trace_now_ns",
]
