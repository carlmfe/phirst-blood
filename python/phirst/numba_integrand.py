"""phirst.numba_integrand — GPU integration via Numba + cuLink.

Only available when:
  - numba is installed
  - libphirst_numba_bridge.so is present (built with PHIRST_NUMBA_BRIDGE=ON)
  - A CUDA GPU is available
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import _phirst as _C

_BRIDGE_ENV_VAR = "PHIRST_BRIDGE_LIB"
_BRIDGE_FILENAME = "libphirst_numba_bridge.so"
_PTX_FUNCTION_RE = re.compile(
    r"(?m)^(\s*(?:\.visible\s+)?\.func\s+)(\([^)]*\)\s+)?([^\s(]+)(\s*\()"
)


@dataclass(frozen=True)
class _IntegrationResultProxy:
    mean: float
    error: float
    n_events: int

    def __repr__(self) -> str:
        return (
            f"IntegrationResult(mean={self.mean}, error={self.error}, "
            f"n_events={self.n_events})"
        )


class PhirstNumbaIntegrand:
    """Wrapper around a Numba CUDA device function and its cached PTX."""

    def __init__(self, device_function: Any, n_particles: int, python_function: Any):
        self.device_function = device_function
        self.n_particles = int(n_particles)
        self.python_function = python_function
        self.__name__ = getattr(python_function, "__name__", type(self).__name__)
        self.__doc__ = getattr(python_function, "__doc__", None)
        self.__wrapped__ = python_function
        self._cached_ptx: Optional[bytes] = None

    def __repr__(self) -> str:
        return f"PhirstNumbaIntegrand(name={self.__name__!r}, n_particles={self.n_particles})"

    @property
    def _ptx(self) -> bytes:
        return self._get_ptx()

    def _get_ptx(self) -> bytes:
        if self._cached_ptx is None:
            self._cached_ptx = _extract_ptx(self.device_function, self.__name__)
        return self._cached_ptx


@functools.lru_cache(maxsize=1)
def _load_bridge():
    """Load libphirst_numba_bridge.so and set up the C ABI."""
    for candidate in _bridge_candidates():
        if not candidate.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(candidate))
        except OSError as exc:
            raise ImportError(
                f"Failed to load PHIRST Numba bridge library at {candidate}."
            ) from exc

        lib.phirst_link_and_launch.restype = ctypes.c_int
        lib.phirst_link_and_launch.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        return lib

    searched = ", ".join(str(path) for path in _bridge_candidates())
    raise ImportError(
        "Could not find libphirst_numba_bridge.so. Set PHIRST_BRIDGE_LIB or "
        "build phirst with PHIRST_NUMBA_BRIDGE=ON. Searched: "
        f"{searched}"
    )


def _make_integrand_decorator(n_particles: int):
    _require_numba_cuda()

    if int(n_particles) != n_particles or int(n_particles) <= 0:
        raise ValueError("n_particles must be a positive integer")
    n_particles = int(n_particles)

    def decorator(fn):
        _, cuda = _require_numba_cuda()
        device_fn = cuda.jit(device=True)(fn)
        return PhirstNumbaIntegrand(device_fn, n_particles, fn)

    return decorator


def _run_numba_integrand(
    integrand: PhirstNumbaIntegrand,
    *,
    n_particles: int,
    cm_energy: float,
    masses,
    n_events: int,
    seed: int,
):
    if integrand.n_particles != int(n_particles):
        raise ValueError(
            f"Numba integrand was compiled for n_particles={integrand.n_particles}, "
            f"got {n_particles}"
        )

    _require_numba_cuda()
    lib = _load_bridge()
    ptx = integrand._ptx
    masses_arr = np.ascontiguousarray(np.asarray(masses, dtype=np.float64))
    mean = ctypes.c_double()
    error = ctypes.c_double()

    status = lib.phirst_link_and_launch(
        ctypes.c_char_p(ptx),
        ctypes.c_size_t(len(ptx)),
        ctypes.c_double(float(cm_energy)),
        masses_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n_particles)),
        ctypes.c_int64(int(n_events)),
        ctypes.c_uint64(int(seed)),
        ctypes.byref(mean),
        ctypes.byref(error),
    )
    if status != 0:
        raise RuntimeError(
            "phirst_link_and_launch failed. Ensure a CUDA GPU is available "
            "and libphirst_numba_bridge.so was built with PHIRST_NUMBA_BRIDGE=ON."
        )

    return _IntegrationResultProxy(
        mean=float(mean.value),
        error=float(error.value),
        n_events=int(n_events),
    )


def _bridge_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []

    env_path = os.environ.get(_BRIDGE_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    package_dir = Path(__file__).resolve().parent
    candidates.append(package_dir / _BRIDGE_FILENAME)
    candidates.append(Path(_C.__file__).resolve().parent / _BRIDGE_FILENAME)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return tuple(unique_candidates)


def _require_numba_cuda():
    try:
        import numba
        from numba import cuda
    except ImportError as exc:
        raise ImportError(
            "phirst.integrand requires numba with CUDA support installed."
        ) from exc

    if not cuda.is_available():
        raise ImportError("phirst.integrand requires a CUDA-capable GPU available to Numba.")

    return numba, cuda


def _extract_ptx(device_function: Any, python_name: str) -> bytes:
    numba, _ = _require_numba_cuda()
    from numba.core import types as nbtypes

    # Use CPointer(float64) so momenta_flat is a raw double* — compatible with the
    # Numba float64 ABI: hidden return-value pointer is param_0, momenta is param_1.
    signature = (nbtypes.CPointer(nbtypes.float64), nbtypes.int32)

    # For device=True functions (CUDADispatcher with device flag), use compile_device.
    # For regular kernels, use compile. Both may raise — suppress all exceptions.
    for method_name in ("compile_device", "compile"):
        compile_fn = getattr(device_function, method_name, None)
        if callable(compile_fn):
            try:
                compile_fn(signature)
                break
            except Exception:
                continue

    ptx_blob: Any = None

    # inspect_asm returns PTX for CUDA (as string or dict); prefer it over inspect_ptx
    # because Numba ≥ 0.65 device functions expose inspect_asm but not inspect_ptx.
    for method_name in ("inspect_asm", "inspect_ptx"):
        inspect_fn = getattr(device_function, method_name, None)
        if not callable(inspect_fn):
            continue
        for args in ((signature,), ()):
            try:
                ptx_blob = inspect_fn(*args)
                if ptx_blob:
                    break
            except Exception:
                continue
        if ptx_blob:
            break

    ptx_bytes = _normalise_ptx_blob(ptx_blob)
    if ptx_bytes is None:
        raise RuntimeError("Unable to extract PTX from the Numba device function.")
    return _rename_ptx_symbol(ptx_bytes, python_name)


def _normalise_ptx_blob(ptx_blob: Any) -> Optional[bytes]:
    if ptx_blob is None:
        return None
    if isinstance(ptx_blob, bytes):
        return ptx_blob
    if isinstance(ptx_blob, str):
        return ptx_blob.encode()
    if isinstance(ptx_blob, dict):
        for value in ptx_blob.values():
            normalised = _normalise_ptx_blob(value)
            if normalised is not None:
                return normalised
        return None
    if isinstance(ptx_blob, (list, tuple)):
        for value in ptx_blob:
            normalised = _normalise_ptx_blob(value)
            if normalised is not None:
                return normalised
        return None
    return None


def _rename_ptx_symbol(ptx_bytes: bytes, python_name: str) -> bytes:
    ptx_text = ptx_bytes.decode("utf-8")
    if "phirst_user_integrand" in ptx_text:
        return ptx_bytes

    # Match PTX .func declaration:  .visible .func (rettype) funcname (
    # Groups: (1) prefix+.func,  (2) optional return-type paren,
    #         (3) function name,  (4) opening arg-paren
    specific_re = re.compile(
        r"(?m)^(\s*(?:\.visible\s+)?\.func\s+)(\([^)]*\)\s+)?"
        + re.escape(python_name)
        + r"(\s*\()"
    )
    renamed_ptx, count = specific_re.subn(
        lambda m: f"{m.group(1)}{m.group(2) or ''}phirst_user_integrand{m.group(3)}",
        ptx_text,
        count=1,
    )
    if count == 1:
        return renamed_ptx.encode("utf-8")

    # Fallback: rename the first .func declaration we find
    renamed_ptx, count = _PTX_FUNCTION_RE.subn(
        lambda m: f"{m.group(1)}{m.group(2) or ''}phirst_user_integrand{m.group(4)}",
        ptx_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not rename the Numba PTX symbol to phirst_user_integrand.")
    return renamed_ptx.encode("utf-8")
