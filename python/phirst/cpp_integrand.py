"""
phirst.cpp_integrand — load a user-compiled C++ integrand shared library.

The library must export phirst_run_integrand() with the C ABI defined by
phirst_add_integrand_module() (cmake/PhirstIntegrand.cmake).
"""

import ctypes
import os
from pathlib import Path


class CppIntegrandModule:
    """
    A user-compiled C++ integrand loaded from a shared library (.so / .dll).

    Attributes
    ----------
    n_particles : int
        The compile-time particle count baked into the library.
    path : str
        Absolute path to the shared library.
    """

    def __init__(self, path: str, n_particles: int) -> None:
        self.path = str(Path(path).resolve())
        self.n_particles = n_particles
        self._lib = _load_lib(self.path)

    def __repr__(self) -> str:
        return f"CppIntegrandModule(path={self.path!r}, n_particles={self.n_particles})"


def _load_lib(path: str) -> ctypes.CDLL:
    """Load the shared library and configure the C ABI."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"phirst.load_integrand_module: shared library not found: {path}\n"
            "Build it with phirst_add_integrand_module() in CMake."
        )
    lib = ctypes.CDLL(path)

    try:
        fn = lib.phirst_run_integrand
    except AttributeError as exc:
        raise ImportError(
            f"phirst.load_integrand_module: {path} does not export "
            "'phirst_run_integrand'. Was it built with phirst_add_integrand_module()?"
        ) from exc

    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    return lib


def run_cpp_integrand(module: CppIntegrandModule, *,
                      n_particles: int,
                      cm_energy: float,
                      masses,
                      n_events: int,
                      seed: int,
                      use_vegas: bool):
    """Call phirst_run_integrand() in the loaded shared library."""
    import numpy as np
    from . import IntegrationResult

    _ = IntegrationResult

    if n_particles != module.n_particles:
        raise ValueError(
            f"n_particles={n_particles} does not match the compiled value "
            f"n_particles={module.n_particles} in {module.path}"
        )

    masses_arr = np.asarray(masses, dtype=np.float64)
    masses_ptr = masses_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    mean_out = ctypes.c_double(0.0)
    error_out = ctypes.c_double(0.0)

    rc = module._lib.phirst_run_integrand(
        ctypes.c_double(cm_energy),
        masses_ptr,
        ctypes.c_int(n_particles),
        ctypes.c_int64(n_events),
        ctypes.c_uint64(seed),
        ctypes.c_int(1 if use_vegas else 0),
        ctypes.byref(mean_out),
        ctypes.byref(error_out),
    )

    if rc != 0:
        raise RuntimeError(
            f"phirst_run_integrand() returned error code {rc} "
            f"(library: {module.path})"
        )

    return _CppResult(mean_out.value, error_out.value, n_events)


class _CppResult:
    """Lightweight result object matching IntegrationResult's public interface."""

    __slots__ = ("mean", "error", "n_events")

    def __init__(self, mean: float, error: float, n_events: int) -> None:
        self.mean = mean
        self.error = error
        self.n_events = n_events

    def __repr__(self) -> str:
        return (
            f"IntegrationResult(mean={self.mean}, error={self.error}, "
            f"n_events={self.n_events})"
        )
