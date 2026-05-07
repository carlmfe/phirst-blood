"""Phirst Blood: Monte Carlo phase space integration for high-energy physics."""

import numpy as np

from . import _phirst as _C
from ._phirst import (
    ConstantIntegrand,
    DrellYanIntegrand,
    EggholderIntegrand,
    IntegrationResult,
    MandelstamSIntegrand2,
    MandelstamSIntegrand3,
    MandelstamSIntegrand4,
    MandelstamSIntegrand5,
    MandelstamSIntegrand6,
)
from ._version import __version__
from .cpp_integrand import CppIntegrandModule

__all__ = [
    "integrand",
    "integrate",
    "generate_phase_space",
    "__version__",
    "IntegrationResult",
    "ConstantIntegrand",
    "DrellYanIntegrand",
    "EggholderIntegrand",
    "MandelstamSIntegrand2",
    "MandelstamSIntegrand3",
    "MandelstamSIntegrand4",
    "MandelstamSIntegrand5",
    "MandelstamSIntegrand6",
    "CppIntegrandModule",
    "load_integrand_module",
]

_SUPPORTED_N = [2, 3, 4, 5, 6]
_MANDELSTAM_N = {
    MandelstamSIntegrand2: 2,
    MandelstamSIntegrand3: 3,
    MandelstamSIntegrand4: 4,
    MandelstamSIntegrand5: 5,
    MandelstamSIntegrand6: 6,
}


def integrand(n_particles):
    """Decorator: compile a Numba device function as a PHIRST GPU integrand."""
    if n_particles not in _SUPPORTED_N:
        raise ValueError(f"n_particles must be one of {_SUPPORTED_N}, got {n_particles}")

    try:
        from .numba_integrand import _make_integrand_decorator

        return _make_integrand_decorator(n_particles)
    except ImportError as e:
        raise ImportError(
            "phirst.integrand requires numba, a CUDA-capable GPU, and "
            "libphirst_numba_bridge.so. Install numba and build phirst "
            "with PHIRST_NUMBA_BRIDGE=ON."
        ) from e


def load_integrand_module(path, *, n_particles):
    """
    Load a user-compiled C++ integrand shared library.

    Parameters
    ----------
    path : str or Path
        Path to the shared library built with phirst_add_integrand_module().
    n_particles : int
        The compile-time particle count the library was built with.

    Returns
    -------
    CppIntegrandModule
        An integrand object usable with phirst.integrate().

    Example
    -------
    >>> mod = phirst.load_integrand_module("libmy_integrand.so", n_particles=2)
    >>> result = phirst.integrate(mod, n_particles=2, cm_energy=91.2)
    """
    from .cpp_integrand import CppIntegrandModule
    if n_particles not in _SUPPORTED_N:
        raise ValueError(f"n_particles must be one of {_SUPPORTED_N}, got {n_particles}")
    return CppIntegrandModule(str(path), n_particles)


def generate_phase_space(*, n_particles, cm_energy, n_events=10_000,
                         masses=None, seed=5489):
    """Generate RAMBO phase-space points and weights.

    Returns
    -------
    momenta : np.ndarray
        Shape ``(n_events, n_particles, 4)`` with ``[E, px, py, pz]`` ordering.
    weights : np.ndarray
        Shape ``(n_events,)`` containing ``exp(log_weight)`` for each event.
    """
    if n_particles not in _SUPPORTED_N:
        raise ValueError(f"n_particles must be one of {_SUPPORTED_N}, got {n_particles}")

    if masses is None:
        masses_arr = np.zeros(n_particles, dtype=np.float64)
    else:
        masses_arr = np.asarray(masses, dtype=np.float64)
    if len(masses_arr) != n_particles:
        raise ValueError(f"masses must have length {n_particles}")

    return _C._generate_phase_space(
        int(n_particles), float(cm_energy), masses_arr, int(n_events), int(seed)
    )


def integrate(integrand, *, n_particles, cm_energy, n_events=100_000,
              masses=None, seed=5489, use_vegas=False):
    """Run Monte Carlo integration over RAMBO phase space."""
    if n_particles not in _SUPPORTED_N:
        raise ValueError(f"n_particles must be one of {_SUPPORTED_N}, got {n_particles}")

    if masses is None:
        masses_arr = np.zeros(n_particles, dtype=np.float64)
    else:
        masses_arr = np.asarray(masses, dtype=np.float64)
    if len(masses_arr) != n_particles:
        raise ValueError(f"masses must have length {n_particles}")

    if isinstance(integrand, ConstantIntegrand):
        return _C._integrate_constant(
            integrand, int(n_particles), float(cm_energy), masses_arr,
            int(n_events), int(seed), bool(use_vegas)
        )
    if isinstance(integrand, DrellYanIntegrand):
        if n_particles != 2:
            raise ValueError("DrellYanIntegrand requires n_particles=2")
        return _C._integrate_drell_yan(
            integrand, int(n_particles), float(cm_energy), masses_arr,
            int(n_events), int(seed), bool(use_vegas)
        )
    if isinstance(integrand, EggholderIntegrand):
        if n_particles < 3:
            raise ValueError("EggholderIntegrand requires n_particles >= 3")
        return _C._integrate_eggholder(
            integrand, int(n_particles), float(cm_energy), masses_arr,
            int(n_events), int(seed), bool(use_vegas)
        )
    if type(integrand) in _MANDELSTAM_N:
        expected_n = _MANDELSTAM_N[type(integrand)]
        if n_particles != expected_n:
            raise ValueError(
                f"MandelstamSIntegrand{expected_n} requires n_particles={expected_n}"
            )
        fn = getattr(_C, f"_integrate_mandelstam{expected_n}")
        return fn(
            integrand, float(cm_energy), masses_arr,
            int(n_events), int(seed), bool(use_vegas)
        )

    try:
        from . import numba_integrand as _numba_mod
        from .numba_integrand import PhirstNumbaIntegrand

        if isinstance(integrand, PhirstNumbaIntegrand):
            return _numba_mod._run_numba_integrand(
                integrand,
                n_particles=n_particles,
                cm_energy=cm_energy,
                masses=masses_arr,
                n_events=n_events,
                seed=seed,
            )
    except ImportError:
        pass

    try:
        from .cpp_integrand import CppIntegrandModule, run_cpp_integrand
        if isinstance(integrand, CppIntegrandModule):
            return run_cpp_integrand(
                integrand,
                n_particles=n_particles,
                cm_energy=cm_energy,
                masses=masses_arr,
                n_events=n_events,
                seed=seed,
                use_vegas=use_vegas,
            )
    except ImportError:
        pass

    if callable(integrand):
        return _C._integrate_callback(
            integrand, int(n_particles), float(cm_energy), masses_arr,
            int(n_events), int(seed)
        )

    raise TypeError(
        f"Unknown integrand type: {type(integrand)}. "
        "Use one of the built-in phirst integrand classes."
    )
