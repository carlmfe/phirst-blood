"""Tests for the C++ extension integrand path (Phase 5)."""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SO_PATH = os.path.join(_REPO_ROOT, "examples", "custom_integrand", "build", "libmy_integrand.so")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_SO_PATH),
    reason=f"libmy_integrand.so not found at {_SO_PATH}",
)

import phirst


def test_load_integrand_module_returns_object():
    mod = phirst.load_integrand_module(_SO_PATH, n_particles=2)
    assert mod.n_particles == 2
    assert os.path.exists(mod.path)


def test_load_integrand_module_wrong_path_raises():
    with pytest.raises(FileNotFoundError):
        phirst.load_integrand_module("/nonexistent/libfoo.so", n_particles=2)


def test_cpp_integrand_finite_result():
    mod = phirst.load_integrand_module(_SO_PATH, n_particles=2)
    result = phirst.integrate(mod, n_particles=2, cm_energy=91.2, n_events=50_000)
    assert np.isfinite(result.mean)
    assert result.error > 0
    assert result.n_events == 50_000


def test_cpp_integrand_reproducible():
    mod = phirst.load_integrand_module(_SO_PATH, n_particles=2)
    r1 = phirst.integrate(mod, n_particles=2, cm_energy=91.2, n_events=10_000, seed=42)
    r2 = phirst.integrate(mod, n_particles=2, cm_energy=91.2, n_events=10_000, seed=42)
    assert r1.mean == r2.mean
    assert r1.error == r2.error


def test_cpp_integrand_n_particles_mismatch_raises():
    mod = phirst.load_integrand_module(_SO_PATH, n_particles=2)
    with pytest.raises(ValueError, match="n_particles"):
        phirst.integrate(mod, n_particles=3, cm_energy=91.2, n_events=1_000)


def test_cpp_integrand_consistent_with_builtin():
    """MyIntegrand (Mandelstam s) should agree with MandelstamSIntegrand2."""
    mod = phirst.load_integrand_module(_SO_PATH, n_particles=2)
    r_cpp = phirst.integrate(mod, n_particles=2, cm_energy=91.2, n_events=100_000, seed=5489)
    r_builtin = phirst.integrate(
        phirst.MandelstamSIntegrand2(scale=1.0),
        n_particles=2,
        cm_energy=91.2,
        n_events=100_000,
        seed=5489,
    )
    diff = abs(r_cpp.mean - r_builtin.mean)
    sigma = max(r_cpp.error, r_builtin.error)
    assert diff < 5 * sigma, (
        f"C++ result {r_cpp.mean:.4e} ± {r_cpp.error:.4e} disagrees with "
        f"built-in {r_builtin.mean:.4e} ± {r_builtin.error:.4e}"
    )
