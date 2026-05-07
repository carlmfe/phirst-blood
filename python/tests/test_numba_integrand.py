"""Tests for the Numba GPU integrand path. Skipped if unavailable."""

import pytest

numba = pytest.importorskip("numba")
numba_cuda = pytest.importorskip("numba.cuda")

if not numba_cuda.is_available():
    pytest.skip("No CUDA GPU available", allow_module_level=True)

import phirst

try:
    from phirst.numba_integrand import _load_bridge

    _load_bridge()
except ImportError as exc:
    pytest.skip(str(exc), allow_module_level=True)


def test_numba_integrand_decorator():
    @phirst.integrand(n_particles=2)
    def my_fn(momenta_flat, n_particles):
        return momenta_flat[0] + momenta_flat[4]

    assert hasattr(my_fn, "n_particles")
    assert my_fn.n_particles == 2


def test_numba_integrand_ptx_extraction():
    @phirst.integrand(n_particles=2)
    def my_fn(momenta_flat, n_particles):
        return 1.0

    ptx = my_fn._get_ptx()
    assert isinstance(ptx, bytes)
    assert b"phirst_user_integrand" in ptx


def test_numba_integrate_constant():
    """A constant=1.0 integrand should match ConstantIntegrand result."""

    @phirst.integrand(n_particles=3)
    def const_fn(momenta_flat, n_particles):
        return 1.0

    result_numba = phirst.integrate(const_fn, n_particles=3, cm_energy=91.2, n_events=100_000)
    result_builtin = phirst.integrate(
        phirst.ConstantIntegrand(1.0), n_particles=3, cm_energy=91.2, n_events=100_000
    )
    assert abs(result_numba.mean - result_builtin.mean) < 5 * result_builtin.error
