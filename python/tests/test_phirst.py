import numpy as np
import pytest

import phirst


def test_constant_integrand_finite():
    result = phirst.integrate(
        phirst.ConstantIntegrand(1.0),
        n_particles=3,
        cm_energy=100.0,
        n_events=10_000,
        seed=5489,
    )
    assert np.isfinite(result.mean)
    assert np.isfinite(result.error)
    assert result.error > 0
    assert result.n_events == 10_000


def test_drell_yan_n2():
    result = phirst.integrate(
        phirst.DrellYanIntegrand(charge=2 / 3, coupling=1 / 137),
        n_particles=2,
        cm_energy=91.2,
        n_events=10_000,
        seed=5489,
    )
    assert np.isfinite(result.mean)
    assert result.mean > 0


def test_eggholder_requires_n3():
    with pytest.raises(ValueError):
        phirst.integrate(phirst.EggholderIntegrand(), n_particles=2, cm_energy=100.0)


def test_reproducibility():
    kwargs = dict(n_particles=2, cm_energy=91.2, n_events=1000, seed=42)
    r1 = phirst.integrate(phirst.ConstantIntegrand(1.0), **kwargs)
    r2 = phirst.integrate(phirst.ConstantIntegrand(1.0), **kwargs)
    assert r1.mean == r2.mean


def test_different_seeds_differ():
    r1 = phirst.integrate(
        phirst.EggholderIntegrand(),
        n_particles=3,
        cm_energy=100.0,
        n_events=1000,
        seed=1,
    )
    r2 = phirst.integrate(
        phirst.EggholderIntegrand(),
        n_particles=3,
        cm_energy=100.0,
        n_events=1000,
        seed=2,
    )
    assert r1.mean != r2.mean


def test_mandelstam_n3():
    result = phirst.integrate(
        phirst.MandelstamSIntegrand3(scale=1.0),
        n_particles=3,
        cm_energy=100.0,
        n_events=10_000,
        seed=5489,
    )
    assert np.isfinite(result.mean)
    assert result.mean > 0


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_constant_all_n(n):
    result = phirst.integrate(
        phirst.ConstantIntegrand(1.0),
        n_particles=n,
        cm_energy=100.0,
        n_events=5000,
        seed=5489,
    )
    assert np.isfinite(result.mean)
    assert result.mean > 0
