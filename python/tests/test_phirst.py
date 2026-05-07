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


def test_generate_phase_space_shape():
    momenta, weights = phirst.generate_phase_space(
        n_particles=3, cm_energy=91.2, n_events=1000, seed=5489
    )
    assert momenta.shape == (1000, 3, 4)
    assert weights.shape == (1000,)
    assert momenta.dtype == np.float64
    assert weights.dtype == np.float64


def test_generate_phase_space_momentum_conservation():
    momenta, weights = phirst.generate_phase_space(
        n_particles=4, cm_energy=200.0, n_events=500, seed=5489
    )
    total = momenta.sum(axis=1)
    np.testing.assert_allclose(total[:, 0], 200.0, atol=1e-8, err_msg="Energy not conserved")
    np.testing.assert_allclose(total[:, 1], 0.0, atol=1e-8, err_msg="px not conserved")
    np.testing.assert_allclose(total[:, 2], 0.0, atol=1e-8, err_msg="py not conserved")
    np.testing.assert_allclose(total[:, 3], 0.0, atol=1e-8, err_msg="pz not conserved")
    assert weights.shape == (500,)


def test_generate_phase_space_positive_energy():
    momenta, weights = phirst.generate_phase_space(
        n_particles=2, cm_energy=91.2, n_events=200, seed=42
    )
    assert np.all(momenta[:, :, 0] > 0)
    assert weights.shape == (200,)


def test_generate_phase_space_finite_weights():
    momenta, weights = phirst.generate_phase_space(
        n_particles=3, cm_energy=100.0, n_events=500, seed=5489
    )
    assert momenta.shape == (500, 3, 4)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)


def test_generate_phase_space_reproducible():
    m1, w1 = phirst.generate_phase_space(n_particles=2, cm_energy=91.2, n_events=100, seed=99)
    m2, w2 = phirst.generate_phase_space(n_particles=2, cm_energy=91.2, n_events=100, seed=99)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(w1, w2)


def test_generate_phase_space_different_seed():
    m1, w1 = phirst.generate_phase_space(n_particles=2, cm_energy=91.2, n_events=100, seed=1)
    m2, w2 = phirst.generate_phase_space(n_particles=2, cm_energy=91.2, n_events=100, seed=2)
    assert not np.array_equal(m1, m2)
    assert np.array_equal(w1, w2)


def test_integrate_callable_constant():
    def f(momenta):
        return np.ones(momenta.shape[0])

    result_py = phirst.integrate(f, n_particles=3, cm_energy=91.2, n_events=50_000, seed=5489)
    result_builtin = phirst.integrate(
        phirst.ConstantIntegrand(1.0), n_particles=3, cm_energy=91.2, n_events=50_000, seed=5489
    )
    assert np.isfinite(result_py.mean)
    assert result_py.error > 0
    diff = abs(result_py.mean - result_builtin.mean)
    assert diff < 5 * result_builtin.error



def test_integrate_callable_energy_sum():
    def f(momenta):
        return momenta[:, :, 0].sum(axis=1)

    result = phirst.integrate(f, n_particles=3, cm_energy=91.2, n_events=10_000, seed=5489)
    assert np.isfinite(result.mean)
    assert result.error > 0



def test_integrate_callable_wrong_shape_raises():
    def bad_f(momenta):
        return np.ones((momenta.shape[0], 2))

    with pytest.raises(Exception):
        phirst.integrate(bad_f, n_particles=2, cm_energy=91.2, n_events=100, seed=5489)



def test_integrate_callable_reproducible():
    def f(momenta):
        return momenta[:, :, 0].sum(axis=1)

    r1 = phirst.integrate(f, n_particles=3, cm_energy=91.2, n_events=1000, seed=7)
    r2 = phirst.integrate(f, n_particles=3, cm_energy=91.2, n_events=1000, seed=7)
    assert r1.mean == r2.mean
