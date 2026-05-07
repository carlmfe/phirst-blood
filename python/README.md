# Phirst Blood Python bindings

Phirst Blood provides Python access to the Phirst Monte Carlo phase-space generator and integrators for high-energy physics. The bindings expose a small NumPy-friendly API for generating RAMBO phase-space points, integrating built-in matrix elements in compiled code, and prototyping custom observables with vectorised Python callables.

## Installation

### Requirements

- Python >= 3.9
- NumPy >= 1.21
- CMake >= 3.18
- C++17 compiler
- `scikit-build-core`
- `nanobind`
- `numba` (optional, for `@phirst.integrand` GPU device integrands)

### Default development install

```bash
pip install scikit-build-core nanobind
pip install -e . --no-build-isolation
```

This builds the Python extension in editable mode and installs the `phirst` package from the repository.

### Build configuration

#### Enabling the Numba GPU bridge

To run a Python device function entirely on the GPU with `@phirst.integrand`, install `numba`
and build the extra bridge library:

```bash
pip install numba
pip install -e . --no-build-isolation \
  -Ccmake.args="-DPHIRST_BACKEND=SERIAL;-DPHIRST_NUMBA_BRIDGE=ON"
```

This installs `libphirst_numba_bridge.so` alongside the Python package. You can also point
`PHIRST_BRIDGE_LIB` at a custom bridge location.


#### Selecting a backend

The Python package is built through scikit-build-core, so backend selection is passed through CMake arguments:

```bash
pip install -e . --no-build-isolation \
  -Ccmake.args="-DPHIRST_BACKEND=CUDA;-DPHIRST_GPU_ARCH=89"
```

| Backend | CMake args | Notes |
|---|---|---|
| SERIAL | `-DPHIRST_BACKEND=SERIAL` | Default CPU build; easiest for development |
| CUDA | `-DPHIRST_BACKEND=CUDA -DPHIRST_GPU_ARCH=89` | NVIDIA GPU backend |
| KOKKOS | `-DPHIRST_BACKEND=KOKKOS` | Requires a Kokkos installation configured for your target device |
| ALPAKA | `-DPHIRST_BACKEND=ALPAKA -DALPAKA_BACKEND=CUDA -DPHIRST_GPU_ARCH=89` | Alpaka with CUDA sub-backend |
| SYCL | `-DPHIRST_BACKEND=SYCL -DSYCL_BACKEND=CUDA -DPHIRST_GPU_ARCH=sm_89` | SYCL with CUDA target |
| HIP | `-DPHIRST_BACKEND=HIP -DPHIRST_GPU_ARCH=gfx1100` | AMD ROCm/HIP backend |

## Quick start

```python
import phirst

result = phirst.integrate(
    phirst.DrellYanIntegrand(charge=2/3, coupling=1/137),
    n_particles=2,
    cm_energy=91.2,
    n_events=1_000_000,
    masses=None,
    seed=5489,
)
print(f"mean = {result.mean:.6e} ± {result.error:.6e}  ({result.n_events} events)")
```

## API reference

### `phirst.__version__`

```python
phirst.__version__
```

Installed package version string.

### `phirst.integrate`

```python
phirst.integrate(
    integrand,
    *,
    n_particles,
    cm_energy,
    n_events=100_000,
    masses=None,
    seed=5489,
    use_vegas=False,
)
```

Run Monte Carlo integration over RAMBO phase space.

**Parameters**

- `integrand`: one of the built-in Phirst integrand objects, or a Python callable.
- `n_particles` (`int`): final-state multiplicity. Supported values are `2..6`.
- `cm_energy` (`float`): centre-of-mass energy in GeV.
- `n_events` (`int`, default `100_000`): number of Monte Carlo samples.
- `masses` (`None` or array-like, default `None`): particle masses in GeV. `None` means all masses are zero.
- `seed` (`int`, default `5489`): random seed for reproducible sampling.
- `use_vegas` (`bool`, default `False`): enable VEGAS adaptive sampling for built-in integrands.

**Returns**

- `IntegrationResult`: object with `mean`, `error`, and `n_events`.

**Built-in integrands**

These run fully in compiled code and are the fastest path:

- `ConstantIntegrand(value=1.0)`
- `DrellYanIntegrand(charge=2/3, coupling=1/137)`
- `EggholderIntegrand(lambda_squared=1e6)`
- `MandelstamSIntegrand2(scale=1.0)`
- `MandelstamSIntegrand3(scale=1.0)`
- `MandelstamSIntegrand4(scale=1.0)`
- `MandelstamSIntegrand5(scale=1.0)`
- `MandelstamSIntegrand6(scale=1.0)`

**Callable integrands**

A Python integrand must be vectorised:

```python
def my_integrand(momenta):
    # momenta.shape == (n_events, n_particles, 4)
    # return shape == (n_events,)
    ...
```

The callable receives all sampled events in one NumPy array and must return one value per event.

**Numba GPU integrands**

Use `@phirst.integrand(n_particles=N)` to compile a Numba CUDA device function to PTX and
execute it through the Numba bridge:

```python
import phirst

@phirst.integrand(n_particles=2)
def my_fn(momenta_flat, n_particles):
    return momenta_flat[0] + momenta_flat[4]
```

`momenta_flat` is a length `n_particles * 4` flat array laid out as
`[E0, px0, py0, pz0, E1, px1, py1, pz1, ...]`.

**Constraints**

- `DrellYanIntegrand` requires `n_particles=2`.
- `EggholderIntegrand` requires `n_particles>=3`.
- `MandelstamSIntegrandN` requires exactly `n_particles=N`.
- `use_vegas` is available only for built-in integrands.

### `phirst.generate_phase_space`

```python
phirst.generate_phase_space(
    *,
    n_particles,
    cm_energy,
    n_events=10_000,
    masses=None,
    seed=5489,
)
```

Generate RAMBO phase-space points and weights without evaluating an integrand.

**Returns**

- `momenta` (`np.ndarray`): shape `(n_events, n_particles, 4)`, dtype `float64`
- `weights` (`np.ndarray`): shape `(n_events,)`, dtype `float64`

`weights` contains `exp(log_weight)` for each generated event.

### Built-in integrand classes

| Class | Signature | Description |
|---|---|---|
| `ConstantIntegrand` | `ConstantIntegrand(value=1.0)` | Constant matrix element; useful for phase-space volume checks |
| `DrellYanIntegrand` | `DrellYanIntegrand(charge=2/3, coupling=1/137)` | Drell-Yan cross-section kernel for 2-particle final states |
| `EggholderIntegrand` | `EggholderIntegrand(lambda_squared=1e6)` | Toy benchmark integrand for `N>=3` |
| `MandelstamSIntegrand2` | `MandelstamSIntegrand2(scale=1.0)` | Mandelstam-like invariant for 2 particles |
| `MandelstamSIntegrand3` | `MandelstamSIntegrand3(scale=1.0)` | Mandelstam-like invariant for 3 particles |
| `MandelstamSIntegrand4` | `MandelstamSIntegrand4(scale=1.0)` | Mandelstam-like invariant for 4 particles |
| `MandelstamSIntegrand5` | `MandelstamSIntegrand5(scale=1.0)` | Mandelstam-like invariant for 5 particles |
| `MandelstamSIntegrand6` | `MandelstamSIntegrand6(scale=1.0)` | Mandelstam-like invariant for 6 particles |

### `IntegrationResult`

```python
result.mean
result.error
result.n_events
```

| Attribute | Type | Meaning |
|---|---|---|
| `mean` | `float` | Weighted Monte Carlo estimate |
| `error` | `float` | Statistical uncertainty (1 sigma) |
| `n_events` | `int` | Number of sampled phase-space points |

## Integration paths

| Path | Input | Execution model | VEGAS | Best use |
|---|---|---|---|---|
| Built-in integrand | `phirst.DrellYanIntegrand(...)`, etc. | Fully compiled C++ | Yes | Production runs and best performance |
| Python callable | `def f(momenta): ...` | Phase-space generation in compiled code, observable in NumPy/Python | No | Fast prototyping and custom observables |
| Numba callable | `@phirst.integrand(n_particles=N)` | Numba PTX linked into a CUDA launch bridge | No | GPU execution for custom device functions |

## Phase space convention

Generated momenta use the layout

```python
momenta[event, particle, mu]
```

with

- `mu = 0`: energy `E`
- `mu = 1`: momentum `px`
- `mu = 2`: momentum `py`
- `mu = 3`: momentum `pz`

So a single event has shape `(n_particles, 4)` and stores four-vectors in `[E, px, py, pz]` order.

The returned `weights` are the Monte Carlo phase-space weights associated with each event. If you evaluate an observable manually,

```python
values = observable(momenta)      # shape (n_events,)
mean = np.mean(values * weights)
```

which reproduces the weighted average used internally by `phirst.integrate`.

## Examples

### 1. Drell-Yan cross-section with a built-in integrand

```python
import phirst

result = phirst.integrate(
    phirst.DrellYanIntegrand(charge=2/3, coupling=1/137),
    n_particles=2,
    cm_energy=91.2,
    n_events=1_000_000,
    masses=None,
    seed=5489,
    use_vegas=False,
)

print(f"Drell-Yan: {result.mean:.6e} ± {result.error:.6e}")
```

### 2. Custom integrand using `generate_phase_space`

```python
import numpy as np
import phirst

momenta, weights = phirst.generate_phase_space(
    n_particles=3,
    cm_energy=91.2,
    n_events=50_000,
    masses=None,
    seed=5489,
)

def energy_fraction(momenta):
    return momenta[:, :, 0].sum(axis=1) / 91.2

values = energy_fraction(momenta)
mean = np.mean(values * weights)
print(f"Manual integral: {mean:.6e}")
```

### 3. Momentum-conservation sanity check

```python
import numpy as np
import phirst

momenta, _ = phirst.generate_phase_space(
    n_particles=4,
    cm_energy=200.0,
    n_events=1000,
    seed=5489,
)

total = momenta.sum(axis=1)
np.testing.assert_allclose(total[:, 0], 200.0, atol=1e-8)
np.testing.assert_allclose(total[:, 1], 0.0, atol=1e-8)
np.testing.assert_allclose(total[:, 2], 0.0, atol=1e-8)
np.testing.assert_allclose(total[:, 3], 0.0, atol=1e-8)
```

## Backend performance

Representative throughput for built-in integrands is highly workload- and hardware-dependent, but the expected scale is:

| Backend | Approx. throughput | Notes |
|---|---:|---|
| SERIAL | ~1 million events/s | Single-thread CPU reference path |
| CUDA | ~100-200 million events/s | NVIDIA GPU |
| KOKKOS | ~100-200 million events/s | Depends on the Kokkos backend and build |
| ALPAKA | ~100-200 million events/s | Similar scale on NVIDIA GPUs |
| SYCL | ~100-200 million events/s | Depends on compiler and backend target |
| HIP | hardware-dependent | AMD path; benchmark on target hardware |

For Python callables, the bottleneck is the Python/NumPy observable evaluation, so expect much lower throughput than the built-in compiled integrands.

## Limitations and known issues

- `n_particles` is currently limited to `2..6`.
- Python callables are CPU-side in Phase 2; only the built-in integrands use the fully compiled integration path.
- VEGAS is available only for built-in integrands.
- Python callables must accept a single NumPy array with shape `(n_events, n_particles, 4)` and return shape `(n_events,)`.
- `masses` must have length exactly equal to `n_particles` when provided.
