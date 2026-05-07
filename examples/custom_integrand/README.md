# Custom integrand module example

Build the shared library:

```bash
cd examples/custom_integrand
mkdir -p build && cd build
cmake ..
cmake --build .
```

This produces `libmy_integrand.so` in the build directory.

Load it from Python:

```python
import phirst

mod = phirst.load_integrand(".../examples/custom_integrand/build/libmy_integrand.so", n_particles=2)
result = mod.integrate(cm_energy=100.0, masses=[0.0, 0.0], n_events=10000, seed=5489)
print(result.mean, result.error)
```
