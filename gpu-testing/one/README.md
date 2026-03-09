# RAMBO Monte Carlo Integrator - Unified Multi-Backend

Header-only implementation supporting Serial, CUDA, Kokkos, SYCL, and Alpaka backends from a single source.

## Requirements

- **CMake** ≥ 3.18
- **C++17** compiler (GCC ≥9, Clang ≥10, MSVC 2019+)
- **Backend-specific** (optional):
  - CUDA: nvcc
  - Kokkos: Kokkos 5.0.1+
  - Alpaka: Alpaka 2.0.0+
  - SYCL: Intel DPC++ or AdaptiveCpp

## Usage

### Build
```bash
# Serial (default)
cmake -DRAMBO_BACKEND=SERIAL ..

# CUDA
cmake -DRAMBO_BACKEND=CUDA ..

# Kokkos (set Kokkos_ROOT or load module)
cmake -DRAMBO_BACKEND=KOKKOS ..

# Alpaka with CUDA (set alpaka_ROOT or load module)
cmake -DRAMBO_BACKEND=ALPAKA -DALPAKA_BACKEND=CUDA ..

# SYCL with CUDA
cmake -DRAMBO_BACKEND=SYCL -DCUDA_GPU_ARCH=sm_89 ..
```

### CMake
```cmake
find_package(rambo-one REQUIRED)
target_link_libraries(my_app PRIVATE rambo::one)
```

### Code
```cpp
#include <rambo/rambo.hpp>

int main() {
    constexpr int nParticles = 2;
    double masses[nParticles] = {0.000511, 0.000511};
    
    rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
    rambo::RamboIntegrator<rambo::DrellYanIntegrand, nParticles> 
        integrator(1000000, integrand);
    
    double mean, error;
    integrator.run(91.2, masses, mean, error, 5489);
}
```

## Custom Integrands

Store all physics parameters in the struct and use `RAMBO_HOST_DEVICE` decorator:

```cpp
struct MyDrellYan {
    double quarkCharge;   // e.g., 2/3 for up-type
    double alphaEM;       // Fine-structure constant
    
    RAMBO_HOST_DEVICE
    MyDrellYan(double eq, double alpha) : quarkCharge(eq), alphaEM(alpha) {}
    
    RAMBO_HOST_DEVICE
    auto evaluate(const double momenta[][4]) const -> double {
        // Compute full differential cross-section from momenta and parameters
        return dsigma;  // No library scaling applied
    }
};
```

## Architecture

```
include/rambo/
├── backend.hpp     # Backend detection & RAMBO_HOST_DEVICE decorators
├── math.hpp        # Portable math (sqrt, log, sin, cos, ...)
├── random.hpp      # XorShift64 RNG for all backends
├── parallel.hpp    # DeviceBuffer, deep_copy, grid_stride_reduce
├── phase_space.hpp # PhaseSpaceGenerator (RAMBO algorithm)
├── integrands.hpp  # DrellYanIntegrand, ConstantIntegrand
├── integrator.hpp  # RamboIntegrator (backend-agnostic)
└── rambo.hpp       # Single include header
```
