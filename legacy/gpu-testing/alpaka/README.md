# RAMBO Monte Carlo Integrator - Alpaka

Header-only Alpaka library for portable GPU/CPU phase space generation.
Supports CUDA, HIP, SYCL, CPU Serial, and OpenMP backends.

## Requirements

- **CMake** ≥ 3.18
- **C++20** compiler (GCC ≥10, Clang ≥12, NVCC ≥11)
- **Alpaka** ≥ 2.0.0, pre-built with desired backend

## Usage

### CMake
```cmake
find_package(rambo-alpaka REQUIRED)
target_link_libraries(my_app PRIVATE rambo::alpaka)
```

### Code
```cpp
#include <rambo/rambo.hpp>

int main() {
    using AccTag = alpaka::TagGpuCudaRt;  // Or TagCpuSerial, TagCpuOmp2Blocks
    
    constexpr int nParticles = 2;
    double masses[nParticles] = {0.000511, 0.000511};
    
    rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
    rambo::RamboIntegrator<AccTag, rambo::DrellYanIntegrand, nParticles> 
        integrator(1000000, integrand);
    
    double mean, error;
    integrator.run(91.2, masses, mean, error, 5489);
}
```

## Custom Integrands

Store all physics parameters in the struct. Use `ALPAKA_FN_HOST_ACC` decorator:

```cpp
struct MyDrellYan {
    double quarkCharge;   // e.g., 2/3 for up-type
    double alphaEM;       // Fine-structure constant
    
    ALPAKA_FN_HOST_ACC MyDrellYan(double eq, double alpha) 
        : quarkCharge(eq), alphaEM(alpha) {}
    
    ALPAKA_FN_HOST_ACC auto evaluate(const double momenta[][4]) const -> double {
        // Compute full differential cross-section from momenta and parameters
        return dsigma;  // No library scaling applied
    }
};
```
