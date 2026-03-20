# RAMBO Monte Carlo Integrator - Kokkos

Header-only, GPU-accelerated phase space generation library using Kokkos.

## Requirements

- **CMake** ≥ 3.18
- **C++17** compiler (GCC ≥9, Clang ≥10, NVCC ≥11)
- **Kokkos** ≥ 4.0, pre-built with desired backend (Serial, OpenMP, CUDA, HIP)

## Usage

### CMake
```cmake
find_package(rambo-kokkos REQUIRED)
target_link_libraries(my_app PRIVATE rambo::kokkos)
```

### Code
```cpp
#include <Kokkos_Core.hpp>
#include <rambo/rambo.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int nParticles = 2;
        double masses[nParticles] = {0.000511, 0.000511};
        
        rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
        rambo::RamboIntegrator<rambo::DrellYanIntegrand, nParticles> 
            integrator(1000000, integrand);
        
        double mean, error;
        integrator.run(91.2, masses, mean, error, 5489);
    }
    Kokkos::finalize();
}
```

## Custom Integrands

Store all physics parameters in the struct. Use `KOKKOS_FUNCTION` decorators:

```cpp
struct MyDrellYan {
    double quarkCharge;   // e.g., 2/3 for up-type
    double alphaEM;       // Fine-structure constant
    
    KOKKOS_FUNCTION MyDrellYan(double eq, double alpha) 
        : quarkCharge(eq), alphaEM(alpha) {}
    
    KOKKOS_INLINE_FUNCTION 
    auto evaluate(const double momenta[][4]) const -> double {
        // Compute full differential cross-section from momenta and parameters
        return dsigma;  // No library scaling applied
    }
};
```

**Note**: Any function called from `evaluate()` must also have `KOKKOS_INLINE_FUNCTION`.
