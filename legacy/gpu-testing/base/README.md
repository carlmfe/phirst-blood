# RAMBO Monte Carlo Integrator - Base (Serial CPU)

Header-only, serial CPU reference implementation. No external dependencies.

## Requirements

- **CMake** ≥ 3.18
- **C++17** compiler (GCC ≥9, Clang ≥10, MSVC 2019+)

## Usage

### CMake
```cmake
find_package(rambo-base REQUIRED)
target_link_libraries(my_app PRIVATE rambo::base)
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

Store all physics parameters (couplings, masses, charges) in the struct:

```cpp
struct MyDrellYan {
    double quarkCharge;   // e.g., 2/3 for up-type
    double alphaEM;       // Fine-structure constant
    
    MyDrellYan(double eq, double alpha) : quarkCharge(eq), alphaEM(alpha) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // Compute full differential cross-section from momenta and parameters
        return dsigma;  // No library scaling applied
    }
};
```
