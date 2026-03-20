# RAMBO Library

A header-only Monte Carlo integration library for high-energy physics phase space generation.

## Overview

RAMBO (RAndom Momenta, Beautifully Organized) generates random 4-momenta for particles in high-energy physics decays and performs Monte Carlo integration to estimate cross-sections.

This is the **library package** of RAMBO, designed to be installed and used in downstream projects.

## Features

- **Header-only**: No compilation required, just include and link
- **Dual backend support**: 
  - Serial base CPU implementation (default, no dependencies)
  - Kokkos backend for GPU/multi-threaded execution
- **Modular design**: Custom integrands can be easily defined
- **Standard physics integrands**: Eggholder, Constant, Mandelstam S

## Building and Installation

### Prerequisites

- CMake ≥ 3.18
- C++17 or later compiler

**Optional:**
- Kokkos library (if using `WITH_KOKKOS=ON`)

### Basic Installation (Serial Backend)

```bash
mkdir build && cd build
cmake ..
make install
```

This installs the RAMBO library with the serial base implementation (no external dependencies).

### Installation with Kokkos Backend

```bash
mkdir build && cd build
cmake -DWITH_KOKKOS=ON ..
make install
```

This configures RAMBO to use the Kokkos backend. Ensure Kokkos is installed and available via `find_package()`:

```bash
# If Kokkos is installed in a non-standard location:
cmake -DWITH_KOKKOS=ON -DKokkos_ROOT=/path/to/kokkos ..
```

## Usage in Downstream Projects

After installation, use RAMBO in your `CMakeLists.txt`:

```cmake
find_package(rambo REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE rambo::rambo)
```

### Example Code

```cpp
#include <rambo/rambo.hpp>

// Define a custom integrand
struct MyIntegrand {
    double scale;
    
    MyIntegrand(double s = 1.0) : scale(s) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // Compute physics function using 4-momenta
        // momenta[i][mu]: i = particle index, mu = 0:energy, 1-3: momentum
        return 1.0;  // Example: constant integrand
    }
};

int main() {
    MyIntegrand integrand(1.0);
    rambo::RamboIntegrator<MyIntegrand, 2> integrator(1000000, integrand);
    
    double masses[] = {0.0, 0.0};
    double mean, error;
    integrator.run(91.2, masses, mean, error, 5489);
    
    return 0;
}
```

**For Kokkos backend, wrap your application:**

```cpp
#include <Kokkos_Core.hpp>
#include <rambo/rambo.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Your RAMBO code here
    }
    Kokkos::finalize();
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `WITH_KOKKOS` | `OFF` | Enable Kokkos backend |
| `CMAKE_INSTALL_PREFIX` | system default | Installation directory |

## Library Components

| Header | Description |
|--------|-------------|
| `rambo.hpp` | Main include file |
| `phase_space.hpp` | Phase space generator and RNG |
| `integrator.hpp` | Integration driver and result structures |
| `integrands.hpp` | Example integrand implementations |

## Performance Characteristics

Performance varies significantly by backend and platform:

- **Serial base**: ~650K events/sec (single-threaded)
- **Kokkos CPU**: ~8M events/sec (multi-threaded)
- **Kokkos GPU**: ~130M events/sec (NVIDIA RTX 2000 Ada)

Results depend on hardware, compiler optimizations, and integrand complexity.

## License

This library is part of the RAMBO project. See main project repository for licensing information.
