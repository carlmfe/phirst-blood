# RAMBO GPU Testing

Monte Carlo phase space integration using the RAMBO algorithm, implemented across multiple GPU frameworks for performance comparison.

## Project Structure

```
gpu-testing/
├── base/                 # Serial CPU reference (C++17, no dependencies)
├── kokkos/               # Kokkos (CUDA/OpenMP/Serial)
├── alpaka/               # Alpaka 2.0.0 (CUDA/HIP/CPU/OpenMP)
├── cuda/                 # Pure CUDA
├── sycl/                 # SYCL (CUDA/HIP/Intel)
├── benchmark.sh          # Performance comparison script
└── check_gpu.sh          # GPU utilization verification
```

Each implementation is a header-only library with:
- `include/rambo/rambo.hpp` - Main include file
- `include/rambo/phase_space.hpp` - PhaseSpaceGenerator, RamboAlgorithm
- `include/rambo/integrator.hpp` - RamboIntegrator class
- `include/rambo/integrands.hpp` - Example integrands (DrellYan, Eggholder, etc.)

## Installation

All implementations follow the same pattern:

```bash
mkdir build && cd build
cmake [OPTIONS] -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make install
```

**Implementation-specific options:**

| Implementation | CMake Options |
|----------------|---------------|
| base | (none required) |
| kokkos | `-DKokkos_ROOT=/path/to/kokkos` |
| cuda | (auto-detects CUDA) |
| alpaka | `-Dalpaka_ROOT=/path/to/alpaka -DALPAKA_BACKEND=CUDA` |
| sycl | `-DCMAKE_CXX_COMPILER=clang++ -DCUDA_GPU_ARCH=sm_XX` |

## Custom Integrands

The integrand struct is your physics payload. Store all parameters needed for evaluation (coupling constants, masses, charges, etc.) — the library passes these to GPU threads automatically.

```cpp
// Example: Drell-Yan cross-section (q qbar -> l+ l-)
struct DrellYanIntegrand {
    double quarkCharge;   // e.g., 2/3 for up-type quarks
    double alphaEM;       // Fine-structure constant (~1/137)
    
    DrellYanIntegrand(double eq, double alpha) 
        : quarkCharge(eq), alphaEM(alpha) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i = particle index, mu = 0:E, 1:px, 2:py, 3:pz
        // Compute Mandelstam variables, matrix element, phase space factors
        // Return the FULL differential cross-section (no library scaling)
        return dsigma;
    }
};

// Usage: all physics parameters provided at construction
DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
```

**Memory note:** Typical physics parameters (5-10 doubles, ~80 bytes) are negligible. For large lookup tables (PDFs, form factors), store a device pointer in the struct.

**GPU backends require device-callable decorators** — see individual README files:
- **Kokkos**: `KOKKOS_FUNCTION`, `KOKKOS_INLINE_FUNCTION`
- **CUDA**: `__host__ __device__`, `__device__`
- **Alpaka**: `ALPAKA_FN_HOST_ACC`
- **SYCL**: Use `sycl::` math functions

## Shell Scripts

### benchmark.sh

Builds and benchmarks all implementations, comparing throughput.

```bash
# Basic usage (10M events, 3 runs each)
./benchmark.sh

# Custom event count
./benchmark.sh 100000000

# Custom event count and seed
./benchmark.sh 100000000 42

# Custom runs
./benchmark.sh 10000000 5489 5
```

**Output includes:**
- Build status for each implementation
- GPU utilization verification
- Throughput comparison (events/sec)
- Relative performance ranking

### check_gpu.sh

Verifies that a program actually uses the GPU during execution.

```bash
# Check GPU utilization for any executable
./check_gpu.sh ./alpaka/build/rambo_alpaka 10000000 5489
```

**Output includes:**
- Initial GPU state
- Maximum GPU utilization during execution
- Confirmation message: `✓ GPU WAS UTILIZED` or `✗ GPU was NOT utilized`

## Quick Start

```bash
# Base (no dependencies)
cd base && mkdir build && cd build && cmake .. && make

# Kokkos
cd kokkos && mkdir build && cd build
cmake -DKokkos_ROOT=/path/to/kokkos .. && make

# CUDA
cd cuda && mkdir build && cd build && cmake .. && make

# Alpaka
cd alpaka && mkdir build && cd build
cmake -Dalpaka_ROOT=/path/to/alpaka -DALPAKA_BACKEND=CUDA .. && make

# SYCL
cd sycl && mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=clang++ -DCUDA_GPU_ARCH=sm_89 .. && make
```

## Performance Reference

| Implementation | Throughput | Hardware |
|---------------|------------|----------|
| Base (Serial) | ~3M ev/s | CPU single-thread |
| Kokkos (CUDA) | ~85M ev/s | RTX 2000 Ada |
| Alpaka (CUDA) | ~115M ev/s | RTX 2000 Ada |
| CUDA | ~165M ev/s | RTX 2000 Ada |
| SYCL (CUDA) | ~155M ev/s | RTX 2000 Ada |

*Results vary by hardware and event count.*
