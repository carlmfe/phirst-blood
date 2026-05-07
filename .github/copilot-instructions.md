# RAMBO GPU Codebase Instructions

## Project Overview
RAMBO (Recursive Algorithm for Multi-Body Omni-sampling) is a high-energy physics Monte Carlo integration library. The project contains multiple backend implementations using different parallel computing frameworks: base CPU, OpenMP, Kokkos, Alpaka, CUDA, and SYCL.

**Core Concept**: Generate random 4-momenta for particles in high-energy physics decays using the RAMBO algorithm, compute physics integrands over these momenta, and perform Monte Carlo integration to estimate cross-sections.

## Current File Structure

```
gpu-testing/
├── alpaka/                    # Alpaka 2.0.0 implementation (CUDA/CPU/OpenMP backends)
│   ├── CMakeLists.txt         # Backend selection via ALPAKA_BACKEND cache variable
│   ├── integrands.hpp         # EggholderIntegrand, ConstantIntegrand, MandelstamSIntegrand
│   ├── integrator.hpp         # RamboIntegrator class, MonteCarloKernel, ReductionKernel
│   ├── main.cpp               # Entry point with runBenchmark() helper
│   ├── rambo_alpaka.hpp       # PhaseSpaceGenerator<nParticles> template
│   └── README.md              # Build and usage documentation
│
├── kokkos/                    # Kokkos implementation (mirrors Alpaka structure)
│   ├── CMakeLists.txt         # Uses find_package(Kokkos REQUIRED)
│   ├── integrands.hpp         # Same integrand classes as Alpaka
│   ├── integrator.hpp         # RamboIntegrator using Kokkos::parallel_reduce
│   ├── main.cpp               # Entry point matching Alpaka's format
│   ├── rambo_kokkos.hpp       # PhaseSpaceGenerator<nParticles> template
│   └── README.md              # Build and usage documentation
│
├── cuda/                      # Direct CUDA implementation
│   ├── CMakeLists.txt         # Auto-detects nvcc and GPU architecture
│   ├── integrands.cuh         # CUDA device integrands
│   ├── integrator.cuh         # CUDA kernel and reduction
│   ├── main.cu                # Entry point
│   ├── rambo_cuda.cuh         # PhaseSpaceGenerator with curand
│   └── README.md              # Build and usage documentation
│
├── sycl/                      # SYCL implementation (CUDA/Intel backends)
│   ├── CMakeLists.txt         # Multi-backend support (CUDA, Intel, AdaptiveCpp)
│   ├── integrands.hpp         # SYCL device integrands
│   ├── integrator.hpp         # SYCL parallel_for with reduction
│   ├── main.cpp               # Entry point
│   ├── rambo_sycl.hpp         # PhaseSpaceGenerator template
│   └── README.md              # Build and usage documentation
│
├── base/                      # Serial CPU reference implementation
│   ├── CMakeLists.txt         # Portable C++20 build
│   ├── integrands.hpp         # Reference integrand implementations
│   ├── integrator.hpp         # Serial Monte Carlo loop
│   ├── main.cpp               # Entry point
│   ├── rambo_base.hpp         # Reference RAMBO algorithm
│   └── README.md              # Build and usage documentation
│
├── omp/                       # OpenMP implementation
│   ├── CMakeLists.txt
│   ├── integrator.cpp/.h
│   ├── main.cpp
│   ├── rambo_omp.cpp/.h
│   └── rng.cpp/.h
│
├── benchmark.sh               # Build and run all implementations
├── check_gpu.sh               # GPU utilization verification script
└── README.md                  # Main documentation
```

## Architecture

### Unified Code Structure (Alpaka/Kokkos/CUDA/SYCL)
All GPU implementations share the same structure:

| Component | Alpaka | Kokkos | CUDA | SYCL | Purpose |
|-----------|--------|--------|------|------|---------|
| Phase Space Generator | `PhaseSpaceGenerator<nParticles>` | Same | Same | Same | RAMBO algorithm |
| Integrands | `EggholderIntegrand`, `ConstantIntegrand`, `MandelstamSIntegrand<N>` | Same | Same | Same | Physics functions |
| Integrator | `RamboIntegrator<AccTag, Integrand, N>` | `RamboIntegrator<Integrand, N>` | `RamboIntegrator<Integrand, N>` | `RamboIntegrator<Integrand, N>` | MC integration |
| Result | `IntegrationResult` struct | Same | Same | Same | Mean, error, sums |
| Entry Point | `runBenchmark<>()` + `main()` | Same | Same | Same | Warmup + timed run |

### Modular Algorithm Design

All implementations use a two-layer design for phase space generation:

1. **Algorithm Layer** (`RamboAlgorithm<nParticles>`):
   - Contains the core physics algorithm
   - Has a `generate()` method with framework-specific decorators
   - Algorithm constants as static constexpr members (`tolerance`, `maxIterations`)

2. **Wrapper Layer** (`PhaseSpaceGenerator<nParticles, Algorithm>`):
   - Provides uniform callable interface via `operator()`
   - Stores configuration (`cmEnergy`, `masses`)
   - Default template parameter: `RamboAlgorithm<nParticles>`

To add a new phase space algorithm:
```cpp
template <int nParticles>
struct MyNewAlgorithm {
    template <typename TRng>
    auto generate(double cmEnergy, const double* masses, 
                  TRng& rng, double momenta[][4]) const -> double {
        // Your implementation here
        return logWeight;
    }
};

// Use it:
PhaseSpaceGenerator<3, MyNewAlgorithm<3>> generator(cmEnergy, masses);
```

### Key Classes and Templates

**PhaseSpaceGenerator<nParticles>** (`rambo_*.hpp`):
```cpp
template <int nParticles>
struct PhaseSpaceGenerator {
    double cmEnergy;
    const double* masses;
    
    // Returns log(weight), fills momenta[nParticles][4]
    template <typename TRng>
    DEVICE_FUNCTION auto operator()(TRng& rng, double momenta[][4]) const -> double;
};
```

**Integrands** (`integrands.hpp`):
```cpp
struct EggholderIntegrand {
    double lambdaSquared;
    DEVICE_FUNCTION auto evaluate(const double momenta[][4]) const -> double;
};

struct ConstantIntegrand { double value; /* ... */ };

template <int nParticles>
struct MandelstamSIntegrand { double scale; /* ... */ };
```

**RamboIntegrator** (`integrator.hpp`):
```cpp
template <typename Integrand, int NumParticles>
class RamboIntegrator {
public:
    void run(double cmEnergy, const double* masses,
             double& mean, double& error, uint64_t seed);
private:
    void copyMassesToDevice(...);
    void launchMonteCarloKernel(...);
    // Alpaka: void performReduction(...);  // Tree reduction
    // Kokkos: Uses parallel_reduce natively
};
```

## Build System

### Portable CMake Configuration
All implementations use portable CMake without environment module dependencies. Libraries are found via standard CMake mechanisms (`find_package` with `*_ROOT` hints).

```cmake
# Alpaka CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project("RamboAlpaka" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ALPAKA_BACKEND "CPU" CACHE STRING "Alpaka backend: CUDA, CPU, OMP")

# Check for CUDA availability at runtime
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND ALPAKA_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(alpaka_ACC_GPU_CUDA_ENABLE ON)
    set(CMAKE_CUDA_ARCHITECTURES "native")  # Auto-detect GPU
    add_compile_definitions(ALPAKA_USE_CUDA)
elseif(ALPAKA_BACKEND STREQUAL "CPU")
    set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
    add_compile_definitions(ALPAKA_USE_CPU_SERIAL)
endif()

# User provides alpaka_ROOT or CMAKE_PREFIX_PATH
find_package(alpaka REQUIRED)
alpaka_add_executable(rambo_alpaka main.cpp)
target_link_libraries(rambo_alpaka PUBLIC alpaka::alpaka)
```

```cmake
# Kokkos CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project("RamboKokkos" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# User provides Kokkos_ROOT or CMAKE_PREFIX_PATH
find_package(Kokkos REQUIRED)
add_executable(rambo_kokkos main.cpp)
target_link_libraries(rambo_kokkos Kokkos::kokkos)
```

### Build Commands
```bash
# Alpaka with CUDA backend
cd gpu-testing/alpaka
rm -rf build && mkdir build && cd build
cmake -DALPAKA_BACKEND=CUDA -Dalpaka_ROOT=/path/to/alpaka ..
make -j4
./rambo_alpaka 1000000 5489

# Alpaka with CPU serial backend (default)
cmake -DALPAKA_BACKEND=CPU -Dalpaka_ROOT=/path/to/alpaka ..

# Kokkos (backend determined by Kokkos installation)
cd gpu-testing/kokkos
rm -rf build && mkdir build && cd build
cmake ..
make -j4
./rambo_kokkos 1000000 5489
```

### Per-backend Build Instructions (modules & commands)

- Base (serial): no modules required

```bash
cd gpu-testing/base
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
./rambo_base 1000000 5489
```

- Alpaka (CUDA backend): load Alpaka module (example: `module load alpaka/2.0.0_cuda`) and ensure CUDA is available

```bash
module load alpaka/2.0.0_cuda
cd gpu-testing/alpaka
rm -rf build && mkdir build && cd build
cmake -DALPAKA_BACKEND=CUDA -Dalpaka_ROOT=/path/to/alpaka ..
make -j$(nproc)
./rambo_alpaka 1000000 5489
```

- Kokkos: load Kokkos module (example: `module load kokkos/5.0.1`) or set `KOKKOS_ROOT`

```bash
module load kokkos/5.0.1
cd gpu-testing/kokkos
rm -rf build && mkdir build && cd build
cmake -DKokkos_ROOT=$KOKKOS_ROOT ..
make -j$(nproc)
./rambo_kokkos 1000000 5489
```

- CUDA: load CUDA module (example: `module load cuda/13.0`) and ensure `nvcc` is on PATH

```bash
module load cuda/13.0
cd gpu-testing/cuda
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
./rambo_cuda 1000000 5489
```

- SYCL (CUDA backend): load SYCL module (example: `module load sycl/cuda`), set `SYCL_CXX` to the SYCL-enabled `clang++` and then configure

```bash
module load sycl/cuda
export SYCL_CXX=clang++
cd gpu-testing/sycl
rm -rf build && mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER="$SYCL_CXX" ..
make -j$(nproc)
./rambo_sycl 1000000 5489
```

Notes:
- Replace `/path/to/alpaka` and `$KOKKOS_ROOT` with actual installation prefixes when using `find_package` hints.
- Use `-DALPAKA_BACKEND=CPU` for Alpaka serial/backends if CUDA is not available.
- For reproducible benchmarking, use the same seed (e.g., `5489`) across backends.

## Alpaka 2.0.0 Specifics

### RNG API (Critical)
Alpaka 2.0.0 separates RNG engine creation from distribution:

```cpp
// Create RNG engine (Philox) with unique seed per thread
auto engine = alpaka::rand::engine::createDefault(
    acc, 
    static_cast<uint32_t>(baseSeed + threadIdx),
    static_cast<uint32_t>((baseSeed + threadIdx) >> 32)
);

// Create uniform distribution [0.0, 1.0)
alpaka::rand::RandDefault rand;
auto dist = alpaka::rand::distribution::createUniformReal<double>(rand);

// Use: double r = dist(engine);
```

**Do NOT use**: `alpaka::rand::uniform(...)` - this doesn't exist in 2.0.0.

### Backend Selection (Compile-Time)
```cpp
// main.cpp
#if defined(ALPAKA_USE_CUDA)
    using DefaultAccTag = alpaka::TagGpuCudaRt;
    constexpr const char* BACKEND_NAME = "CUDA GPU";
#elif defined(ALPAKA_USE_CPU_OMP)
    using DefaultAccTag = alpaka::TagCpuOmp2Blocks;
    constexpr const char* BACKEND_NAME = "CPU OpenMP";
#elif defined(ALPAKA_USE_CPU_SERIAL)
    using DefaultAccTag = alpaka::TagCpuSerial;
    constexpr const char* BACKEND_NAME = "CPU Serial";
#endif
```

### Kernel and Memory Patterns
```cpp
// Kernel struct with operator()
template <typename Integrand, int NumParticles>
struct MonteCarloKernel {
    double* sums;
    // ...
    
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        // ...
    }
};

// Memory allocation and transfer
auto deviceBuf = alpaka::allocBuf<double, Idx>(devAcc, extent);
auto hostBuf = alpaka::allocBuf<double, Idx>(devHost, extent);
alpaka::memcpy(queue, deviceBuf, hostBuf);
alpaka::wait(queue);

// Kernel launch
alpaka::KernelCfg<Acc> const cfg = {extent, elemPerThread};
auto const workDiv = alpaka::getValidWorkDiv(cfg, devAcc, kernel);
alpaka::exec<Acc>(queue, workDiv, kernel);
```

## Kokkos Specifics

### Lambda Visibility for CUDA
When using CUDA backend, lambdas in class methods must be in **public** scope:

```cpp
template <typename Integrand, int NumParticles>
class RamboIntegrator {
public:
    // Must be public for CUDA extended __host__ __device__ lambda
    void launchMonteCarloKernel(...) {
        Kokkos::parallel_reduce("MC", nEvents,
            KOKKOS_LAMBDA(int64_t i, double& sum, double& sum2) {
                // ...
            },
            Kokkos::Sum<double>(sumVal),
            Kokkos::Sum<double>(sum2Val)
        );
    }
private:
    // Private members only, no lambdas
};
```

### RNG Pool Pattern
```cpp
using RngPool = Kokkos::Random_XorShift64_Pool<>;
RngPool rngPool(seed);

Kokkos::parallel_reduce(...,
    KOKKOS_LAMBDA(...) {
        auto rng = rngPool.get_state();
        double r = rng.drand();  // [0, 1)
        // ...
        rngPool.free_state(rng);
    }, ...);
```

## Development Workflow

### File Editing Rules (CRITICAL)
- **NEVER** remove and rewrite entire files. Use incremental edits with `replace_string_in_file` or `multi_replace_string_in_file`.
- **NEVER** use terminal commands like `cat`, `echo >>`, `sed -i`, or heredocs to edit files. ALWAYS use the designated editing functions (`replace_string_in_file`, `multi_replace_string_in_file`, `create_file` for new files only).
- When making large changes, break them into multiple smaller `replace_string_in_file` calls.
- For new files, use `create_file`. For existing files, ONLY use `replace_string_in_file` or `multi_replace_string_in_file`.

### Incremental Development Process
1. **Start with working reference**: Use Kokkos or base implementation as template
2. **Port structure first**: Copy file organization, class names, variable names
3. **Compile incrementally**: Fix one file at a time, don't attempt full build initially
4. **Test with CPU backend first**: Easier debugging, faster iteration
5. **Verify GPU utilization**: Use `check_gpu.sh` to confirm GPU is actually used
6. **Benchmark and compare**: Use same seed (5489) for reproducibility

### GPU Verification Script
```bash
# Usage: ./check_gpu.sh <executable> [args...]
./check_gpu.sh ./build/rambo_alpaka 10000000 5489

# Output includes:
# - Initial/final GPU state
# - Max/avg GPU utilization during execution
# - ✓ GPU WAS UTILIZED confirmation
```

### Keeping Implementations Synchronized
When updating one implementation (e.g., Alpaka), update others to match:

1. **Variable names**: Use same names across frameworks (`cmEnergy`, `logWeight`, `momenta`)
2. **Class structure**: Same class names, same method signatures
3. **Output format**: Same benchmark output (Mean, Error, Time, Throughput)
4. **Documentation**: Same comments and section headers

### Adding New Features
1. Implement in Alpaka first (most flexible backend selection)
2. Test with CUDA backend + GPU verification
3. Test with CPU backend for debugging
4. Port to Kokkos with same structure
5. Update base implementation if algorithm changed
6. Update this documentation
7. **Update README.md files** if user-facing behavior changes (defaults, outputs, usage)
8. **Propose updates to this instructions file** if new patterns, pitfalls, or architectural insights were discovered (consult user before adding)

### README Maintenance
After making changes, check if any README files need updating:
- `gpu-testing/README.md` - Main overview, integrand math, script usage
- `gpu-testing/base/README.md` - Base implementation specifics
- `gpu-testing/kokkos/README.md` - Kokkos implementation specifics  
- `gpu-testing/alpaka/README.md` - Alpaka implementation specifics
- `gpu-testing/cuda/README.md` - CUDA implementation specifics
- `gpu-testing/sycl/README.md` - SYCL implementation specifics

Update READMEs when:
- Default parameter values change
- Output format changes
- New features or scripts are added
- Build requirements change

## Library Packaging

### Design Decision: Header-Only
The RAMBO library uses a **header-only** design because:
- Template-heavy code (integrands, particle counts are template parameters)
- Avoids ABI compatibility issues with Kokkos backends
- Simplifies distribution (no need to match library build with user's Kokkos)
- Standard pattern for performance-critical C++ libraries (Eigen, Kokkos, etc.)

### Library Structure (Kokkos Reference Implementation)
```
kokkos/
├── CMakeLists.txt                    # INTERFACE library + install rules
├── main.cpp                          # Example application (not part of library)
├── include/
│   └── rambo/
│       ├── rambo.hpp                 # Main include (includes all below)
│       ├── phase_space.hpp           # PhaseSpaceGenerator, RamboAlgorithm
│       ├── integrator.hpp            # RamboIntegrator template
│       └── integrands.hpp            # Example integrand implementations
├── cmake/
│   └── rambo-kokkosConfig.cmake.in   # Package config template
└── README.md                         # Library documentation
```

### CMakeLists.txt Pattern for Header-Only Library
```cmake
# Create INTERFACE library (no compiled code)
add_library(rambo_kokkos_lib INTERFACE)
add_library(rambo::kokkos ALIAS rambo_kokkos_lib)

# Include directories with generator expressions
target_include_directories(rambo_kokkos_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies (propagated to users)
target_link_libraries(rambo_kokkos_lib INTERFACE Kokkos::kokkos)

# Minimum C++ standard
target_compile_features(rambo_kokkos_lib INTERFACE cxx_std_17)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install and export targets for find_package()
install(TARGETS rambo_kokkos_lib
    EXPORT rambo-kokkos-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT rambo-kokkos-targets
    FILE rambo-kokkos-targets.cmake
    NAMESPACE rambo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-kokkos
)
```

### Namespace Convention
All library code lives in `namespace rambo { }`:
- `rambo::PhaseSpaceGenerator<N>`
- `rambo::RamboAlgorithm<N>`
- `rambo::RamboIntegrator<Integrand, N>`
- `rambo::DrellYanIntegrand`, `rambo::ConstantIntegrand`, etc.
- `rambo::IntegrationResult`

### Creating Custom Integrands
Users wrap their physics functions by creating a struct with an `evaluate()` method:

```cpp
// Original function (must be device-callable for GPU)
KOKKOS_INLINE_FUNCTION
double myMatrixElement(const double p1[4], const double p2[4]) {
    // Physics calculation using 4-momenta
    return result;
}

// Wrapper struct for the library
struct MyIntegrand {
    double scale;  // Optional parameters
    
    KOKKOS_FUNCTION MyIntegrand(double s = 1.0) : scale(s) {}
    
    KOKKOS_INLINE_FUNCTION 
    auto evaluate(const double momenta[][4]) const -> double {
        // momenta[i][mu]: i = particle, mu = 0:E, 1:px, 2:py, 3:pz
        return myMatrixElement(momenta[0], momenta[1]) * scale;
    }
};

// Usage
MyIntegrand integrand(1.0);
rambo::RamboIntegrator<MyIntegrand, 2> integrator(nEvents, integrand);
```

**Key requirements for integrands:**
- Must have `evaluate(const double momenta[][4]) const -> double` method
- Constructor and evaluate must have `KOKKOS_FUNCTION` / `KOKKOS_INLINE_FUNCTION` decorators
- Any called functions must also be device-callable for GPU backends

### User Requirements (Document in README)
1. **Kokkos pre-installed** with desired backend
2. **CMake ≥3.18** for generator expressions and package config
3. **C++17 or later** compiler
4. **User responsibility**: `Kokkos::initialize()` / `Kokkos::finalize()`

### Downstream Usage Pattern
```cmake
# User's CMakeLists.txt
find_package(rambo-kokkos REQUIRED)
target_link_libraries(my_app PRIVATE rambo::kokkos)
```

```cpp
// User's code
#include <Kokkos_Core.hpp>
#include <rambo/rambo.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
        rambo::RamboIntegrator<rambo::DrellYanIntegrand, 2> 
            integrator(1000000, integrand);
        double mean, error;
        integrator.run(91.2, masses, mean, error, 5489);
    }
    Kokkos::finalize();
}
```

### Porting to Other Backends
When creating library versions for Alpaka, SYCL, CUDA:
1. Same directory structure (`include/rambo/`)
2. Same namespace (`rambo::`)
3. Same class names and interfaces
4. Backend-specific macros (`KOKKOS_FUNCTION` → `ALPAKA_FN_ACC`, etc.)
5. Different package name (`rambo-alpaka`, `rambo-cuda`, etc.)

## Performance Reference

| Backend | Events/sec | Notes |
|---------|------------|-------|
| Alpaka CUDA | ~130M | RTX 2000 Ada |
| Kokkos CUDA | ~135M | RTX 2000 Ada |
| CUDA | ~165M | RTX 2000 Ada |
| SYCL CUDA | ~155M | RTX 2000 Ada |
| Alpaka CPU Serial | ~650K | Single thread |
| Alpaka CPU OpenMP | ~8M | Multi-threaded |

## Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| CMake can't find framework | Library path not set | Use `-DKokkos_ROOT=...` or `-Dalpaka_ROOT=...` |
| Alpaka RNG compile error | Wrong API for version | Use `engine::createDefault` + `distribution::createUniformReal` |
| Kokkos lambda error | Private method with lambda | Move method to public section |
| Kokkos CUDA kernel not running | C++ standard set before find_package | Call `find_package(Kokkos)` FIRST, don't set CMAKE_CXX_STANDARD |
| Results are NaN/Inf | RNG state=0 | Ensure non-zero seed, check thread indexing |
| GPU not utilized | Wrong backend | Verify compile definitions, check `check_gpu.sh` output |
| Performance regression | Missing warmup | Add warmup run before timed benchmark |
| SYCL compile flags error | Flags passed as single string | Use separate entries in `target_compile_options` |

## Integration Points & Dependencies

| Backend | Library Path Variable | C++ Standard | Key API |
|---------|----------------------|--------------|---------|
| alpaka  | `alpaka_ROOT` | Set by Alpaka | `alpaka::exec`, `alpaka::rand::engine::createDefault` |
| kokkos  | `Kokkos_ROOT` | Set by Kokkos | `Kokkos::parallel_reduce`, `Kokkos::Random_XorShift64_Pool` |
| sycl    | `CMAKE_CXX_COMPILER` (SYCL clang++) | C++20 | `sycl::parallel_for`, `sycl::reduction` |
| cuda    | Auto-detected from PATH | C++20 | CUDA kernels, `__device__` functions |
| base    | None | C++20 | Standard library only |
| omp     | None (system) | C++20 | `#pragma omp parallel` |

**CRITICAL for Kokkos/Alpaka**: Do NOT set `CMAKE_CXX_STANDARD` before `find_package()`. These libraries set their own compiler flags based on how they were built. Setting the standard beforehand can cause silent failures where CUDA kernels don't execute.

## Agent Roster

Four specialised agents assist with this repository. Each is activated via its instruction
file in `.github/instructions/`. When working in one agent's domain, do not make changes
that belong to another agent — instead stop and tell the user which agent to invoke and
what to ask.

| Agent | Instruction file | Owns |
|-------|-----------------|------|
| **Deployment** | `deployment.instructions.md` | CI/CD workflows (`.github/workflows/`), build environments, `module load` setup, GPU architecture configuration, `CMakeLists.txt` build system |
| **Testing** | `testing.instructions.md` | GoogleTest suite (`tests/`), cross-backend consistency checks, test `CMakeLists.txt` |
| **C++ Development** | `cpp-development.instructions.md` | Library headers (`include/phirst/`), new backends, integrands, algorithms, code style (clang-format/tidy) |
| **Python Interface** | `python-interface.instructions.md` | pybind11 bindings (`python/`), `pyproject.toml`, scikit-build-core, Python test suite |

---

## Machine-Specific Configuration (Developer Reference)

**Note**: This section is for Copilot/developer reference only. These paths are specific to the development machine and should NOT appear in user-facing documentation or code.

### Current Development Environment
- **GPU**: NVIDIA RTX 2000 Ada (sm_89)
- **CUDA Toolkit**: 13.0, installed at `/usr/local/cuda-13.0`
- **Kokkos**: Development version at `/opt/kokkos/5.0.1`
- **Alpaka**: Version 2.0.0 with CUDA at `/opt/alpaka/2.0.0-cuda`
- **SYCL**: Intel DPC++/LLVM with CUDA backend, module `sycl/cuda`

### Module Commands (ALWAYS use these)
```bash
# Load modules before building - DO NOT source setvars.sh or set paths manually
module load kokkos/5.0.1        # For Kokkos builds
module load alpaka/2.0.0_cuda # For Alpaka builds  
module load sycl/cuda         # For SYCL builds
```

### Build Commands for This Machine
```bash
# Kokkos (after module load kokkos/5.0.1)
cmake ..

# Alpaka with CUDA (after module load alpaka/2.0.0_cuda)
cmake -DALPAKA_BACKEND=CUDA ..

# SYCL (after module load sycl/cuda)
cmake -DCUDA_GPU_ARCH=sm_89 ..

# CUDA (auto-detected from PATH)
cmake ..
```

### GPU Architecture
The local GPU (RTX 2000 Ada) uses `sm_89`. When testing locally, override the default `sm_70`:
```bash
cmake -DCUDA_GPU_ARCH=sm_89 ..  # For SYCL/OMP
```

