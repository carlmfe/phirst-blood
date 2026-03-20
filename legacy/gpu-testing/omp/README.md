# RAMBO OpenMP Implementation

High-performance **multi-vendor GPU** implementation of the RAMBO Monte Carlo integrator using OpenMP offloading.

## Overview

This implementation uses OpenMP target offloading to perform parallel Monte Carlo integration on **NVIDIA, AMD, and Intel GPUs**. OpenMP provides a familiar pragma-based approach to GPU programming while maintaining code readability.

## Supported GPU Vendors

| Vendor | Target | Compiler | Auto-Detection |
|--------|--------|----------|----------------|
| **NVIDIA** | nvptx64 | Clang with CUDA | ✅ via nvidia-smi |
| **AMD** | amdgcn | Clang with AMDGPU/AOMP | ✅ via rocminfo |
| **Intel** | spir64 | Intel icpx | ✅ automatic |
| **CPU** | host | Any OpenMP compiler | N/A |

## File Structure

```
omp/
├── CMakeLists.txt      # Multi-vendor GPU build configuration
├── integrator.cpp/h    # OpenMP-based parallel integration
├── main.cpp            # Entry point with benchmark
├── rambo_omp.cpp/h     # Phase space generator
├── rng.cpp/h           # Random number generation
└── README.md           # This file
```

## Prerequisites

- OpenMP-capable compiler with GPU offloading support:
  - **NVIDIA**: Clang built with CUDA support
  - **AMD**: Clang with AMDGPU support, or AOMP
  - **Intel**: Intel oneAPI icpx
  - **CPU**: Any compiler with OpenMP support (GCC, Clang, etc.)
- CMake 3.18+

## Build Instructions

### NVIDIA GPUs
```bash
cd gpu-testing/omp
mkdir build && cd build

# Auto-detect GPU architecture
cmake -DCMAKE_CXX_COMPILER=clang++ -DOMP_TARGET=NVIDIA ..
make

# Or specify architecture manually
cmake -DCMAKE_CXX_COMPILER=clang++ \
      -DOMP_TARGET=NVIDIA \
      -DCUDA_GPU_ARCH=sm_89 ..
make
```

### AMD GPUs
```bash
# Auto-detect GPU architecture
cmake -DCMAKE_CXX_COMPILER=clang++ -DOMP_TARGET=AMD ..
make

# Or specify architecture manually
cmake -DCMAKE_CXX_COMPILER=clang++ \
      -DOMP_TARGET=AMD \
      -DAMD_GPU_ARCH=gfx1100 ..
make
```

### Intel GPUs
```bash
cmake -DCMAKE_CXX_COMPILER=icpx -DOMP_TARGET=INTEL ..
make
```

### CPU-only (no GPU offloading)
```bash
cmake -DOMP_TARGET=CPU ..
make
```

## Target Selection

| Target | Vendor | CMake Flags | Architecture |
|--------|--------|-------------|--------------|
| NVIDIA | NVIDIA | `-DOMP_TARGET=NVIDIA` | `-DCUDA_GPU_ARCH=sm_XX` |
| AMD | AMD | `-DOMP_TARGET=AMD` | `-DAMD_GPU_ARCH=gfxXXXX` |
| INTEL | Intel | `-DOMP_TARGET=INTEL` | Auto-detected |
| CPU | Any | `-DOMP_TARGET=CPU` | N/A |

## Usage

```
./rambo_omp [nEvents] [seed]
```

- `nEvents`: Number of Monte Carlo events (default: 100000)
- `seed`: Random seed for reproducibility (default: 5489)

## GPU Architecture Reference

### NVIDIA
| Architecture | GPUs |
|--------------|------|
| sm_80 | A100, RTX 30xx (GA102) |
| sm_86 | RTX 30xx (GA10x) |
| sm_89 | RTX 40xx, RTX Ada |
| sm_90 | H100 |

### AMD
| Architecture | GPUs |
|--------------|------|
| gfx906 | MI50, Radeon VII |
| gfx908 | MI100 |
| gfx90a | MI210, MI250 |
| gfx1030 | RX 6800, 6900 |
| gfx1100 | RX 7900 |

## Compiler Requirements

OpenMP GPU offloading requires specific compiler configurations:

### NVIDIA (Clang)
```bash
# Clang must be built with CUDA support
clang++ --version
# Should show: ... CUDA support
```

### AMD (Clang/AOMP)
```bash
# AOMP (AMD's OpenMP compiler)
# Or Clang built with AMDGPU support
```

### Intel (icpx)
```bash
# Intel oneAPI DPC++/C++ compiler
icpx --version
```

## Notes

- GPU offloading performance varies significantly based on compiler and driver versions
- For best NVIDIA performance, use recent Clang (15+) with CUDA toolkit
- For AMD, AOMP typically provides better support than upstream Clang
- CPU fallback (`-DOMP_TARGET=CPU`) works with any OpenMP compiler
