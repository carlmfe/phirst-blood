cmake_minimum_required(VERSION 3.18)
project(rambo_omp CXX)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check for C++20 support (OpenMP offloading requires Clang or GCC with offload support)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Clang 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "GCC 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# =============================================================================
# GPU Target Selection (Multi-Vendor)
# =============================================================================
# OpenMP offloading can target NVIDIA, AMD, or Intel GPUs depending on compiler.
#
# For NVIDIA GPUs (Clang with CUDA support):
#   cmake -DOMP_TARGET=NVIDIA ..
#
# For AMD GPUs (Clang with ROCm/AMDGPU support, or AOMP):
#   cmake -DOMP_TARGET=AMD ..
#
# For Intel GPUs (Intel oneAPI compiler icpx):
#   cmake -DOMP_TARGET=INTEL ..
#
# For CPU-only OpenMP (no offloading):
#   cmake -DOMP_TARGET=CPU ..
# =============================================================================
set(OMP_TARGET "NVIDIA" CACHE STRING "OpenMP target: NVIDIA, AMD, INTEL, CPU")
set_property(CACHE OMP_TARGET PROPERTY STRINGS NVIDIA AMD INTEL CPU)
message(STATUS "OpenMP target: ${OMP_TARGET}")

# =============================================================================
# GPU Architecture Detection (Multi-Vendor)
# =============================================================================
# NVIDIA: sm_80, sm_86, sm_89, sm_90
# AMD: gfx906, gfx908, gfx90a, gfx1030, gfx1100
# Intel: auto-detected
# =============================================================================

# NVIDIA GPU architecture
set(CUDA_GPU_ARCH "native" CACHE STRING "NVIDIA GPU architecture: native (auto-detect), or sm_XX")

# AMD GPU architecture
set(AMD_GPU_ARCH "native" CACHE STRING "AMD GPU architecture: native (auto-detect), or gfxXXXX")

# Auto-detect NVIDIA GPU architecture
if(CUDA_GPU_ARCH STREQUAL "native" AND OMP_TARGET STREQUAL "NVIDIA")
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_COMPUTE_CAP
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NVIDIA_SMI_RESULT
        )
        if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")
            string(REPLACE "." "" GPU_ARCH_NUM "${GPU_COMPUTE_CAP}")
            set(CUDA_GPU_ARCH "sm_${GPU_ARCH_NUM}")
            message(STATUS "Auto-detected NVIDIA GPU: ${GPU_COMPUTE_CAP} -> ${CUDA_GPU_ARCH}")
        else()
            message(WARNING "Failed to detect NVIDIA GPU, using default sm_80")
            set(CUDA_GPU_ARCH "sm_80")
        endif()
    else()
        message(WARNING "nvidia-smi not found, using default sm_80")
        set(CUDA_GPU_ARCH "sm_80")
    endif()
endif()

# Auto-detect AMD GPU architecture
if(AMD_GPU_ARCH STREQUAL "native" AND OMP_TARGET STREQUAL "AMD")
    find_program(ROCMINFO rocminfo)
    if(ROCMINFO)
        execute_process(
            COMMAND rocminfo
            OUTPUT_VARIABLE ROCM_INFO_OUTPUT
            ERROR_QUIET
            RESULT_VARIABLE ROCMINFO_RESULT
        )
        if(ROCMINFO_RESULT EQUAL 0)
            string(REGEX MATCH "gfx[0-9a-z]+" AMD_GPU_ARCH "${ROCM_INFO_OUTPUT}")
            if(AMD_GPU_ARCH)
                message(STATUS "Auto-detected AMD GPU: ${AMD_GPU_ARCH}")
            else()
                message(WARNING "Failed to parse AMD GPU architecture, using default gfx906")
                set(AMD_GPU_ARCH "gfx906")
            endif()
        else()
            message(WARNING "rocminfo failed, using default gfx906")
            set(AMD_GPU_ARCH "gfx906")
        endif()
    else()
        message(WARNING "rocminfo not found, using default gfx906")
        set(AMD_GPU_ARCH "gfx906")
    endif()
endif()

# =============================================================================
# OpenMP Offloading Flags (Target-Specific)
# =============================================================================
if(OMP_TARGET STREQUAL "NVIDIA")
    # NVIDIA GPU offloading (requires Clang with CUDA support)
    message(STATUS "Configuring for NVIDIA GPU offloading")
    message(STATUS "GPU architecture: ${CUDA_GPU_ARCH}")
    
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(WARNING "NVIDIA OpenMP offloading works best with Clang. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()
    
    set(OPENMP_FLAGS
        -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        -Xopenmp-target=nvptx64-nvidia-cuda -march=${CUDA_GPU_ARCH}
    )
    
elseif(OMP_TARGET STREQUAL "AMD")
    # AMD GPU offloading (requires Clang with AMDGPU support or AOMP)
    message(STATUS "Configuring for AMD GPU offloading")
    message(STATUS "GPU architecture: ${AMD_GPU_ARCH}")
    
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(WARNING "AMD OpenMP offloading works best with Clang/AOMP. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()
    
    set(OPENMP_FLAGS
        -fopenmp
        -fopenmp-targets=amdgcn-amd-amdhsa
        -Xopenmp-target=amdgcn-amd-amdhsa -march=${AMD_GPU_ARCH}
    )
    
elseif(OMP_TARGET STREQUAL "INTEL")
    # Intel GPU offloading (requires Intel oneAPI icpx compiler)
    message(STATUS "Configuring for Intel GPU offloading")
    
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        message(WARNING "Intel OpenMP offloading requires icpx compiler. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()
    
    set(OPENMP_FLAGS
        -fiopenmp
        -fopenmp-targets=spir64
    )
    
elseif(OMP_TARGET STREQUAL "CPU")
    # CPU-only OpenMP (no offloading)
    message(STATUS "Configuring for CPU-only OpenMP (no GPU offloading)")
    
    set(OPENMP_FLAGS
        -fopenmp
    )
    
else()
    message(FATAL_ERROR "Unknown OMP_TARGET: ${OMP_TARGET}. Use NVIDIA, AMD, INTEL, or CPU.")
endif()

# =============================================================================
# Build executable
# =============================================================================
add_executable(rambo_omp
    main.cpp
    integrator.cpp
    rambo_omp.cpp
    rng.cpp
)

target_compile_options(rambo_omp PRIVATE ${OPENMP_FLAGS})
target_link_options(rambo_omp PRIVATE ${OPENMP_FLAGS})

# =============================================================================
# Print configuration
# =============================================================================
message(STATUS "")
message(STATUS "=== OpenMP Build Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP Target: ${OMP_TARGET}")
if(OMP_TARGET STREQUAL "NVIDIA")
    message(STATUS "GPU Architecture: ${CUDA_GPU_ARCH}")
elseif(OMP_TARGET STREQUAL "AMD")
    message(STATUS "GPU Architecture: ${AMD_GPU_ARCH}")
endif()
message(STATUS "==================================")
