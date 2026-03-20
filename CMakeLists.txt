cmake_minimum_required(VERSION 3.18)
project("RamboOne" CXX)

# =============================================================================
# Backend Selection
# =============================================================================
# Usage: cmake -DRAMBO_BACKEND=CUDA ..
# Valid backends: SERIAL, CUDA, KOKKOS, ALPAKA, SYCL
# Default: SERIAL

set(RAMBO_BACKEND "SERIAL" CACHE STRING "Backend: SERIAL, CUDA, KOKKOS, ALPAKA, SYCL")
set_property(CACHE RAMBO_BACKEND PROPERTY STRINGS SERIAL CUDA KOKKOS ALPAKA SYCL)

message(STATUS "RAMBO Backend: ${RAMBO_BACKEND}")

# =============================================================================
# Common Settings
# =============================================================================

# Default to Release build for good performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source files
set(SOURCES examples/drell_yan.cpp)

# =============================================================================
# Backend-Specific Configuration
# =============================================================================

if(RAMBO_BACKEND STREQUAL "SERIAL")
    # ---------------------------
    # Serial CPU Backend
    # ---------------------------
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    add_executable(ex_drell_yan ${SOURCES})
    target_include_directories(ex_drell_yan PRIVATE include)
    target_compile_definitions(ex_drell_yan PRIVATE RAMBO_BACKEND_SERIAL)
    
elseif(RAMBO_BACKEND STREQUAL "CUDA")
    # ---------------------------
    # CUDA Backend
    # ---------------------------
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found. Use -DRAMBO_BACKEND=SERIAL for CPU build.")
    endif()
    enable_language(CUDA)
    
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
    
    # For CUDA, rename drell_yan.cpp to drell_yan.cu
    set_source_files_properties(examples/drell_yan.cpp PROPERTIES LANGUAGE CUDA)
    
    add_executable(ex_drell_yan ${SOURCES})
    target_include_directories(ex_drell_yan PRIVATE include)
    target_compile_definitions(ex_drell_yan PRIVATE RAMBO_BACKEND_CUDA)
    target_compile_options(ex_drell_yan PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
    
elseif(RAMBO_BACKEND STREQUAL "KOKKOS")
    # ---------------------------
    # Kokkos Backend
    # ---------------------------
    # Do NOT set CMAKE_CXX_STANDARD before find_package(Kokkos)!
    # Kokkos sets its own compiler flags based on how it was built.
    
    find_package(Kokkos REQUIRED)
    
    add_executable(ex_drell_yan ${SOURCES})
    target_include_directories(ex_drell_yan PRIVATE include)
    target_compile_definitions(ex_drell_yan PRIVATE RAMBO_BACKEND_KOKKOS)
    target_link_libraries(ex_drell_yan PRIVATE Kokkos::kokkos)
    
elseif(RAMBO_BACKEND STREQUAL "ALPAKA")
    # ---------------------------
    # Alpaka Backend
    # ---------------------------
    set(ALPAKA_BACKEND "CPU" CACHE STRING "Alpaka sub-backend: CUDA, CPU, OMP")
    set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS CUDA CPU OMP)
    
    message(STATUS "Alpaka sub-backend: ${ALPAKA_BACKEND}")
    
    include(CheckLanguage)
    check_language(CUDA)
    
    if(ALPAKA_BACKEND STREQUAL "CUDA" AND CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
        set(alpaka_ACC_GPU_CUDA_ENABLE ON)
        add_compile_definitions(ALPAKA_USE_CUDA)
    elseif(ALPAKA_BACKEND STREQUAL "OMP")
        find_package(OpenMP REQUIRED)
        set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON)
        add_compile_definitions(ALPAKA_USE_CPU_OMP)
    else()
        set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
        add_compile_definitions(ALPAKA_USE_CPU_SERIAL)
    endif()
    
    find_package(alpaka REQUIRED)
    
    alpaka_add_executable(ex_drell_yan ${SOURCES})
    target_include_directories(ex_drell_yan PRIVATE include)
    target_compile_definitions(ex_drell_yan PRIVATE RAMBO_BACKEND_ALPAKA)
    target_link_libraries(ex_drell_yan PRIVATE alpaka::alpaka)
    
elseif(RAMBO_BACKEND STREQUAL "SYCL")
    # ---------------------------
    # SYCL Backend
    # ---------------------------
    # Requires SYCL-enabled compiler (e.g., Intel DPC++, AdaptiveCpp)
    # Use: cmake -DCMAKE_CXX_COMPILER=clang++ -DRAMBO_BACKEND=SYCL ..
    
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    set(SYCL_BACKEND "CUDA" CACHE STRING "SYCL device backend: CUDA, INTEL")
    set_property(CACHE SYCL_BACKEND PROPERTY STRINGS CUDA INTEL)
    
    message(STATUS "SYCL device backend: ${SYCL_BACKEND}")
    
    add_executable(ex_drell_yan ${SOURCES})
    target_include_directories(ex_drell_yan PRIVATE include)
    target_compile_definitions(ex_drell_yan PRIVATE RAMBO_BACKEND_SYCL)
    
    if(SYCL_BACKEND STREQUAL "CUDA")
        set(CUDA_GPU_ARCH "sm_70" CACHE STRING "CUDA GPU architecture")
        target_compile_options(ex_drell_yan PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
            -Xsycl-target-backend
            --cuda-gpu-arch=${CUDA_GPU_ARCH}
        )
        target_link_options(ex_drell_yan PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
    else()
        target_compile_options(ex_drell_yan PRIVATE -fsycl)
        target_link_options(ex_drell_yan PRIVATE -fsycl)
    endif()
    
else()
    message(FATAL_ERROR "Unknown backend: ${RAMBO_BACKEND}. Use SERIAL, CUDA, KOKKOS, ALPAKA, or SYCL.")
endif()

# =============================================================================
# Installation (Optional)
# =============================================================================

include(GNUInstallDirs)

# Install headers
install(DIRECTORY include/rambo
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
