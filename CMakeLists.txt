cmake_minimum_required(VERSION 3.18)
project("Phirst Blood" CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PhirstDetect)

# =============================================================================
# Backend Selection
# =============================================================================
# Usage: cmake -DPHIRST_BACKEND=CUDA ..
# Valid backends: AUTO, SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, HIP
# Default: AUTO (probes for GPU toolchains in order: CUDA > KOKKOS/HIP > SYCL > SERIAL)

set(PHIRST_BACKEND "AUTO" CACHE STRING "Backend: AUTO, SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, HIP")
set_property(CACHE PHIRST_BACKEND PROPERTY STRINGS AUTO SERIAL CUDA KOKKOS ALPAKA SYCL HIP)

# GPU architecture override (optional).
# CUDA/Alpaka:  single SM number or semicolon-separated list, e.g. "89" or "80;89"
# SYCL/CUDA:    full arch string, e.g. "sm_89"
# HIP:          GFX string or semicolon-separated list, e.g. "gfx1100" or "gfx908;gfx90a"
# Leave empty to let each backend auto-detect (default behaviour).
set(PHIRST_GPU_ARCH "" CACHE STRING "Override GPU architecture (e.g. 89 for CUDA/Alpaka, sm_89 for SYCL, gfx1100 for HIP)")

# --- AUTO detection ---
if(PHIRST_BACKEND STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(PHIRST_BACKEND "CUDA")
        message(STATUS "AUTO: CUDA compiler found → selecting CUDA backend")
    else()
        find_program(_ROCM_AGENT rocm_agent_enumerator
            PATHS $ENV{ROCM_PATH}/bin)
        find_program(_ROCMINFO rocminfo
            PATHS $ENV{ROCM_PATH}/bin)
        if(_ROCM_AGENT OR _ROCMINFO)
            # ROCm present — prefer Kokkos, then native HIP, then SYCL/AMD
            find_package(Kokkos QUIET)
            if(Kokkos_FOUND)
                set(PHIRST_BACKEND "KOKKOS")
                message(STATUS "AUTO: ROCm + Kokkos found → selecting KOKKOS backend")
            else()
                check_language(HIP)
                if(CMAKE_HIP_COMPILER)
                    set(PHIRST_BACKEND "HIP")
                    message(STATUS "AUTO: ROCm + HIP compiler found → selecting HIP backend")
                else()
                    set(PHIRST_BACKEND "SYCL")
                    set(SYCL_BACKEND "AMD" CACHE STRING "SYCL device backend (auto-set)" FORCE)
                    message(STATUS "AUTO: ROCm found (no Kokkos, no HIP compiler) → selecting SYCL/AMD backend")
                endif()
            endif()
        else()
            find_program(_SYCL_LS sycl-ls)
            if(_SYCL_LS)
                execute_process(COMMAND ${_SYCL_LS}
                    OUTPUT_VARIABLE _sycl_out OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                if(_sycl_out MATCHES "[Ii]ntel")
                    set(PHIRST_BACKEND "SYCL")
                    set(SYCL_BACKEND "INTEL" CACHE STRING "SYCL device backend (auto-set)" FORCE)
                    message(STATUS "AUTO: Intel GPU found via sycl-ls → selecting SYCL/Intel backend")
                else()
                    set(PHIRST_BACKEND "SERIAL")
                    message(STATUS "AUTO: sycl-ls found but no Intel GPU — falling back to SERIAL")
                endif()
            else()
                set(PHIRST_BACKEND "SERIAL")
                message(STATUS "AUTO: No GPU toolchain found → selecting SERIAL backend")
            endif()
        endif()
    endif()
endif()

message(STATUS "PHIRST Backend: ${PHIRST_BACKEND}")

# Name of examples based on backend
string(TOLOWER "ex_drell_yan_${PHIRST_BACKEND}" EXENAME_DY)
string(TOLOWER "ex_eggholder_${PHIRST_BACKEND}" EXENAME_EGG)

# =============================================================================
# Common Settings
# =============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Default to Release build for good performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Test suite (disabled by default; enable with -DPHIRST_BUILD_TESTS=ON)
option(PHIRST_BUILD_TESTS "Build the GoogleTest suite alongside the examples" OFF)
if(PHIRST_BUILD_TESTS)
    enable_testing()
endif()

# Examples: ON when building as top-level project, OFF when used as subdirectory
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(_phirst_default_examples ON)
else()
    set(_phirst_default_examples OFF)
endif()
option(PHIRST_BUILD_EXAMPLES "Build example executables" ${_phirst_default_examples})
option(PHIRST_NUMBA_BRIDGE "Build the Numba/cuLink bridge for CUDA Python device integrands" OFF)
option(SYCL_INTEL_AOT "Enable Intel SYCL AOT compilation (requires ocloc)" OFF)

# Variables for package config template (populated per-backend below)
set(PHIRST_SYCL_BACKEND "")
set(PHIRST_SYCL_CUDA_ARCH "")
set(PHIRST_SYCL_AMD_ARCH "")

# Source files
set(SOURCES_DY examples/drell_yan.cpp)
set(SOURCES_EGG examples/eggholder.cpp)

# =============================================================================
# Backend-Specific Configuration
# =============================================================================

if(PHIRST_BACKEND STREQUAL "SERIAL")
    include(backend/Serial)
elseif(PHIRST_BACKEND STREQUAL "CUDA")
    include(backend/CUDA)
elseif(PHIRST_BACKEND STREQUAL "KOKKOS")
    include(backend/Kokkos)
elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    include(backend/Alpaka)
elseif(PHIRST_BACKEND STREQUAL "SYCL")
    include(backend/SYCL)
elseif(PHIRST_BACKEND STREQUAL "HIP")
    include(backend/HIP)
else()
    message(FATAL_ERROR "Unknown backend: ${PHIRST_BACKEND}. Use SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, or HIP.")
endif()

include(PhirstInstall)
