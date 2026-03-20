cmake_minimum_required(VERSION 3.18)
project(rambo-gpu LANGUAGES CXX)

# =============================================================================
# Backend Selection
# =============================================================================
# Select exactly ONE backend. All implementations share the rambo:: namespace,
# so only one can be installed at a time.
#
# Usage: cmake -DRAMBO_BACKEND=<backend> ..
# Valid backends: BASE, KOKKOS, CUDA, ALPAKA, SYCL

set(RAMBO_BACKEND "BASE" CACHE STRING "Backend to build: BASE, KOKKOS, CUDA, ALPAKA, SYCL")
set_property(CACHE RAMBO_BACKEND PROPERTY STRINGS BASE KOKKOS CUDA ALPAKA SYCL)

# Normalize to uppercase
string(TOUPPER "${RAMBO_BACKEND}" RAMBO_BACKEND_UPPER)

# =============================================================================
# Validate Backend Selection
# =============================================================================
set(VALID_BACKENDS BASE KOKKOS CUDA ALPAKA SYCL)
if(NOT RAMBO_BACKEND_UPPER IN_LIST VALID_BACKENDS)
    message(FATAL_ERROR 
        "Invalid RAMBO_BACKEND: '${RAMBO_BACKEND}'\n"
        "Valid options: BASE, KOKKOS, CUDA, ALPAKA, SYCL")
endif()

# =============================================================================
# Status Summary
# =============================================================================
message(STATUS "")
message(STATUS "RAMBO GPU Library Configuration:")
message(STATUS "  Selected backend: ${RAMBO_BACKEND_UPPER}")
message(STATUS "")

# =============================================================================
# Add Selected Backend
# =============================================================================
if(RAMBO_BACKEND_UPPER STREQUAL "BASE")
    add_subdirectory(base)
elseif(RAMBO_BACKEND_UPPER STREQUAL "KOKKOS")
    add_subdirectory(kokkos)
elseif(RAMBO_BACKEND_UPPER STREQUAL "CUDA")
    add_subdirectory(cuda)
elseif(RAMBO_BACKEND_UPPER STREQUAL "ALPAKA")
    add_subdirectory(alpaka)
elseif(RAMBO_BACKEND_UPPER STREQUAL "SYCL")
    add_subdirectory(sycl)
endif()
