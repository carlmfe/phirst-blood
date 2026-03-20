#pragma once
#ifndef PHIRST_BACKEND_HPP
#define PHIRST_BACKEND_HPP

/**
 * @file backend.hpp
 * @brief Backend detection and portable macro definitions.
 * 
 * This header provides a unified abstraction layer for multiple compute backends:
 * - SERIAL (default): Standard C++ with no parallelization
 * - CUDA: NVIDIA GPU via CUDA
 * - KOKKOS: Kokkos performance portability library
 * - ALPAKA: Alpaka abstraction library
 * - SYCL: SYCL/DPC++ (Intel, CUDA, or HIP backend)
 * 
 * Usage:
 *   Define exactly one of the following before including this header:
 *   - PHIRST_BACKEND_SERIAL (or none defined)
 *   - PHIRST_BACKEND_CUDA
 *   - PHIRST_BACKEND_KOKKOS
 *   - PHIRST_BACKEND_ALPAKA
 *   - PHIRST_BACKEND_SYCL
 * 
 * This header then provides:
 *   - Function decorators: PHIRST_DEVICE, PHIRST_HOST, PHIRST_HOST_DEVICE, PHIRST_INLINE
 *   - Backend identification macros
 *   - Common includes for each backend
 */

// =============================================================================
// Backend Detection and Validation
// =============================================================================

// Count how many backends are defined
#define PHIRST_BACKEND_COUNT 0

#if defined(PHIRST_BACKEND_CUDA)
    #undef PHIRST_BACKEND_COUNT
    #define PHIRST_BACKEND_COUNT 1
#endif

#if defined(PHIRST_BACKEND_KOKKOS)
    #if PHIRST_BACKEND_COUNT > 0
        #error "Multiple PHIRST backends defined. Define only one of: PHIRST_BACKEND_SERIAL, PHIRST_BACKEND_CUDA, PHIRST_BACKEND_KOKKOS, PHIRST_BACKEND_ALPAKA, PHIRST_BACKEND_SYCL"
    #endif
    #undef PHIRST_BACKEND_COUNT
    #define PHIRST_BACKEND_COUNT 1
#endif

#if defined(PHIRST_BACKEND_ALPAKA)
    #if PHIRST_BACKEND_COUNT > 0
        #error "Multiple PHIRST backends defined. Define only one of: PHIRST_BACKEND_SERIAL, PHIRST_BACKEND_CUDA, PHIRST_BACKEND_KOKKOS, PHIRST_BACKEND_ALPAKA, PHIRST_BACKEND_SYCL"
    #endif
    #undef PHIRST_BACKEND_COUNT
    #define PHIRST_BACKEND_COUNT 1
#endif

#if defined(PHIRST_BACKEND_SYCL)
    #if PHIRST_BACKEND_COUNT > 0
        #error "Multiple PHIRST backends defined. Define only one of: PHIRST_BACKEND_SERIAL,PHIRST_BACKEND_CUDA,PHIRSTO_BACKEND_KOKKOS, PHIRST_BACKEND_ALPAKA, PHIRST_BACKEND_SYCL"
    #endif
    #undef PHIRST_BACKEND_COUNT
    #define PHIRST_BACKEND_COUNT 1
#endif

// Default to SERIAL if no backend specified
#if !defined(PHIRST_BACKEND_CUDA) && !defined(PHIRST_BACKEND_KOKKOS) && \
    !defined(PHIRST_BACKEND_ALPAKA) && !defined(PHIRST_BACKEND_SYCL) && \
    !defined(PHIRST_BACKEND_SERIAL)
    #define PHIRST_BACKEND_SERIAL
#endif

// =============================================================================
// Backend-Specific Includes
// =============================================================================

#if defined(PHIRST_BACKEND_CUDA)
    #include <cuda_runtime.h>
    #include <cmath>
    #include <cstdint>
#elif defined(PHIRST_BACKEND_KOKKOS)
    #include <Kokkos_Core.hpp>
    #include <Kokkos_Random.hpp>
#elif defined(PHIRST_BACKEND_ALPAKA)
    #include <alpaka/alpaka.hpp>
    #include <cmath>
    #include <cstdint>
#elif defined(PHIRST_BACKEND_SYCL)
    #include <sycl/sycl.hpp>
    #include <cstdint>
#else // SERIAL
    #include <cmath>
    #include <cstdint>
#endif

// Common includes for all backends
#include <limits>

// =============================================================================
// Function Decorators
// =============================================================================

/**
 * PHIRST_DEVICE - Marks a function as callable from device code only.
 * PHIRST_HOST - Marks a function as callable from host code only.
 * PHIRST_HOST_DEVICE - Marks a function as callable from both host and device.
 * PHIRST_INLINE - Suggests inlining (device-aware).
 * PHIRST_FORCEINLINE - Forces inlining where supported.
 */

#if defined(PHIRST_BACKEND_CUDA)
    #define PHIRST_DEVICE __device__
    #define PHIRST_HOST __host__
    #define PHIRST_HOST_DEVICE __host__ __device__
    #define PHIRST_INLINE __device__ __host__ inline
    #define PHIRST_FORCEINLINE __device__ __host__ __forceinline__

#elif defined(PHIRST_BACKEND_KOKKOS)
    #define PHIRST_DEVICE KOKKOS_FUNCTION
    #define PHIRST_HOST KOKKOS_FUNCTION
    #define PHIRST_HOST_DEVICE KOKKOS_FUNCTION
    #define PHIRST_INLINE KOKKOS_INLINE_FUNCTION
    #define PHIRST_FORCEINLINE KOKKOS_FORCEINLINE_FUNCTION

#elif defined(PHIRST_BACKEND_ALPAKA)
    #define PHIRST_DEVICE ALPAKA_FN_ACC
    #define PHIRST_HOST ALPAKA_FN_HOST
    #define PHIRST_HOST_DEVICE ALPAKA_FN_HOST_ACC
    #define PHIRST_INLINE ALPAKA_FN_HOST_ACC inline
    #define PHIRST_FORCEINLINE ALPAKA_FN_HOST_ACC inline

#elif defined(PHIRST_BACKEND_SYCL)
    // SYCL uses regular functions within kernels; no special decorators needed
    #define PHIRST_DEVICE
    #define PHIRST_HOST
    #define PHIRST_HOST_DEVICE
    #define PHIRST_INLINE inline
    #define PHIRST_FORCEINLINE inline

#else // SERIAL
    #define PHIRST_DEVICE
    #define PHIRST_HOST
    #define PHIRST_HOST_DEVICE
    #define PHIRST_INLINE inline
    #define PHIRST_FORCEINLINE inline

#endif

// =============================================================================
// Backend Name String (for runtime identification)
// =============================================================================

namespace phirst {

#if defined(PHIRST_BACKEND_CUDA)
    inline constexpr const char* BACKEND_NAME = "CUDA";
#elif defined(PHIRST_BACKEND_KOKKOS)
    inline constexpr const char* BACKEND_NAME = "Kokkos";
#elif defined(PHIRST_BACKEND_ALPAKA)
    inline constexpr const char* BACKEND_NAME = "Alpaka";
#elif defined(PHIRST_BACKEND_SYCL)
    inline constexpr const char* BACKEND_NAME = "SYCL";
#else
    inline constexpr const char* BACKEND_NAME = "Serial";
#endif

} // namespace phirst

// =============================================================================
// Static Assertions for Sanity Checks
// =============================================================================

#if defined(PHIRST_BACKEND_KOKKOS)
    static_assert(__cplusplus >= 201703L, "Kokkos backend requires C++17 or later");
#elif defined(PHIRST_BACKEND_ALPAKA)
    static_assert(__cplusplus >= 201703L, "Alpaka backend requires C++17 or later");
#elif defined(PHIRST_BACKEND_SYCL)
    static_assert(__cplusplus >= 201703L, "SYCL backend requires C++17 or later");
#endif

#endif // PHIRST_BACKEND_HPP
