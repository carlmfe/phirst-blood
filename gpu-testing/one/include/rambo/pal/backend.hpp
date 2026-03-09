#pragma once
#ifndef RAMBO_BACKEND_HPP
#define RAMBO_BACKEND_HPP

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
 *   - RAMBO_BACKEND_SERIAL (or none defined)
 *   - RAMBO_BACKEND_CUDA
 *   - RAMBO_BACKEND_KOKKOS
 *   - RAMBO_BACKEND_ALPAKA
 *   - RAMBO_BACKEND_SYCL
 * 
 * This header then provides:
 *   - Function decorators: RAMBO_DEVICE, RAMBO_HOST, RAMBO_HOST_DEVICE, RAMBO_INLINE
 *   - Backend identification macros
 *   - Common includes for each backend
 */

// =============================================================================
// Backend Detection and Validation
// =============================================================================

// Count how many backends are defined
#define RAMBO_BACKEND_COUNT 0

#if defined(RAMBO_BACKEND_CUDA)
    #undef RAMBO_BACKEND_COUNT
    #define RAMBO_BACKEND_COUNT 1
#endif

#if defined(RAMBO_BACKEND_KOKKOS)
    #if RAMBO_BACKEND_COUNT > 0
        #error "Multiple RAMBO backends defined. Define only one of: RAMBO_BACKEND_SERIAL, RAMBO_BACKEND_CUDA, RAMBO_BACKEND_KOKKOS, RAMBO_BACKEND_ALPAKA, RAMBO_BACKEND_SYCL"
    #endif
    #undef RAMBO_BACKEND_COUNT
    #define RAMBO_BACKEND_COUNT 1
#endif

#if defined(RAMBO_BACKEND_ALPAKA)
    #if RAMBO_BACKEND_COUNT > 0
        #error "Multiple RAMBO backends defined. Define only one of: RAMBO_BACKEND_SERIAL, RAMBO_BACKEND_CUDA, RAMBO_BACKEND_KOKKOS, RAMBO_BACKEND_ALPAKA, RAMBO_BACKEND_SYCL"
    #endif
    #undef RAMBO_BACKEND_COUNT
    #define RAMBO_BACKEND_COUNT 1
#endif

#if defined(RAMBO_BACKEND_SYCL)
    #if RAMBO_BACKEND_COUNT > 0
        #error "Multiple RAMBO backends defined. Define only one of: RAMBO_BACKEND_SERIAL, RAMBO_BACKEND_CUDA, RAMBO_BACKEND_KOKKOS, RAMBO_BACKEND_ALPAKA, RAMBO_BACKEND_SYCL"
    #endif
    #undef RAMBO_BACKEND_COUNT
    #define RAMBO_BACKEND_COUNT 1
#endif

// Default to SERIAL if no backend specified
#if !defined(RAMBO_BACKEND_CUDA) && !defined(RAMBO_BACKEND_KOKKOS) && \
    !defined(RAMBO_BACKEND_ALPAKA) && !defined(RAMBO_BACKEND_SYCL) && \
    !defined(RAMBO_BACKEND_SERIAL)
    #define RAMBO_BACKEND_SERIAL
#endif

// =============================================================================
// Backend-Specific Includes
// =============================================================================

#if defined(RAMBO_BACKEND_CUDA)
    #include <cuda_runtime.h>
    #include <cmath>
    #include <cstdint>
#elif defined(RAMBO_BACKEND_KOKKOS)
    #include <Kokkos_Core.hpp>
    #include <Kokkos_Random.hpp>
#elif defined(RAMBO_BACKEND_ALPAKA)
    #include <alpaka/alpaka.hpp>
    #include <cmath>
    #include <cstdint>
#elif defined(RAMBO_BACKEND_SYCL)
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
 * RAMBO_DEVICE - Marks a function as callable from device code only.
 * RAMBO_HOST - Marks a function as callable from host code only.
 * RAMBO_HOST_DEVICE - Marks a function as callable from both host and device.
 * RAMBO_INLINE - Suggests inlining (device-aware).
 * RAMBO_FORCEINLINE - Forces inlining where supported.
 */

#if defined(RAMBO_BACKEND_CUDA)
    #define RAMBO_DEVICE __device__
    #define RAMBO_HOST __host__
    #define RAMBO_HOST_DEVICE __host__ __device__
    #define RAMBO_INLINE __device__ __host__ inline
    #define RAMBO_FORCEINLINE __device__ __host__ __forceinline__

#elif defined(RAMBO_BACKEND_KOKKOS)
    #define RAMBO_DEVICE KOKKOS_FUNCTION
    #define RAMBO_HOST KOKKOS_FUNCTION
    #define RAMBO_HOST_DEVICE KOKKOS_FUNCTION
    #define RAMBO_INLINE KOKKOS_INLINE_FUNCTION
    #define RAMBO_FORCEINLINE KOKKOS_FORCEINLINE_FUNCTION

#elif defined(RAMBO_BACKEND_ALPAKA)
    #define RAMBO_DEVICE ALPAKA_FN_ACC
    #define RAMBO_HOST ALPAKA_FN_HOST
    #define RAMBO_HOST_DEVICE ALPAKA_FN_HOST_ACC
    #define RAMBO_INLINE ALPAKA_FN_HOST_ACC inline
    #define RAMBO_FORCEINLINE ALPAKA_FN_HOST_ACC inline

#elif defined(RAMBO_BACKEND_SYCL)
    // SYCL uses regular functions within kernels; no special decorators needed
    #define RAMBO_DEVICE
    #define RAMBO_HOST
    #define RAMBO_HOST_DEVICE
    #define RAMBO_INLINE inline
    #define RAMBO_FORCEINLINE inline

#else // SERIAL
    #define RAMBO_DEVICE
    #define RAMBO_HOST
    #define RAMBO_HOST_DEVICE
    #define RAMBO_INLINE inline
    #define RAMBO_FORCEINLINE inline

#endif

// =============================================================================
// Backend Name String (for runtime identification)
// =============================================================================

namespace rambo {

#if defined(RAMBO_BACKEND_CUDA)
    inline constexpr const char* BACKEND_NAME = "CUDA";
#elif defined(RAMBO_BACKEND_KOKKOS)
    inline constexpr const char* BACKEND_NAME = "Kokkos";
#elif defined(RAMBO_BACKEND_ALPAKA)
    inline constexpr const char* BACKEND_NAME = "Alpaka";
#elif defined(RAMBO_BACKEND_SYCL)
    inline constexpr const char* BACKEND_NAME = "SYCL";
#else
    inline constexpr const char* BACKEND_NAME = "Serial";
#endif

} // namespace rambo

// =============================================================================
// Static Assertions for Sanity Checks
// =============================================================================

#if defined(RAMBO_BACKEND_KOKKOS)
    static_assert(__cplusplus >= 201703L, "Kokkos backend requires C++17 or later");
#elif defined(RAMBO_BACKEND_ALPAKA)
    static_assert(__cplusplus >= 201703L, "Alpaka backend requires C++17 or later");
#elif defined(RAMBO_BACKEND_SYCL)
    static_assert(__cplusplus >= 201703L, "SYCL backend requires C++17 or later");
#endif

#endif // RAMBO_BACKEND_HPP
