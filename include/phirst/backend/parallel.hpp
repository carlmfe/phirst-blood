#pragma once
#ifndef PHIRST_PARALLEL_HPP
#define PHIRST_PARALLEL_HPP

/**
 * @file parallel.hpp
 * @brief Portable parallel execution abstractions.
 * 
 * Provides low-level, backend-agnostic primitives for parallel execution.
 * All backends provide the same set of functions with consistent signatures.
 * 
 * Key abstractions:
 * - DeviceBuffer<T>: Manages device memory allocation
 * - deep_copy(): Memory transfer between host and device
 * - fill_buffer(): Initialize buffer with value
 * - fence(): Synchronization barrier
 * - GridConfig: Thread grid configuration helper
 * - seed_for_thread(): RNG seeding helper
 * - grid_stride_reduce(): Generic parallel reduction primitive
 */

#include "config.hpp"
#include <cstdint>

namespace phirst {

// =============================================================================
// Grid Configuration
// =============================================================================

struct GridConfig {
    int64_t totalThreads;
    int64_t blockSize;
    int64_t numBlocks;

    static GridConfig compute(int64_t nWork, int64_t blockSz = 256, int64_t maxBlks = 1024) {
        GridConfig cfg;
        cfg.blockSize = blockSz;
        cfg.numBlocks = (nWork + blockSz - 1) / blockSz;
        if (cfg.numBlocks > maxBlks) cfg.numBlocks = maxBlks;
        cfg.totalThreads = cfg.numBlocks * cfg.blockSize;
        return cfg;
    }
};

// =============================================================================
// RNG Seeding Helper
// =============================================================================

PHIRST_HOST_DEVICE
inline uint64_t seed_for_thread(uint64_t baseSeed, int64_t threadIdx) {
    uint64_t seed = baseSeed ^ (static_cast<uint64_t>(threadIdx) * 2685821657736338717ULL);
    return (seed == 0) ? baseSeed + 1 : seed;
}

// =============================================================================
// host_reduce - Host-side Reduction Helper
// =============================================================================

template <typename T>
T host_reduce(const T* data, int64_t n) {
    T sum = T{};
    for (int64_t i = 0; i < n; ++i) sum += data[i];
    return sum;
}

} // namespace phirst

// =============================================================================
// Backend Implementation Inclusions
// =============================================================================

#if defined(PHIRST_BACKEND_SERIAL)
    #include "parallel_serial.hpp"
#elif defined(PHIRST_BACKEND_KOKKOS)
    #include "parallel_kokkos.hpp"
#elif defined(PHIRST_BACKEND_SYCL)
    #include "parallel_sycl.hpp"
#elif defined(PHIRST_BACKEND_ALPAKA)
    #include "parallel_alpaka.hpp"
#elif defined(PHIRST_BACKEND_CUDA)
    #include "parallel_cuda.hpp"
#endif

namespace phirst {

// =============================================================================
// Active Backend Alias
// =============================================================================

#if defined(PHIRST_BACKEND_SERIAL)
    namespace backend_impl = serial_impl;
#elif defined(PHIRST_BACKEND_CUDA)
    namespace backend_impl = cuda_impl;
#elif defined(PHIRST_BACKEND_KOKKOS)
    namespace backend_impl = kokkos_impl;
#elif defined(PHIRST_BACKEND_SYCL)
    namespace backend_impl = sycl_impl;
#elif defined(PHIRST_BACKEND_ALPAKA)
    namespace backend_impl = alpaka_impl;
#else
    #error "No valid PHIRST backend defined for parallel.hpp"
#endif

// =============================================================================
// Public Interface Aliases
// =============================================================================

// Execution Space and Kernel Acc
#if !defined(PHIRST_BACKEND_ALPAKA)
    using KernelAcc = backend_impl::KernelAcc;
#else
    using KernelAcc = backend_impl::AlpakaAcc;
#endif

#if defined(PHIRST_BACKEND_SERIAL)
    using DefaultSpace = backend_impl::HostSpace;
#else
    struct DeviceSpace {};
    using DefaultSpace = DeviceSpace;
#endif

// DeviceBuffer
template <typename T>
using DeviceBuffer = backend_impl::DeviceBuffer<T>;

// Functions
using backend_impl::deep_copy;
using backend_impl::fill_buffer;
using backend_impl::fence;
using backend_impl::atomic_add;
using backend_impl::run_single_thread;
using backend_impl::grid_stride_reduce;

} // namespace phirst

#endif // PHIRST_PARALLEL_HPP
