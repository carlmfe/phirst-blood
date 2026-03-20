#pragma once
#ifndef RAMBO_RANDOM_HPP
#define RAMBO_RANDOM_HPP

/**
 * @file random.hpp
 * @brief Portable random number generation.
 * 
 * Provides a unified XorShift64 RNG that works on all backends.
 * This is a simple, fast PRNG suitable for Monte Carlo simulations.
 * 
 * For Kokkos and Alpaka, consider using their native RNG pools
 * for better performance in production code.
 */

#include "config.hpp"

namespace rambo {

// =============================================================================
// XorShift64 RNG (Portable Implementation)
// =============================================================================

/**
 * XorShift64 random number generator.
 * Simple, fast, and works on all backends including device code.
 */
RAMBO_INLINE auto xorshift64(uint64_t& state) -> uint64_t {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

/**
 * Generate a uniform random double in [0, 1).
 * Uses 53 bits of the 64-bit output for full double precision.
 */
RAMBO_INLINE auto uniformRandom(uint64_t& state) -> double {
    return static_cast<double>(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * Generate a uniform random double in [a, b).
 */
RAMBO_INLINE auto uniformRandom(uint64_t& state, double a, double b) -> double {
    return a + uniformRandom(state) * (b - a);
}

/**
 * Generate a uniform random integer in [0, n).
 */
RAMBO_INLINE auto uniformRandomInt(uint64_t& state, int64_t n) -> int64_t {
    return static_cast<int64_t>(xorshift64(state) % static_cast<uint64_t>(n));
}

/**
 * Initialize RNG state from a seed and thread index.
 * Ensures different threads get different streams.
 */
RAMBO_INLINE auto initRngState(uint64_t seed, int64_t threadIdx) -> uint64_t {
    uint64_t state = seed + static_cast<uint64_t>(threadIdx) * 6364136223846793005ULL;
    // Warm up the RNG
    xorshift64(state);
    xorshift64(state);
    return state;
}

// =============================================================================
// Backend-Specific RNG Wrappers (Optional)
// =============================================================================

#if defined(RAMBO_BACKEND_KOKKOS)

/**
 * Kokkos RNG pool type alias.
 * XorShift64 is a good balance of speed and quality.
 */
using KokkosRngPool = Kokkos::Random_XorShift64_Pool<>;

/**
 * Create a Kokkos RNG pool with the given seed.
 */
inline auto createKokkosRngPool(uint64_t seed) -> KokkosRngPool {
    return KokkosRngPool(seed);
}

#endif // RAMBO_BACKEND_KOKKOS

#if defined(RAMBO_BACKEND_ALPAKA)

/**
 * Alpaka RNG state structure for device code.
 */
template <typename TAcc>
struct AlpakaRng {
    uint64_t state;
    
    ALPAKA_FN_ACC AlpakaRng(TAcc const& acc, uint64_t seed, int64_t threadIdx) {
        (void)acc;
        state = initRngState(seed, threadIdx);
    }
    
    ALPAKA_FN_ACC auto uniform() -> double {
        return uniformRandom(state);
    }
};

#endif // RAMBO_BACKEND_ALPAKA

} // namespace rambo

#endif // RAMBO_RANDOM_HPP
