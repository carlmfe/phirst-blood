#pragma once
#ifndef RAMBO_HPP
#define RAMBO_HPP

// =============================================================================
// RAMBO - Portable Phase Space Generator Library
// =============================================================================
// Main include file - includes all library components.
//
// This unified library supports multiple backends:
//   - RAMBO_BACKEND_SERIAL  : CPU serial (default)
//   - RAMBO_BACKEND_CUDA    : NVIDIA CUDA
//   - RAMBO_BACKEND_KOKKOS  : Kokkos (CUDA/CPU)
//   - RAMBO_BACKEND_ALPAKA  : Alpaka (CUDA/CPU/OpenMP)
//   - RAMBO_BACKEND_SYCL    : SYCL (CUDA/Intel)
//
// Usage:
//   #include <rambo/rambo.hpp>
//
//   rambo::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
//   rambo::RamboIntegrator<rambo::DrellYanIntegrand, 2> integrator(nEvents, integrand);
//   
//   double mean, error;
//   integrator.run(cmEnergy, masses, mean, error, seed);
//
// Build with backend selection:
//   cmake -DRAMBO_BACKEND=KOKKOS ..
//   cmake -DRAMBO_BACKEND=CUDA ..
//   cmake -DRAMBO_BACKEND=SYCL -DCMAKE_CXX_COMPILER=clang++ ..
// =============================================================================

#include "backend/config.hpp"
#include "backend/math.hpp"
#include "backend/random.hpp"
#include "phase_space.hpp"
#include "integrands.hpp"
#include "integrator.hpp"

namespace rambo {

// Version information
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/**
 * Get version string.
 */
inline auto getVersionString() -> const char* {
    return "1.0.0";
}

} // namespace rambo

#endif // RAMBO_HPP
