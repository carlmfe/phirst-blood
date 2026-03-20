#pragma once
#ifndef PHIRST_HPP
#define PHIRST_HPP

// =============================================================================
// PHIRST - Phase-space Integration and Random Sampling Toolkit
// =============================================================================
// Main include file - includes all library components.
//
// This unified library supports multiple backends:
//   - PHIRST_BACKEND_SERIAL  : CPU serial (default)
//   - PHIRST_BACKEND_CUDA    : NVIDIA CUDA
//   - PHIRST_BACKEND_KOKKOS  : Kokkos (CUDA/CPU)
//   - PHIRST_BACKEND_ALPAKA  : Alpaka (CUDA/CPU/OpenMP)
//   - PHIRST_BACKEND_SYCL    : SYCL (CUDA/Intel)
//
// Usage:
//   #include <phirst/phirst.hpp>
//
//   phirst::DrellYanIntegrand integrand(2.0/3.0, 1.0/137.0);
//   phirst::RamboIntegrator<phirst::DrellYanIntegrand, 2> integrator(nEvents, integrand);
//   
//   double mean, error;
//   integrator.run(cmEnergy, masses, mean, error, seed);
//
// Build with backend selection:
//   cmake -DPHIRST_BACKEND=KOKKOS ..
//   cmake -DPHIRST_BACKEND=CUDA ..
//   cmake -DPHIRST_BACKEND=SYCL -DCMAKE_CXX_COMPILER=clang++ ..
// =============================================================================

#include "backend/config.hpp"
#include "backend/math.hpp"
#include "backend/random.hpp"
#include "phase_space.hpp"
#include "integrands.hpp"
#include "integrator.hpp"

namespace phirst {

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

} // namespace phirst

#endif // PHIRST_HPP
