#pragma once
#ifndef RAMBO_KOKKOS_HPP
#define RAMBO_KOKKOS_HPP

// =============================================================================
// RAMBO - Phase Space Generator Library (Kokkos GPU Implementation)
// =============================================================================
// Main include file - includes all library components
//
// Usage:
//   #include <Kokkos_Core.hpp>
//   #include <rambo/rambo.hpp>
//
//   int main(int argc, char* argv[]) {
//       Kokkos::initialize(argc, argv);
//       {
//           rambo::RamboIntegrator<MyIntegrand, 2> integrator(nEvents, integrand);
//           integrator.run(cmEnergy, masses, mean, error, seed);
//       }
//       Kokkos::finalize();
//   }
// =============================================================================

#include "phase_space.hpp"
#include "integrator.hpp"
#include "integrands.hpp"

namespace rambo {

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace rambo

#endif // RAMBO_KOKKOS_HPP
