#pragma once
#ifndef RAMBO_BASE_HPP
#define RAMBO_BASE_HPP

// =============================================================================
// RAMBO - Phase Space Generator Library (Base/Serial Implementation)
// =============================================================================
// Main include file - includes all library components
//
// Usage:
//   #include <rambo/rambo.hpp>
//
//   rambo::RamboIntegrator<MyIntegrand, 2> integrator(nEvents, integrand);
//   integrator.run(cmEnergy, masses, mean, error, seed);
// =============================================================================

#include "phase_space.hpp"
#include "integrator.hpp"
#include "integrands.hpp"

namespace rambo {

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace rambo

#endif // RAMBO_BASE_HPP
