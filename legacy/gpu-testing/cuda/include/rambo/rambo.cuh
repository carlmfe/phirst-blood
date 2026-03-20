#pragma once
#ifndef RAMBO_CUDA_HPP
#define RAMBO_CUDA_HPP

// =============================================================================
// RAMBO - Phase Space Generator Library (CUDA Implementation)
// =============================================================================

#include "phase_space.cuh"
#include "integrator.cuh"
#include "integrands.cuh"

namespace rambo {

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace rambo

#endif // RAMBO_CUDA_HPP
