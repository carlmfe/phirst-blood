#pragma once

#include <cstddef>
#include <cstdint>

// Numba float64 calling convention: return value is written via a hidden first pointer
// argument, and the PTX return register is always set to 0 (int32 = .b32).
// Signature must match PTX: (.b32 retval, .b64 retval_ptr, .b64 momenta, .b32 n_particles)
extern "C" __device__ int phirst_user_integrand(
    double* retval_ptr, const double* momenta_flat, int n_particles);
