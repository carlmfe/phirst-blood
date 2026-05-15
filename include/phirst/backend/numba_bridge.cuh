#pragma once

#include <cstddef>
#include <cstdint>

// Numba float64 calling convention: return value is written via a hidden first pointer
// argument, and the PTX return register is always set to 0 (int32 = .b32).
// Signature must match PTX: (.b32 retval, .b64 retval_ptr, .b64 momenta, .b32 n_particles)
extern "C" __device__ int phirst_user_integrand(
    double* retval_ptr, const double* momenta_flat, int n_particles);

extern "C" __global__ void phirst_numba_mc_kernel(
    double cmEnergy,
    const double* masses,
    int nParticles,
    int64_t nEvents,
    uint64_t baseSeed,
    double* sumOut,
    double* sum2Out);

void phirst_numba_mc_launch(
    double cmEnergy,
    const double* d_masses,
    int nParticles,
    int64_t nEvents,
    uint64_t baseSeed,
    double* d_sum,
    double* d_sum2,
    int numBlocks,
    int blockSize);
