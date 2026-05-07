#pragma once

#include <cstddef>
#include <cstdint>

extern "C" __device__ double phirst_user_integrand(const double* momenta_flat, int n_particles);

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
