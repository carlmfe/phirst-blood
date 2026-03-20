#pragma once
#ifndef RAMBO_CUDA_INTEGRATOR_CUH
#define RAMBO_CUDA_INTEGRATOR_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "phase_space.cuh"

namespace rambo
{

    // =============================================================================
    // Integration result
    // =============================================================================

    // Stores accumulators and computes simple Monte Carlo statistics.
    struct IntegrationResult
    {
        double mean = 0.0;
        double error = 0.0;
        double sum = 0.0;
        double sumSquared = 0.0;
        int64_t nEvents = 0;

        void computeStatistics()
        {
            mean = sum / static_cast<double>(nEvents);
            double variance = (sumSquared / static_cast<double>(nEvents)) - (mean * mean);
            error = std::sqrt(std::fabs(variance) / static_cast<double>(nEvents));
        }
    };

    // =============================================================================
    // Monte Carlo kernel
    // =============================================================================

    // Simple GPU kernel: grid-stride loop with per-thread accumulation and atomic
    // reduction into two device scalars (sum, sumSquared).
    template <typename Integrand, int NumParticles, typename Algorithm>
    __global__ void MonteCarloKernel(
        double cmEnergy,
        const double *masses,
        Integrand integrand,
        double *deviceSum,
        double *deviceSum2,
        uint64_t baseSeed,
        int64_t nEvents)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        uint64_t rngState = baseSeed ^ (uint64_t(idx) * 2685821657736338717ULL);
        if (rngState == 0)
            rngState = baseSeed + 1;

        double localSum = 0.0;
        double localSum2 = 0.0;

        // Create generator once per thread - pre-computes mass-dependent quantities
        PhaseSpaceGenerator<NumParticles, Algorithm> rambo(cmEnergy, masses);

        while (idx < nEvents)
        {
            double momenta[NumParticles][4];
            double logWeight = rambo(rngState, momenta);

            double fx = integrand.evaluate(momenta);
            double weightedValue = fx * exp(logWeight);

            localSum += weightedValue;
            localSum2 += weightedValue * weightedValue;

            idx += gridDim.x * blockDim.x;
        }

        atomicAdd(deviceSum, localSum);
        atomicAdd(deviceSum2, localSum2);
    }

    // =============================================================================
    // Integrator class
    // =============================================================================

    template <typename Integrand, int NumParticles, typename Algorithm = RamboAlgorithm<NumParticles>>
    class RamboIntegrator
    {
    public:
        RamboIntegrator(int64_t nEvents, const Integrand &integrand)
            : nEvents_(nEvents), integrand_(integrand) {}

        void run(double cmEnergy, const double *hostMasses,
                 double &mean, double &error,
                 uint64_t seed = 5489ULL)
        {

            IntegrationResult result;
            result.nEvents = nEvents_;

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            int blockSize = 256;
            // Cap number of blocks to avoid excessive atomic contention (match Alpaka heuristic)
            int maxBlocksCap = 1024;
            int requestedBlocks = static_cast<int>((nEvents_ + blockSize - 1) / blockSize);
            int numBlocks = std::min(requestedBlocks, maxBlocksCap);
            numBlocks = std::min(numBlocks, static_cast<int>(prop.maxGridSize[0]));

            double *deviceMasses = nullptr;
            double *deviceSum = nullptr;
            double *deviceSum2 = nullptr;

            cudaMalloc(&deviceMasses, sizeof(double) * NumParticles);
            cudaMalloc(&deviceSum, sizeof(double));
            cudaMalloc(&deviceSum2, sizeof(double));

            cudaMemcpy(deviceMasses, hostMasses, sizeof(double) * NumParticles, cudaMemcpyHostToDevice);
            cudaMemset(deviceSum, 0, sizeof(double));
            cudaMemset(deviceSum2, 0, sizeof(double));

            MonteCarloKernel<Integrand, NumParticles, Algorithm><<<numBlocks, blockSize>>>(
                cmEnergy, deviceMasses, integrand_, deviceSum, deviceSum2, seed, nEvents_);

            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            }

            cudaMemcpy(&result.sum, deviceSum, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&result.sumSquared, deviceSum2, sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(deviceMasses);
            cudaFree(deviceSum);
            cudaFree(deviceSum2);

            result.computeStatistics();
            mean = result.mean;
            error = result.error;
        }

    private:
        int64_t nEvents_;
        Integrand integrand_;
    };

} // namespace rambo

#endif // RAMBO_CUDA_INTEGRATOR_CUH
