#pragma once
#ifndef RAMBO_KOKKOS_INTEGRATOR_HPP
#define RAMBO_KOKKOS_INTEGRATOR_HPP

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cmath>

#include "phase_space.hpp"

namespace rambo {

// =============================================================================
// Integration result
// =============================================================================

// Stores accumulators and computes simple Monte Carlo statistics.
struct IntegrationResult {
    double mean = 0.0;
    double error = 0.0;
    double sum = 0.0;
    double sumSquared = 0.0;
    int64_t nEvents = 0;
    
    void computeStatistics() {
        mean = sum / static_cast<double>(nEvents);
        double variance = (sumSquared / static_cast<double>(nEvents)) - (mean * mean);
        error = std::sqrt(std::fabs(variance) / static_cast<double>(nEvents));
    }
};


// =============================================================================
// Integrator class
// =============================================================================

template <typename Integrand, int NumParticles, typename Algorithm = RamboAlgorithm<NumParticles>>
class RamboIntegrator {
public:
    // Construct with number of Monte Carlo events and integrand instance.
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    // Run the integrator; returns mean and error by reference.
    void run(double cmEnergy, const double* masses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        IntegrationResult result;
        result.nEvents = nEvents_;
        launchMonteCarloKernel(cmEnergy, masses, result.sum, result.sumSquared, seed);
        result.computeStatistics();
        mean = result.mean;
        error = result.error;
    }

// Must be public for CUDA extended __host__ __device__ lambda
public:
    int64_t nEvents_;
    Integrand integrand_;
    
    // Kokkos parallel Monte Carlo integration with grid-stride loop and atomics
    void launchMonteCarloKernel(double cmEnergy, const double* masses,
                                 double& sum, double& sumSquared, uint64_t seed) {
        // Copy masses to device
        Kokkos::View<double[NumParticles]> deviceMasses("masses");
        auto hostMasses = Kokkos::create_mirror_view(deviceMasses);
        for (int i = 0; i < NumParticles; ++i) {
            hostMasses(i) = masses[i];
        }
        Kokkos::deep_copy(deviceMasses, hostMasses);
        
        // Allocate device scalars for atomic reduction
        Kokkos::View<double> deviceSum("sum");
        Kokkos::View<double> deviceSum2("sum2");
        Kokkos::deep_copy(deviceSum, 0.0);
        Kokkos::deep_copy(deviceSum2, 0.0);
        
        const int64_t nEvents = nEvents_;
        const Integrand integrand = integrand_;
        
        // Use grid-stride loop like CUDA/SYCL/Alpaka
        // Cap total threads to reduce atomic contention
        constexpr int64_t blockSize = 256;
        constexpr int64_t maxBlocks = 1024;
        int64_t numBlocks = std::min((nEvents + blockSize - 1) / blockSize, maxBlocks);
        int64_t totalThreads = numBlocks * blockSize;
        
        Kokkos::parallel_for("MonteCarloIntegration", totalThreads,
            KOKKOS_LAMBDA(const int64_t threadIdx) {
                // Initialize RNG state uniquely per thread (same formula as CUDA/SYCL)
                uint64_t rngState = seed ^ (static_cast<uint64_t>(threadIdx) * 2685821657736338717ULL);
                if (rngState == 0) rngState = seed + 1;
                
                // Get masses from device view
                double localMasses[NumParticles];
                for (int j = 0; j < NumParticles; ++j) {
                    localMasses[j] = deviceMasses(j);
                }
                
                // Thread-local accumulators
                double localSum = 0.0;
                double localSum2 = 0.0;
                
                // Create generator once per thread (pre-computes mass quantities)
                PhaseSpaceGenerator<NumParticles, Algorithm> generator(localMasses);
                
                // Grid-stride loop: each thread processes multiple events
                for (int64_t idx = threadIdx; idx < nEvents; idx += totalThreads) {
                    double momenta[NumParticles][4];
                    
                    double logWeight = generator(cmEnergy, rngState, momenta);
                    double fx = integrand.evaluate(momenta);
                    double weightedValue = fx * Kokkos::exp(logWeight);
                    
                    localSum += weightedValue;
                    localSum2 += weightedValue * weightedValue;
                }
                
                // Single atomic add per thread (not per event)
                Kokkos::atomic_add(&deviceSum(), localSum);
                Kokkos::atomic_add(&deviceSum2(), localSum2);
            }
        );
        
        Kokkos::fence();
        
        // Copy results back to host
        auto hostSum = Kokkos::create_mirror_view(deviceSum);
        auto hostSum2 = Kokkos::create_mirror_view(deviceSum2);
        Kokkos::deep_copy(hostSum, deviceSum);
        Kokkos::deep_copy(hostSum2, deviceSum2);
        
        sum = hostSum();
        sumSquared = hostSum2();
    }
};

} // namespace rambo

#endif // RAMBO_KOKKOS_INTEGRATOR_HPP
