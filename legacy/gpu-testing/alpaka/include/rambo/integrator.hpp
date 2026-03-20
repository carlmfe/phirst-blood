#pragma once
#ifndef RAMBO_ALPAKA_INTEGRATOR_HPP
#define RAMBO_ALPAKA_INTEGRATOR_HPP

#include <alpaka/alpaka.hpp>
#include <cmath>
#include <cstdint>
#include <concepts>

#include "phase_space.hpp"

namespace rambo {

// =============================================================================
// Concepts
// =============================================================================

template <typename T>
concept IntegrandConcept = requires(T integrand, const double momenta[][4]) {
    { integrand.evaluate(momenta) } -> std::convertible_to<double>;
};

// =============================================================================
// Algorithm concept: simple marker-based check
// Algorithms (RamboAlgorithm, RamboDietAlgorithm) provide a nested
// `algorithm_tag` type to satisfy this concept. This keeps the concept
// simple and robust across device/host callable signatures.
// =============================================================================

template <typename T>
concept AlgorithmConcept = requires { typename T::algorithm_tag; };

// =============================================================================
// Result structure
// =============================================================================

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
// Optimized Monte Carlo Kernel with Grid-Stride Loop and Atomic Reduction
// =============================================================================
// Each thread processes multiple events using a grid-stride loop,
// accumulates results locally, then uses atomicAdd for final reduction.
// This eliminates per-event global memory writes and multi-pass reduction.
// =============================================================================

template <typename Algorithm, IntegrandConcept Integrand, int NumParticles>
    requires AlgorithmConcept<Algorithm>
struct MonteCarloKernel {
    double energy;
    const double* masses;
    Integrand integrand;
    double* globalSum;
    double* globalSum2;
    uint64_t baseSeed;
    int64_t nEvents;
    
    ALPAKA_FN_HOST_ACC MonteCarloKernel(
        double e, const double* m, const Integrand& integ,
        double* s, double* s2, uint64_t seed, int64_t n)
        : energy(e), masses(m), integrand(integ),
          globalSum(s), globalSum2(s2), baseSeed(seed), nEvents(n) {}
    
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void {
        // Get linear thread index and grid size
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];
        auto const gridSize = alpaka::mapIdx<1u>(globalThreadExtent, globalThreadExtent)[0];
        
        // Initialize RNG state uniquely per thread (same formula as CUDA/SYCL)
        uint64_t rngState = baseSeed ^ (static_cast<uint64_t>(threadIdx) * 2685821657736338717ULL);
        if (rngState == 0) rngState = baseSeed + 1;
        
        // Thread-local accumulators (no global memory traffic until end)
        double localSum = 0.0;
        double localSum2 = 0.0;
        
        // Create generator once per thread - pre-computes mass-dependent quantities
        PhaseSpaceGenerator<NumParticles, Algorithm> rambo(energy, masses);
        
        // Grid-stride loop: each thread processes multiple events
        for (int64_t idx = static_cast<int64_t>(threadIdx); idx < nEvents; idx += static_cast<int64_t>(gridSize)) {
            // Generate phase space point
            double momenta[NumParticles][4];
            double logWeight = rambo(rngState, momenta);
            
            // Evaluate integrand
            double fx = integrand.evaluate(momenta);
            double weightedValue = fx * std::exp(logWeight);
            
            // Accumulate locally (no global memory write per event)
            localSum += weightedValue;
            localSum2 += weightedValue * weightedValue;
        }
        
        // Single atomic add per thread (not per event)
        alpaka::atomicAdd(acc, globalSum, localSum, alpaka::hierarchy::Grids{});
        alpaka::atomicAdd(acc, globalSum2, localSum2, alpaka::hierarchy::Grids{});
    }
};

// =============================================================================
// Integrator Class
// =============================================================================

template <typename TAccTag, IntegrandConcept Integrand, int NumParticles, typename Algorithm = RamboAlgorithm<NumParticles>>
    requires AlgorithmConcept<Algorithm>
class RamboIntegrator {
public:
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    using DevAcc = alpaka::Dev<Acc>;
    using DevHost = alpaka::DevCpu;
    
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    void run(double energy, const double* hostMasses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        auto const platformAcc = alpaka::Platform<Acc>{};
        auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);
        Queue queue(devAcc);
        
        // Allocate device memory
        auto deviceMasses = alpaka::allocBuf<double, Idx>(devAcc, static_cast<Idx>(NumParticles));
        auto deviceSum = alpaka::allocBuf<double, Idx>(devAcc, static_cast<Idx>(1));
        auto deviceSum2 = alpaka::allocBuf<double, Idx>(devAcc, static_cast<Idx>(1));
        
        // Initialize sums to zero
        alpaka::memset(queue, deviceSum, 0);
        alpaka::memset(queue, deviceSum2, 0);
        
        // Copy masses to device
        copyMassesToDevice(devHost, queue, hostMasses, deviceMasses);
        
        // Launch optimized Monte Carlo kernel
        launchMonteCarloKernel(devAcc, queue, energy, deviceMasses, deviceSum, deviceSum2, seed);
        
        // Copy results back (just 2 scalars)
        result.sum = copyScalarFromDevice(devHost, queue, deviceSum);
        result.sumSquared = copyScalarFromDevice(devHost, queue, deviceSum2);
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
private:
    int64_t nEvents_;
    Integrand integrand_;
    
    template <typename MassBuf>
    void copyMassesToDevice(const DevHost& devHost, Queue& queue, 
                            const double* hostMasses, MassBuf& deviceMasses) {
        auto hostMassesBuf = alpaka::allocBuf<double, Idx>(devHost, static_cast<Idx>(NumParticles));
        for (int i = 0; i < NumParticles; ++i) {
            hostMassesBuf[static_cast<Idx>(i)] = hostMasses[i];
        }
        alpaka::memcpy(queue, deviceMasses, hostMassesBuf);
        alpaka::wait(queue);
    }
    
    template <typename MassBuf, typename SumBuf>
    void launchMonteCarloKernel(const DevAcc& devAcc, Queue& queue, double energy,
                                 MassBuf& deviceMasses, SumBuf& deviceSum, 
                                 SumBuf& deviceSum2, uint64_t seed) {
        // Configure kernel launch: use reasonable block/grid sizes
        // Let alpaka figure out optimal work division based on device
        constexpr Idx blockSize = 256;
        Idx numBlocks = std::min(
            static_cast<Idx>((nEvents_ + blockSize - 1) / blockSize),
            static_cast<Idx>(1024)  // Cap blocks to avoid excessive atomics contention
        );
        Idx totalThreads = numBlocks * blockSize;
        
        alpaka::Vec<Dim, Idx> const extent{totalThreads};
        
        MonteCarloKernel<Algorithm, Integrand, NumParticles> kernel(
            energy,
            alpaka::getPtrNative(deviceMasses),
            integrand_,
            alpaka::getPtrNative(deviceSum),
            alpaka::getPtrNative(deviceSum2),
            seed,
            nEvents_
        );
        
        // Create work division with specified block size
        alpaka::Vec<Dim, Idx> const elementsPerThread{1};
        alpaka::Vec<Dim, Idx> const threadsPerBlock{blockSize};
        alpaka::Vec<Dim, Idx> const blocksPerGrid{numBlocks};
        
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>(blocksPerGrid, threadsPerBlock, elementsPerThread);
        
        alpaka::exec<Acc>(queue, workDiv, kernel);
        alpaka::wait(queue);
    }
    
    template <typename SumBuf>
    double copyScalarFromDevice(const DevHost& devHost, Queue& queue, SumBuf& deviceBuf) {
        auto hostResult = alpaka::allocBuf<double, Idx>(devHost, static_cast<Idx>(1));
        alpaka::memcpy(queue, hostResult, deviceBuf, static_cast<Idx>(1));
        alpaka::wait(queue);
        return hostResult[0];
    }
};

} // namespace rambo

#endif // RAMBO_ALPAKA_INTEGRATOR_HPP
