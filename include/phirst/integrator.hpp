#pragma once
#ifndef PHIRST_INTEGRATOR_HPP
#define PHIRST_INTEGRATOR_HPP

/**
 * @file integrator.hpp
 * @brief Portable Monte Carlo integrator for RAMBO phase space.
 * 
 * This file provides a backend-agnostic RamboIntegrator class that performs
 * Monte Carlo integration over phase space using the RAMBO algorithm.
 * 
 * Uses the grid_stride_reduce primitive from parallel.hpp.
 * The MC-specific logic (phase space generation, integrand evaluation) is
 * encapsulated in a work functor that is passed to the generic primitive.
 */

#include "backend/math.hpp"
#include "backend/random.hpp"
#include "backend/parallel.hpp"
#include "phase_space.hpp"
#include "vegas.hpp"
#include "contrib/HEPUtils/Vectors.h"
#include <iostream>
#include <algorithm>

namespace phirst {

// =============================================================================
// Integration Result
// =============================================================================

/**
 * Stores accumulators and computes simple Monte Carlo statistics.
 */
struct IntegrationResult {
    double mean = 0.0;
    double error = 0.0;
    double sum = 0.0;
    double sum2 = 0.0;
    int64_t nEvents = 0;

    /**
     * Compute mean and standard error from accumulated sums.
     */
    void computeStatistics() {
        if (nEvents == 0) { return; }
        mean = sum / static_cast<double>(nEvents);
        double variance = (sum2 / static_cast<double>(nEvents)) - (mean * mean);
        error = math::sqrt(math::fabs(variance) / static_cast<double>(nEvents));
    }
};

// =============================================================================
// MC Work Functor - Encapsulates MC integration logic
// =============================================================================

/**
 * Work functor for Monte Carlo phase space integration.
 * This struct encapsulates the per-event computation:
 * 1. Seed RNG from work index
 * 2. Generate phase space point
 * 3. Evaluate integrand
 * 4. Accumulate weighted value and squared value
 * 
 * Used with grid_stride_reduce primitive.
 */
template <typename Generator, typename Integrand, int NumParticles>
struct MCWorkFunctor {
    Generator generator;
    Integrand integrand;
    double cmEnergy;
    uint64_t baseSeed;
    double masses[NumParticles] = {};

    PHIRST_HOST_DEVICE
    MCWorkFunctor(const Generator& gen, const Integrand& integ, double E,
                  const double* m, uint64_t seed)
        : generator(gen), integrand(integ), cmEnergy(E), baseSeed(seed) {
        for (int i = 0; i < NumParticles; ++i) { masses[i] = m[i]; }
    }

    template <typename Acc>
    PHIRST_HOST_DEVICE
    void operator()(const Acc& /*acc*/, int64_t workIdx, double& sum, double& sum2) const {
        // Compute per-event RNG seed from work index
        uint64_t rngState = seed_for_thread(baseSeed, workIdx);

        double r[Generator::nRandomNumbers];
        for (int i = 0; i < Generator::nRandomNumbers; ++i) {
            r[i] = uniformRandom(rngState);
        }

        double rawMomenta[NumParticles][4];
        double logWeight = generator(cmEnergy, r, rawMomenta);

        // Use mkXYZM with the known mass to avoid sqrt(E²-p²) NaN for massless particles
        HEPUtils::P4 momenta[NumParticles];
        for (int i = 0; i < NumParticles; ++i) {
            momenta[i] = HEPUtils::P4::mkXYZM(
                rawMomenta[i][1], rawMomenta[i][2], rawMomenta[i][3], masses[i]
            );
        }

        double fx = integrand.evaluate(momenta);
        double weightedValue = fx * math::exp(logWeight);

        sum += weightedValue;
        sum2 += weightedValue * weightedValue;
    }
};

// =============================================================================
// Integrator Class
// =============================================================================

/**
 * Monte Carlo integrator using RAMBO phase space generation.
 * 
 * @tparam Integrand Type with `evaluate(const HEPUtils::P4 momenta[])` method.
 * @tparam NumParticles Number of final-state particles.
 * @tparam Algorithm Phase space algorithm (default: RamboAlgorithm).
 */
template <typename Integrand, int NumParticles, typename Algorithm = RamboAlgorithm<NumParticles>>
class RamboIntegrator {
public:
    /**
     * Construct integrator with event count and integrand.
     * @param nEvents Number of Monte Carlo samples.
     * @param integrand Physics integrand to evaluate.
     */
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}

    /**
     * Run the Monte Carlo integration.
     * @param cmEnergy Center-of-mass energy (GeV).
     * @param masses Array of particle masses (length NumParticles).
     * @param mean Output: estimated mean of integrand * weight.
     * @param error Output: statistical error on the mean.
     * @param seed RNG seed for reproducibility.
     */
    void run(double cmEnergy, const double* masses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        IntegrationResult result;
        result.nEvents = nEvents_;

        // Create phase space generator and work functor
        using Generator = PhaseSpaceGenerator<NumParticles, Algorithm>;
        Generator generator(masses);

        MCWorkFunctor<Generator, Integrand, NumParticles> work(
            generator, integrand_, cmEnergy, masses, seed);

        // Use the generic grid_stride_reduce primitive
        grid_stride_reduce(
            nEvents_,
            work,
            result.sum,
            result.sum2
        );

        result.computeStatistics();
        mean = result.mean;
        error = result.error;
    }

    /**
     * Run the Monte Carlo integration using VEGAS.
     * @param cmEnergy Center-of-mass energy (GeV).
     * @param masses Array of particle masses (length NumParticles).
     * @param mean Output: estimated mean of integrand * weight.
     * @param error Output: statistical error on the mean.
     * @param seed RNG seed for reproducibility.
     */
    void runVegas(double cmEnergy, const double* masses,
                  double& mean, double& error,
                  uint64_t seed = 5489ULL) {

        using Generator = PhaseSpaceGenerator<NumParticles, Algorithm>;

        VegasParams params;
        params.nDim = Generator::nRandomNumbers;

        // Probe events per iteration: small enough to keep GPU atomic contention
        // manageable (~1000 updates per bin), large enough for reliable adaptation.
        if (params.nCallsPerIter < 0) {
            params.nCallsPerIter = 10000;
        }
        const int64_t nProbePerIter = params.nCallsPerIter;

        if (params.nBins < 0) {
            params.nBins = static_cast<int>(std::min(
                math::pow(static_cast<double>(nProbePerIter),
                          1.0 / static_cast<double>(params.nDim)),
                50.0));
            params.nBins = std::max(params.nBins, 2);
        }

        // Allocate device memory for the VEGAS grid
        DeviceBuffer<double> d_xi(static_cast<int64_t>(params.nDim) * (params.nBins + 1));
        DeviceBuffer<double> d_binAcc(static_cast<int64_t>(params.nDim) * params.nBins);
        DeviceBuffer<int>    d_binCounts(static_cast<int64_t>(params.nDim) * params.nBins);

        // Create the grid struct with pointers to device memory
        VegasGrid grid {
            d_xi.data(), d_binAcc.data(), d_binCounts.data(),
            params.nDim, params.nBins, params.alpha
        };

        // Initialize the grid on the device
        InitGridWorkFunctor initFunctor{grid};
        run_single_thread(initFunctor);

        Generator generator(masses);

        // Two-phase per iteration:
        //   Phase 1 (probe):     nProbePerIter events WITH bin updates → grid adapts.
        //                        Atomic contention is manageable due to small event count.
        //   Phase 2 (integrate): nIntegratePerIter events WITHOUT bin updates → no atomics.
        //                        This is the performance-critical path; runs at flat-MC speed.
        using ProbeF = VegasWorkFunctor<Generator, Integrand, NumParticles, 10, true>;
        using IntF   = VegasWorkFunctor<Generator, Integrand, NumParticles, 10, false>;

        const int nIters = std::max(1, params.nIterations);
        const int64_t nIntegratePerIter = std::max(static_cast<int64_t>(1), nEvents_ / static_cast<int64_t>(nIters));

        IntegrationResult totalResult;
        totalResult.nEvents = static_cast<int64_t>(nIters) * nIntegratePerIter;

        for (int iter = 0; iter < nIters; ++iter) {
            // Phase 1: probe (small, with atomics) to gather bin statistics
            const uint64_t probeSeed = seed ^ (static_cast<uint64_t>(iter + 1) * 0xBF58476D1CE4E5B9ULL);
            ProbeF probeWork(generator, integrand_, cmEnergy, masses, probeSeed, grid);
            double probeSum = 0.0;
            double probeSum2 = 0.0;
            grid_stride_reduce(nProbePerIter, probeWork, probeSum, probeSum2);

            // Adapt the VEGAS grid and reset bin accumulators for the next probe
            AdaptGridWorkFunctor<100> adaptFunctor{grid};
            run_single_thread(adaptFunctor);

            // Phase 2: integrate with the adapted grid, no bin updates (no GPU atomics)
            const uint64_t intSeed = seed + static_cast<uint64_t>(iter + 1) * 0x94D049BB133111EBULL;
            IntF intWork(generator, integrand_, cmEnergy, masses, intSeed, grid);
            double iterSum = 0.0;
            double iterSum2 = 0.0;
            grid_stride_reduce(nIntegratePerIter, intWork, iterSum, iterSum2);

            totalResult.sum  += iterSum;
            totalResult.sum2 += iterSum2;
        }

        totalResult.computeStatistics();
        mean  = totalResult.mean;
        error = totalResult.error;
    }

private:
    int64_t nEvents_;
    Integrand integrand_;
};

} // namespace phirst

#endif // PHIRST_INTEGRATOR_HPP