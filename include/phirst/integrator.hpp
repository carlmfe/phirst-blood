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
    double masses[NumParticles];

    PHIRST_HOST_DEVICE
    MCWorkFunctor(const Generator& gen, const Integrand& integ, double E,
                  const double* m, uint64_t seed)
        : generator(gen), integrand(integ), cmEnergy(E), baseSeed(seed) {
        for (int i = 0; i < NumParticles; ++i) masses[i] = m[i];
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
 * @tparam Integrand Type with `evaluate(const double momenta[][4])` method.
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
     * Run the Monte Carlo integration.
     * @param cmEnergy Center-of-mass energy (GeV).
     * @param masses Array of particle masses (length NumParticles).
     * @param mean Output: estimated mean of integrand * weight.
     * @param error Output: statistical error on the mean.
     * @param seed RNG seed for reproducibility.
     */
    void runVegas(double cmEnergy, const double* masses,
                  double& mean, double& error,
                  uint64_t seed = 5489ULL) {
        IntegrationResult result;
        result.nEvents = nEvents_;

        // Create phase space generator and work functor
        using Generator = PhaseSpaceGenerator<NumParticles, Algorithm>;
        Generator generator(masses);

        VegasParams vegasParams;

        VegasWorkFunctor<Generator, Integrand, NumParticles> work(
            generator, integrand_, cmEnergy, masses, seed, vegasParams);

        const int nIters = (nEvents_ + vegasParams.nCallsPerIter - 1) \
                / vegasParams.nCallsPerIter;

        for (int iter = 0; iter < nIters; ++iter) {
            double iterSum = 0.0;
            double iterSum2 = 0.0;

            // Run iteration of VEGAS sampling
            grid_stride_reduce(
                nEvents_,
                work,
                result.sum,
                result.sum2
            );

            /*
             *
            // Compute iteration statistics
            double nCalls = static_cast<double>(params_.nCallsPerIter);
            double iterMean = iterSum / nCalls;
            double iterVar = (iterSum2 / nCalls) - (iterMean * iterMean);
            double iterError = std::sqrt(std::fabs(iterVar) / nCalls);

            // Combine with previous iterations (weighted average)
            if (iterError > 0.0) {
                double w = 1.0 / (iterError * iterError);
                sumI += iterMean * w;
                sumW += w;

                if (iter > 0) {
                    double combined = sumI / sumW;
                    sumChi2 += (iterMean - combined) * (iterMean - combined) * w;
                }
            }

            // Update result
            if (sumW > 0.0) {
                result.integral = sumI / sumW;
                result.error = std::sqrt(1.0 / sumW);
                result.chiSquared = (iter > 0) ? sumChi2 / iter : 0.0;
            }

            // Early-stop check: relative error criterion
            if (params_.stopRelError > 0.0) {
                double relErr = (result.integral != 0.0) ? std::fabs(result.error / result.integral) : 1e9;
                if ((iter + 1) >= params_.minIterations && relErr <= params_.stopRelError) {
                    result.nIterations = iter + 1;
                    #if defined(DEBUG) || defined(_DEBUG)
                    if (params_.verbose) {
                        std::cout << "  Early stopping at iteration " << (iter + 1)
                                    << " (rel err = " << relErr << ")\\n";
                    }
                    #endif
                    break;
                }
            }

            // Adapt grid for next iteration (except last)
            if (iter < params_.nIterations - 1) {
                run_single_thread(AdaptGridWorkFunctor{});
            }
             */
            run_single_thread(AdaptGridWorkFunctor{});

        }

        result.computeStatistics();
        mean = result.mean;
        error = result.error;
    }




private:
    int64_t nEvents_;
    Integrand integrand_;
};

} // namespace phirst

#endif // PHIRST_INTEGRATOR_HPP