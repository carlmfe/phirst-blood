 #pragma once
#ifndef RAMBO_INTEGRATOR_HPP
#define RAMBO_INTEGRATOR_HPP

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
#include "phase_space.hpp"
#include "backend/parallel.hpp"

namespace rambo {

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
    double sumSquared = 0.0;
    int64_t nEvents = 0;
    
    /**
     * Compute mean and standard error from accumulated sums.
     */
    void computeStatistics() {
        mean = sum / static_cast<double>(nEvents);
        double variance = (sumSquared / static_cast<double>(nEvents)) - (mean * mean);
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
    
    RAMBO_HOST_DEVICE
    MCWorkFunctor(const Generator& gen, const Integrand& integ, double E, uint64_t seed)
        : generator(gen), integrand(integ), cmEnergy(E), baseSeed(seed) {}
    
    RAMBO_HOST_DEVICE
    void operator()(int64_t workIdx, double& acc1, double& acc2) const {
        // Compute per-event RNG seed from work index
        uint64_t rngState = seed_for_thread(baseSeed, workIdx);
        
        double momenta[NumParticles][4];
        double logWeight = generator(cmEnergy, rngState, momenta);
        double fx = integrand.evaluate(momenta);
        double weightedValue = fx * math::exp(logWeight);
        
        acc1 += weightedValue;
        acc2 += weightedValue * weightedValue;
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
            generator, integrand_, cmEnergy, seed);
        
        // Use the generic grid_stride_reduce primitive
        grid_stride_reduce(
            nEvents_,
            work,
            result.sum,
            result.sumSquared
        );
        
        result.computeStatistics();
        mean = result.mean;
        error = result.error;
    }

private:
    int64_t nEvents_;
    Integrand integrand_;
};

} // namespace rambo

#endif // RAMBO_INTEGRATOR_HPP
