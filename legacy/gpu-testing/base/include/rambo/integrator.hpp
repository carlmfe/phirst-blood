#pragma once
#ifndef RAMBO_BASE_INTEGRATOR_HPP
#define RAMBO_BASE_INTEGRATOR_HPP

#include <cstdint>
#include <cmath>

#include "phase_space.hpp"

namespace rambo {

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

private:
    int64_t nEvents_;
    Integrand integrand_;
    
    // Simple serial Monte Carlo loop (kept intentionally straightforward).
    void launchMonteCarloKernel(double cmEnergy, const double* masses,
                                 double& sum, double& sumSquared, uint64_t seed) {
        PhaseSpaceGenerator<NumParticles, Algorithm> generator(masses);
        uint64_t rngState = seed;
        if (rngState == 0) rngState = 1;

        double localSum = 0.0;
        double localSum2 = 0.0;
        double momenta[NumParticles][4];

        for (int64_t i = 0; i < nEvents_; ++i) {
            double logWeight = generator(cmEnergy, rngState, momenta);
            double fx = integrand_.evaluate(momenta);
            double weightedValue = fx * std::exp(logWeight);
            localSum += weightedValue;
            localSum2 += weightedValue * weightedValue;
        }

        sum = localSum;
        sumSquared = localSum2;
    }
};

} // namespace rambo

#endif // RAMBO_BASE_INTEGRATOR_HPP
