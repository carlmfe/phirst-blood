#pragma once
#ifndef PHIRST_VEGAS_HPP
#define PHIRST_VEGAS_HPP

#include "backend/random.hpp"
#include "backend/math.hpp"
#include "contrib/HEPUtils/Vectors.h"


#include <vector>
#include <cstdint>

namespace phirst {

/** \brief VEGAS configuration parameters. */
struct VegasParams {
    int nDim = 2;                 // Number of integration dimensions
    int nBins = -1;               // Number of bins per dimension
    int nIterations = 10;         // Number of adaptation iterations
    int64_t nCallsPerIter = -1; // Samples per iteration
    double alpha = 1.5;           // Grid adaptation damping parameter
    bool verbose = true;          // Print iteration results
    int minIterations = 1;        // Do not stop before this many iterations
    double stopRelError = 0.0;    // If >0, stop when relative error <= this value
};

/** \brief VEGAS integration result. */
struct VegasResult {
    double integral = 0.0;        // Final integral estimate
    double error = 0.0;           // Statistical error
    double chiSquared = 0.0;      // Chi-squared per DOF
    int nIterations = 0;          // Iterations completed
    std::vector<double> iterIntegrals;  // Per-iteration results
    std::vector<double> iterErrors;     // Per-iteration errors
};

/** \brief C-style struct holding pointers to VEGAS grid data on the device. */
struct VegasGrid {
    double* xi;       // Grid boundaries [nDim * (nBins + 1)]
    double* binAcc;   // Bin accumulators [nDim * nBins]
    int*    binCounts;  // Bin counts [nDim * nBins]
    int     nDim;
    int     nBins;
    double  alpha;

    PHIRST_HOST_DEVICE
    double transform(const double* u, double* x, int* bins) const {
        double jacobian = 1.0;
        for (int d = 0; d < nDim; ++d) {
            double pos = u[d] * nBins;
            int bin = static_cast<int>(pos);
            if (bin >= nBins) bin = nBins - 1;
            if (bin < 0) bin = 0;

            double frac = pos - bin;
            double xLow = xi[d * (nBins + 1) + bin];
            double xHigh = xi[d * (nBins + 1) + bin + 1];
            double dx = xHigh - xLow;

            x[d] = xLow + frac * dx;
            bins[d] = bin;
            jacobian *= dx * nBins;
        }
        return jacobian;
    }
};

template <typename Generator, typename Integrand, int NumParticles, int MaxDim = 10>
struct VegasWorkFunctor {
    Generator generator;
    Integrand integrand;
    double cmEnergy;
    uint64_t baseSeed;
    double masses[NumParticles];
    VegasGrid grid;

    PHIRST_HOST_DEVICE
    VegasWorkFunctor(const Generator& gen, const Integrand& integ, double E, 
                     const double* m, uint64_t seed, const VegasGrid& g)
        : generator(gen), integrand(integ), cmEnergy(E), baseSeed(seed), 
          grid(g) {
        for (int i = 0; i < NumParticles; ++i) this->masses[i] = m[i];
    }

    template <typename Acc>
    PHIRST_HOST_DEVICE
    void operator()(const Acc& acc, int64_t workIdx, double& sum, double& sum2) const {
        uint64_t rngState = seed_for_thread(baseSeed, workIdx);
        double r[Generator::nRandomNumbers];
        for (int i = 0; i < Generator::nRandomNumbers; ++i) {
            r[i] = uniformRandom(rngState);
        }

        double x[MaxDim];
        int bins[MaxDim];
        double jacobian = grid.transform(r, x, bins);

        double rawMomenta[NumParticles][4];
        double logWeight = generator(cmEnergy, x, rawMomenta);

        HEPUtils::P4 momenta[NumParticles];
        for (int i = 0; i < NumParticles; ++i) {
            momenta[i] = HEPUtils::P4::mkXYZM(
                rawMomenta[i][1], rawMomenta[i][2], rawMomenta[i][3], masses[i]
            );
        }
        
        double fx = integrand.evaluate(momenta);
        double weightedValue = fx * math::exp(logWeight) * jacobian;

        sum += weightedValue;
        sum2 += weightedValue * weightedValue;

        double abs_weighted_val = math::fabs(weightedValue);
        for (int d = 0; d < grid.nDim; ++d) {
            int bin_idx = d * grid.nBins + bins[d];
            atomic_add(acc, &grid.binAcc[bin_idx], abs_weighted_val);
            atomic_add(acc, &grid.binCounts[bin_idx], 1);
        }
    }
};

/**
 * @brief Functor to initialize the VEGAS grid on the device.
 */
struct InitGridWorkFunctor {
    VegasGrid grid;

    template <typename Acc>
    PHIRST_HOST_DEVICE
    void operator()(const Acc& /*acc*/) const {
        for (int d = 0; d < grid.nDim; ++d) {
            for (int b = 0; b <= grid.nBins; ++b) {
                grid.xi[d * (grid.nBins + 1) + b] = static_cast<double>(b) / grid.nBins;
            }
            for (int b = 0; b < grid.nBins; ++b) {
                grid.binAcc[d * grid.nBins + b] = 0.0;
                grid.binCounts[d * grid.nBins + b] = 0;
            }
        }
    }
};

/**
 * @brief Functor to adapt the VEGAS grid on the device.
 */
template<int MaxBins = 100> // MaxBins should be >= VegasParams::nBins
struct AdaptGridWorkFunctor {
    VegasGrid grid;

    template <typename Acc>
    PHIRST_HOST_DEVICE
    void operator()(const Acc& /*acc*/) const {
        // Temporary storage. These are allocated on the stack, so nBins must be reasonable.
        double newXi[MaxBins + 1];
        double avgF[MaxBins];
        double weights[MaxBins];

        // --- Process each dimension independently ---
        for (int d = 0; d < grid.nDim; ++d) {
            // --- Step 1: Compute average |f| per bin for the current dimension ---
            double sumF = 0.0;
            for (int i = 0; i < grid.nBins; ++i) {
                const int bin_idx = d * grid.nBins + i;
                if (grid.binCounts[bin_idx] > 0) {
                    avgF[i] = grid.binAcc[bin_idx] / static_cast<double>(grid.binCounts[bin_idx]);
                } else {
                    avgF[i] = 0.0;
                }
                sumF += avgF[i];
            }

            if (sumF <= 0.0) continue;

            // --- Step 2: Smooth weights based on the VEGAS algorithm ---
            double sumWeights = 0.0;
            const double avgF_total = sumF / grid.nBins;

            for (int i = 0; i < grid.nBins; ++i) {
                if (avgF[i] > 0.0) {
                    double fi = avgF[i] / avgF_total;
                    double logFi = math::log(fi);
                    if (math::fabs(logFi) > 1e-10) {
                        weights[i] = math::pow(math::fabs((fi - 1.0) / logFi), grid.alpha);
                    } else {
                        weights[i] = 1.0;
                    }
                } else {
                    weights[i] = 1.0;
                }
                sumWeights += weights[i];
            }

            // --- Step 3: Compute new grid points based on the smoothed weights ---
            newXi[0] = 0.0;
            newXi[grid.nBins] = 1.0;
            const double targetWeight = sumWeights / grid.nBins;
            double cumWeight = 0.0;
            int oldBin = 0;

            for (int i = 1; i < grid.nBins; ++i) {
                double target = i * targetWeight;
                while (cumWeight + weights[oldBin] < target && oldBin < grid.nBins - 1) {
                    cumWeight += weights[oldBin];
                    oldBin++;
                }
                const double remain = target - cumWeight;
                const double frac = (weights[oldBin] > 0.0) ? remain / weights[oldBin] : 0.0;
                const int grid_idx_low = d * (grid.nBins + 1) + oldBin;
                const double oldLow = grid.xi[grid_idx_low];
                const double oldHigh = grid.xi[grid_idx_low + 1];
                newXi[i] = oldLow + frac * (oldHigh - oldLow);
            }

            // --- Step 4: Overwrite the old grid with the new grid for this dimension ---
            for (int i = 0; i <= grid.nBins; ++i) {
                grid.xi[d * (grid.nBins + 1) + i] = newXi[i];
            }
        }

        // --- Step 5: Reset all accumulators for the next iteration ---
        for (int i = 0; i < grid.nDim * grid.nBins; ++i) {
            grid.binAcc[i] = 0.0;
            grid.binCounts[i] = 0;
        }
    }
};

} // namespace phirst

#endif // PHIRST_VEGAS_HPP
