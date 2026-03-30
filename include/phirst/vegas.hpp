#pragma once
#ifndef PHIRST_VEGAS_HPP
#define PHIRST_VEGAS_HPP

#include "backend/random.hpp"
#include "backend/math.hpp"
#include "contrib/HEPUtils/Vectors.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace phirst {

/** \brief VEGAS configuration parameters. */
struct VegasParams {
    int nDim = 2;                 // Number of integration dimensions
    int nBins = 50;               // Number of bins per dimension
    int nIterations = 10;         // Number of adaptation iterations
    int64_t nCallsPerIter = 100000; // Samples per iteration
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

/** \brief Adaptive VEGAS grid used to map uniforms->[0,1]^d and accumulate bins. */
template <int MaxDim = 10>
class VegasGrid {
public:
    static constexpr int MAX_BINS = 100;

    double* xi_;
    double* binAcc_;
    int* binCounts_;

    int nDim_;
    int nBins_;
    double alpha_;

    VegasGrid() : xi_(nullptr), binAcc_(nullptr), binCounts_(nullptr), nDim_(0), nBins_(0), alpha_(1.5) {}

    VegasGrid(int nDim, int nBins, double alpha = 1.5)
        : nDim_(nDim), nBins_(nBins), alpha_(alpha) {

        if (nDim > MaxDim) {
            throw std::runtime_error("VegasGrid: nDim exceeds MaxDim");
        }
        if (nBins > MAX_BINS) {
            throw std::runtime_error("VegasGrid: nBins exceeds MAX_BINS");
        }

        xi_        = new double[nDim * (nBins + 1)];
        binAcc_    = new double[nDim * nBins];
        binCounts_ = new int[nDim * nBins];

        for (int d = 0; d < nDim_; ++d) {
            for (int b = 0; b <= nBins_; ++b) {
                xi(d, b) = static_cast<double>(b) / nBins_;
                if (b == nBins_) continue;
                binAcc(d, b) = 0.0;
                binCounts(d, b) = 0;
            }
        }
    }

    ~VegasGrid() {
        delete[] xi_;
        delete[] binAcc_;
        delete[] binCounts_;
    }

    PHIRST_INLINE double& xi(int d, int b)        { return xi_[d * (nBins_ + 1) + b]; }
    PHIRST_INLINE double& binAcc(int d, int b)    { return binAcc_[d * nBins_ + b]; }
    PHIRST_INLINE int&    binCounts(int d, int b) { return binCounts_[d * nBins_ + b]; }

    PHIRST_HOST_DEVICE
    double transform(const double* u, double* x, int* bins) {
        double jacobian = 1.0;
        for (int d = 0; d < nDim_; ++d) {
            double pos = u[d] * nBins_;
            int bin = static_cast<int>(pos);
            if (bin >= nBins_) bin = nBins_ - 1;
            if (bin < 0) bin = 0;

            double frac = pos - bin;
            double xLow = xi(d, bin);
            double xHigh = xi(d, bin + 1);
            double dx = xHigh - xLow;

            x[d] = xLow + frac * dx;
            bins[d] = bin;
            jacobian *= dx * nBins_;
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
    std::array<double, NumParticles> masses;
    VegasParams params_;
    VegasGrid<MaxDim>* grid_;

    PHIRST_HOST_DEVICE
    VegasWorkFunctor(const Generator& gen, const Integrand& integ, double E, 
                     uint64_t seed, const std::array<double, NumParticles>& m, 
                     VegasParams p, VegasGrid<MaxDim>* g)
        : generator(gen), integrand(integ), cmEnergy(E), baseSeed(seed), 
          masses(m), params_(p), grid_(g) {}

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
        double jacobian = grid_->transform(r, x, bins);

        HEPUtils::P4 momenta[NumParticles];
        // Generator logic to fill momenta from x would go here
        
        double fx = integrand.evaluate(momenta);
        double weightedValue = fx * jacobian;

        sum += weightedValue;
        sum2 += weightedValue * weightedValue;

        // Inside VegasWorkFunctor::operator()
        double abs_weighted_val = math::fabs(weightedValue);
        for (int d = 0; d < params_.nDim; ++d) {
            atomic_add(acc, &grid_->binAcc(d, bins[d]), abs_weighted_val);
            atomic_add(acc, &grid_->binCounts(d, bins[d]), 1);
        }
    }
};

/**
 * @brief Functor to adapt the VEGAS grid on the device.
 *
 * This is intended to be launched as a single-threaded kernel. It reads the
 * bin accumulators filled by the main integration functor, calculates the new
 * grid boundaries based on the VEGAS algorithm, and overwrites the old grid.
 * It also resets the accumulators for the next iteration.
 */
template<int MaxBins = 100> // MaxBins should be >= VegasParams::nBins
struct AdaptGridWorkFunctor {
    double* xi;         // Device pointer to grid boundaries [nDim * (nBins + 1)]
    double* binAcc;     // Device pointer to bin accumulators [nDim * nBins]
    int*    binCounts;  // Device pointer to bin counts [nDim * nBins]
    int     nDim;
    int     nBins;
    double  alpha;

    template <typename Acc>
    PHIRST_HOST_DEVICE
    void operator()(const Acc& /*acc*/) const {
        // Temporary storage. These are allocated on the stack, so nBins must be reasonable.
        double newXi[MaxBins + 1];
        double avgF[MaxBins];
        double weights[MaxBins];

        // --- Process each dimension independently ---
        for (int d = 0; d < nDim; ++d) {
            // --- Step 1: Compute average |f| per bin for the current dimension ---
            double sumF = 0.0;
            for (int i = 0; i < nBins; ++i) {
                const int bin_idx = d * nBins + i;
                if (binCounts[bin_idx] > 0) {
                    avgF[i] = binAcc[bin_idx] / static_cast<double>(binCounts[bin_idx]);
                } else {
                    avgF[i] = 0.0;
                }
                sumF += avgF[i];
            }

            // If nothing was accumulated in this dimension, skip adaptation for it.
            if (sumF <= 0.0) {
                continue;
            }

            // --- Step 2: Smooth weights based on the VEGAS algorithm ---
            double sumWeights = 0.0;
            const double avgF_total = sumF / nBins;

            for (int i = 0; i < nBins; ++i) {
                if (avgF[i] > 0.0) {
                    // Normalized weight, fi = <f_i> / <<f>>
                    double fi = avgF[i] / avgF_total;
                    // Damped weight: w = ((fi - 1) / ln(fi))^alpha
                    double logFi = math::log(fi);
                    if (math::fabs(logFi) > 1e-10) {
                        weights[i] = math::pow(math::fabs((fi - 1.0) / logFi), alpha);
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
            newXi[nBins] = 1.0;
            const double targetWeight = sumWeights / nBins;
            double cumWeight = 0.0;
            int oldBin = 0;

            for (int i = 1; i < nBins; ++i) {
                double target = i * targetWeight;

                // Find which old bin the new grid line falls into
                while (cumWeight + weights[oldBin] < target && oldBin < nBins - 1) {
                    cumWeight += weights[oldBin];
                    oldBin++;
                }

                // Interpolate within the old bin to find the new grid line position
                const double remain = target - cumWeight;
                const double frac = (weights[oldBin] > 0.0) ? remain / weights[oldBin] : 0.0;

                const int grid_idx_low = d * (nBins + 1) + oldBin;
                const double oldLow = xi[grid_idx_low];
                const double oldHigh = xi[grid_idx_low + 1];

                newXi[i] = oldLow + frac * (oldHigh - oldLow);
            }

            // --- Step 4: Overwrite the old grid with the new grid for this dimension ---
            for (int i = 0; i <= nBins; ++i) {
                xi[d * (nBins + 1) + i] = newXi[i];
            }
        }

        // --- Step 5: Reset all accumulators for the next iteration ---
        for (int i = 0; i < nDim * nBins; ++i) {
            binAcc[i] = 0.0;
            binCounts[i] = 0;
        }
    }
};

} // namespace phirst

#endif // PHIRST_VEGAS_HPP