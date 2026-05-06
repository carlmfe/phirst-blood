#include <gtest/gtest.h>

#include "phirst/phirst.hpp"
#include "phirst/backend/parallel.hpp"

#include <cmath>
#include <vector>

using namespace phirst;

// =============================================================================
// Helper: allocate a VegasGrid with std::vector backing storage
// =============================================================================

struct GridAllocator {
    int nDim, nBins;
    std::vector<double> xi;
    std::vector<double> binAcc;
    std::vector<int>    binCounts;
    VegasGrid grid;

    GridAllocator(int nd, int nb)
        : nDim(nd), nBins(nb),
          xi(nd * (nb + 1), 0.0),
          binAcc(nd * nb, 0.0),
          binCounts(nd * nb, 0) {
        grid = { xi.data(), binAcc.data(), binCounts.data(), nd, nb, 1.5 };
    }
};

// =============================================================================
// InitGridWorkFunctor
// =============================================================================

TEST(VegasGrid, InitGridFunctorUniform) {
    GridAllocator g(2, 5);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    for (int d = 0; d < g.nDim; ++d) {
        for (int b = 0; b <= g.nBins; ++b) {
            EXPECT_NEAR(g.xi[d * (g.nBins + 1) + b],
                        static_cast<double>(b) / g.nBins, 1e-15)
                << "d=" << d << " b=" << b;
        }
        for (int b = 0; b < g.nBins; ++b) {
            EXPECT_DOUBLE_EQ(g.binAcc   [d * g.nBins + b], 0.0);
            EXPECT_EQ       (g.binCounts[d * g.nBins + b], 0);
        }
    }
}

// =============================================================================
// VegasGrid::transform
// =============================================================================

TEST(VegasGrid, TransformUniformIsIdentity) {
    // On a uniform grid dx = 1/nBins, jacobian per dim = dx * nBins = 1.
    GridAllocator g(2, 10);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    double u[2]    = {0.35, 0.72};
    double x[2]    = {};
    int    bins[2] = {};
    double jacobian = g.grid.transform(u, x, bins);

    EXPECT_NEAR(jacobian, 1.0, 1e-14);
    EXPECT_NEAR(x[0], u[0], 1e-14);
    EXPECT_NEAR(x[1], u[1], 1e-14);
}

TEST(VegasGrid, TransformBinAssignmentCorrect) {
    // 1D grid with 4 bins. u=0.6 → pos=2.4 → bin=2.
    // xi=[0, 0.25, 0.5, 0.75, 1.0]; x = 0.5 + 0.4*0.25 = 0.6.
    GridAllocator g(1, 4);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    double u[1]    = {0.6};
    double x[1]    = {};
    int    bins[1] = {};
    g.grid.transform(u, x, bins);

    EXPECT_EQ(bins[0], 2);
    EXPECT_NEAR(x[0], 0.6, 1e-14);
}

TEST(VegasGrid, TransformBoundaryU0) {
    // u=0 should map to x=0 and bin=0
    GridAllocator g(1, 5);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    double u[1] = {0.0}, x[1] = {};
    int bins[1] = {};
    g.grid.transform(u, x, bins);

    EXPECT_EQ(bins[0], 0);
    EXPECT_NEAR(x[0], 0.0, 1e-14);
}

// =============================================================================
// AdaptGridWorkFunctor
// =============================================================================

TEST(VegasGrid, AdaptConcentratesHighWeightRegion) {
    // 1D, 5 bins. Seed bin 0 with much higher weight. After adaptation,
    // bin 0 should be narrower: xi[1] < 1/nBins.
    const int nBins = 5;
    GridAllocator g(1, nBins);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    g.binAcc[0] = 100.0;  g.binCounts[0] = 1;
    for (int b = 1; b < nBins; ++b) {
        g.binAcc[b] = 1.0; g.binCounts[b] = 1;
    }

    AdaptGridWorkFunctor<10> adapt{g.grid};
    adapt(KernelAcc{});

    // High-weight bin 0 should attract more points → narrower width
    double uniformWidth = 1.0 / nBins;
    EXPECT_LT(g.xi[1], uniformWidth);

    // Boundaries stay in [0,1] and remain monotone
    EXPECT_NEAR(g.xi[0],     0.0, 1e-14);
    EXPECT_NEAR(g.xi[nBins], 1.0, 1e-14);
    for (int b = 0; b < nBins; ++b)
        EXPECT_LT(g.xi[b], g.xi[b + 1]) << "non-monotone at b=" << b;

    // Accumulators are reset after adaptation
    for (int b = 0; b < nBins; ++b) {
        EXPECT_DOUBLE_EQ(g.binAcc   [b], 0.0);
        EXPECT_EQ       (g.binCounts[b], 0);
    }
}

TEST(VegasGrid, AdaptUniformWeightsPreservesGrid) {
    // If all bins have equal weight, the grid should not change significantly.
    const int nBins = 5;
    GridAllocator g(1, nBins);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    // Copy original boundaries for comparison
    std::vector<double> xiBefore = g.xi;

    for (int b = 0; b < nBins; ++b) {
        g.binAcc[b] = 1.0; g.binCounts[b] = 1;
    }
    AdaptGridWorkFunctor<10> adapt{g.grid};
    adapt(KernelAcc{});

    for (int b = 0; b <= nBins; ++b)
        EXPECT_NEAR(g.xi[b], xiBefore[b], 1e-12) << "boundary changed at b=" << b;
}

// =============================================================================
// VegasWorkFunctor — integration via grid_stride_reduce
// NOTE: RamboDietAlgorithm<3> has nRandomNumbers=5 which is ≤ MaxDim=10.
//       (Standard RamboAlgorithm<N> with N≥3 has 4*N≥12 > MaxDim=10 and would
//       overflow the stack-allocated x[] / bins[] arrays in VegasWorkFunctor.)
//
// SERIAL-ONLY: GridAllocator uses std::vector (host memory) and calls the
// initialiser functor directly from host code. On GPU backends, grid_stride_reduce
// would launch a kernel that dereferences those host pointers — undefined behaviour.
// The GPU path for VegasWorkFunctor is exercised by VegasIntegration.* tests, which
// go through runVegas() and properly allocate DeviceBuffer (device pointers).
// =============================================================================

#if defined(PHIRST_BACKEND_SERIAL)
TEST(VegasWorkFunctor, AccumulatesFiniteValues) {
    constexpr int NP = 3;
    using DietGen = PhaseSpaceGenerator<NP, RamboDietAlgorithm<NP>>;

    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy   = 200.0;
    uint64_t seed     = 5489ULL;

    const int nDim = DietGen::nRandomNumbers;  // 5
    const int nBins = 10;
    GridAllocator g(nDim, nBins);
    InitGridWorkFunctor init{g.grid};
    init(KernelAcc{});

    DietGen gen(masses);
    ConstantIntegrand integ(1.0);

    VegasWorkFunctor<DietGen, ConstantIntegrand, NP> work(
        gen, integ, cmEnergy, masses, seed, g.grid);

    const int64_t nWork = 64;
    double sum = 0.0, sum2 = 0.0;
    phirst::grid_stride_reduce(nWork, work, sum, sum2);

    EXPECT_TRUE(std::isfinite(sum));
    EXPECT_TRUE(std::isfinite(sum2));
    EXPECT_GT(sum, 0.0);
    EXPECT_GT(sum2, 0.0);

    // Cauchy-Schwarz: nWork * sum2 >= sum^2
    EXPECT_GE(static_cast<double>(nWork) * sum2, sum * sum - 1e-10);

    // binAcc should have received non-zero contributions
    double totalBinAcc = 0.0;
    for (auto v : g.binAcc) totalBinAcc += v;
    EXPECT_GT(totalBinAcc, 0.0);
}
#endif // PHIRST_BACKEND_SERIAL

// =============================================================================
// RamboIntegrator::runVegas() end-to-end
// =============================================================================

TEST(VegasIntegration, ConstantIntegrandFinitePositive) {
    constexpr int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy   = 200.0;
    int64_t nEvents   = 1000;

    ConstantIntegrand integ(1.0);
    RamboIntegrator<ConstantIntegrand, NP, RamboDietAlgorithm<NP>> integrator(nEvents, integ);

    double mean = 0.0, error = 0.0;
    integrator.runVegas(cmEnergy, masses, mean, error, 5489ULL);

    EXPECT_TRUE(std::isfinite(mean));
    EXPECT_TRUE(std::isfinite(error));
    EXPECT_GT(mean, 0.0);
    EXPECT_GE(error, 0.0);
}

TEST(VegasIntegration, SeedReproducibility) {
    // ConstantIntegrand + massless Diet logWeight (compile-time constant) makes
    // every event contribute the same value regardless of seed. Use DrellYanIntegrand
    // with N=2 Diet so that angular dependence on random numbers varies across seeds.
    constexpr int NP = 2;
    double masses[NP] = {0.0, 0.0};
    double cmEnergy   = 91.2;
    int64_t nEvents   = 1000;

    DrellYanIntegrand integ(2.0/3.0, 1.0/137.0);
    RamboIntegrator<DrellYanIntegrand, NP, RamboDietAlgorithm<NP>> integrator(nEvents, integ);

    double m1 = 0, e1 = 0, m2 = 0, e2 = 0, m3 = 0, e3 = 0;
    integrator.runVegas(cmEnergy, masses, m1, e1, 5489ULL);
    integrator.runVegas(cmEnergy, masses, m2, e2, 5489ULL);
    integrator.runVegas(cmEnergy, masses, m3, e3, 9999ULL);

    EXPECT_DOUBLE_EQ(m1, m2);  // same seed → identical result
    EXPECT_NE(m1, m3);         // different seed → different result
}

TEST(VegasIntegration, AgreesWithFlatIntegration) {
    // Both flat and VEGAS integration are unbiased estimators of the same integral.
    // With moderate N they should agree within a generous statistical tolerance.
    constexpr int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy   = 200.0;
    int64_t nEvents   = 5000;

    ConstantIntegrand integ(1.0);
    RamboIntegrator<ConstantIntegrand, NP, RamboDietAlgorithm<NP>> integrator(nEvents, integ);

    double mean_flat = 0.0, err_flat = 0.0;
    integrator.run(cmEnergy, masses, mean_flat, err_flat, 5489ULL);

    double mean_vegas = 0.0, err_vegas = 0.0;
    integrator.runVegas(cmEnergy, masses, mean_vegas, err_vegas, 5489ULL);

    EXPECT_TRUE(std::isfinite(mean_flat));
    EXPECT_TRUE(std::isfinite(mean_vegas));

    // Allow 10 combined standard errors to avoid flakiness
    double tolerance = 10.0 * (err_flat + err_vegas);
    EXPECT_NEAR(mean_vegas, mean_flat, tolerance);
}
