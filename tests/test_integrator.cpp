#include <gtest/gtest.h>
#include "test_utils.hpp"

#include "phirst/phirst.hpp"
#include "phirst/backend/parallel.hpp"

using namespace phirst;

TEST(IntegrationResult, ComputeStatistics) {
    IntegrationResult r;
    r.nEvents = 4;
    r.sum = 10.0; // 1+2+3+4
    r.sum2 = 30.0; // 1+4+9+16
    r.computeStatistics();
    EXPECT_DOUBLE_EQ(r.mean, 2.5);
    double variance = (r.sum2 / static_cast<double>(r.nEvents)) - (r.mean * r.mean);
    double expectedError = math::sqrt(math::fabs(variance) / static_cast<double>(r.nEvents));
    EXPECT_DOUBLE_EQ(r.error, expectedError);
}

// Mock generator that returns logWeight = 0 and fills momenta with zeros.
// nRandomNumbers = 1 (not 0) to avoid zero-sized stack arrays in CUDA device code.
template <int NP>
struct MockGenerator {
    static constexpr int nRandomNumbers = 1;

    PHIRST_HOST_DEVICE auto operator()(double /*cmEnergy*/, uint64_t& /*rngState*/, double momenta[][4]) const -> double {
        for (int i = 0; i < NP; ++i) {
            for (int k = 0; k < 4; ++k) momenta[i][k] = 0.0;
        }
        return 0.0; // log weight = 0 -> weight = 1
    }

    PHIRST_HOST_DEVICE auto operator()(double /*cmEnergy*/, const double* /*r*/, double momenta[][4]) const -> double {
        for (int i = 0; i < NP; ++i) {
            for (int k = 0; k < 4; ++k) momenta[i][k] = 0.0;
        }
        return 0.0; // log weight = 0 -> weight = 1
    }
};

TEST(MCWorkFunctor, AccumulatesCorrectly) {
    const int NP = 2;
    double masses[NP] = {0.0, 0.0};
    MockGenerator<NP> gen;
    ConstantIntegrand integ(3.0);

    MCWorkFunctor<MockGenerator<NP>, ConstantIntegrand, NP> work(gen, integ, 100.0, masses, 12345ULL);

    int64_t nWork = 37;
    double sum = 0.0, sumSq = 0.0;
    phirst::grid_stride_reduce(nWork, work, sum, sumSq);

    EXPECT_DOUBLE_EQ(sum, 3.0 * static_cast<double>(nWork));
    EXPECT_DOUBLE_EQ(sumSq, 9.0 * static_cast<double>(nWork));
}

// Compare RamboIntegrator.run with direct sampling using the same generator
TEST(RamboIntegrator, MatchesDirectSampling) {
    const int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    ConstantIntegrand integ(2.5);
    int64_t nEvents = 256;
    double cmEnergy = 91.2;
    uint64_t seed = 5489ULL;

    // Run integrator
    RamboIntegrator<ConstantIntegrand, NP> integrator(nEvents, integ);
    double mean_int = 0.0, err_int = 0.0;
    integrator.run(cmEnergy, masses, mean_int, err_int, seed);

    // Direct sampling
    using GenT = PhaseSpaceGenerator<NP, RamboAlgorithm<NP>>;
    GenT gen(masses);

    double sum = 0.0, sumSq = 0.0;
    for (int64_t i = 0; i < nEvents; ++i) {
        uint64_t rng = seed_for_thread(seed, i);
        double raw[NP][4];
        double logW = gen(cmEnergy, rng, raw);
        double w = phirst::math::exp(logW);
        double fx = integ.evaluate(reinterpret_cast<HEPUtils::P4*>(raw)); // ConstantIntegrand ignores momenta but types match
        double val = fx * w;
        sum += val;
        sumSq += val * val;
    }

    double mean_direct = sum / static_cast<double>(nEvents);
    double variance = (sumSq / static_cast<double>(nEvents)) - (mean_direct * mean_direct);
    double err_direct = math::sqrt(math::fabs(variance) / static_cast<double>(nEvents));

    EXPECT_NEAR(mean_int, mean_direct, 1e-12);
    EXPECT_NEAR(err_int, err_direct, 1e-12);
}

TEST(IntegrationResult, ZeroEventsIsNoOp) {
    IntegrationResult r;
    r.nEvents = 0;
    r.sum = 999.0;   // should be ignored
    r.sum2 = 999.0;
    r.computeStatistics();
    // mean and error remain at their default 0.0
    EXPECT_DOUBLE_EQ(r.mean, 0.0);
    EXPECT_DOUBLE_EQ(r.error, 0.0);
}

TEST(IntegrationResult, IdenticalValuesGiveZeroError) {
    // All 100 samples equal 3.5 → variance = 0 → error = 0
    IntegrationResult r;
    r.nEvents = 100;
    r.sum  = 100 * 3.5;
    r.sum2 = 100 * 3.5 * 3.5;
    r.computeStatistics();
    EXPECT_NEAR(r.mean, 3.5, 1e-12);
    EXPECT_NEAR(r.error, 0.0, 1e-12);
}

TEST(IntegrationResult, NegativeVarianceClamped) {
    // Arrange sum/sum2 so that floating-point cancellation yields a tiny negative variance.
    // mean = 1.5, N*mean^2 = 4.5; set sum2 just below 4.5 so variance ≈ -epsilon.
    IntegrationResult r;
    r.nEvents = 2;
    r.sum  = 3.0;
    r.sum2 = 4.5 - 1e-14;
    r.computeStatistics();
    // fabs(variance) prevents sqrt from receiving a negative argument
    EXPECT_TRUE(isfinite(r.error));
    EXPECT_GE(r.error, 0.0);
}
