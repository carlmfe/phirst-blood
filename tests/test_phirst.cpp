#include <gtest/gtest.h>

#include "phirst/backend/math.hpp"
#include "phirst/backend/random.hpp"
#include "phirst/phase_space.hpp"
#include "phirst/backend/parallel.hpp"

#include <cmath>
#include <limits>

using namespace phirst;

// File-scope work functor for the grid_stride_reduce test
// (local classes cannot have template member functions in C++)
struct SumWork {
    template <typename Acc>
    PHIRST_HOST_DEVICE void operator()(const Acc&, int64_t idx, double &acc1, double &acc2) const {
        acc1 += static_cast<double>(idx);
        acc2 += static_cast<double>(idx) * static_cast<double>(idx);
    }
};

TEST(MathBasics, SqrtPiSin) {
    EXPECT_DOUBLE_EQ(math::sqrt(4.0), 2.0);
    EXPECT_NEAR(math::sin(math::pi / 2.0), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(math::pi, 3.14159265358979323846264338327950288);
}

TEST(Random, UniformRangeAndInt) {
    uint64_t state = 123456789ULL;
    double u1 = uniformRandom(state);
    EXPECT_GE(u1, 0.0);
    EXPECT_LT(u1, 1.0);

    uint64_t state2 = 424242ULL;
    double a = -2.5, b = 7.3;
    double u2 = uniformRandom(state2, a, b);
    EXPECT_GE(u2, a);
    EXPECT_LT(u2, b);

    uint64_t state3 = 99ULL;
    int64_t n = 100;
    int64_t r = uniformRandomInt(state3, n);
    EXPECT_GE(r, 0);
    EXPECT_LT(r, n);
}

TEST(PhaseSpace, TwoParticleEnergyConservation) {
    double masses[2] = {0.0, 0.0};
    PhaseSpaceGenerator<2> gen(masses);
    uint64_t rng = initRngState(5489ULL, 0);
    double mom[2][4];
    double logW = gen(100.0, rng, mom);
    double sumE = mom[0][0] + mom[1][0];
    EXPECT_NEAR(sumE, 100.0, 1e-10);
    EXPECT_TRUE(std::isfinite(logW));
}

TEST(Parallel, GridStrideReduceSum) {
    int64_t nWork = 1000;
    SumWork work;

    double sum = 0.0, sumSq = 0.0;
    phirst::grid_stride_reduce(nWork, work, sum, sumSq);

    double expectedSum = static_cast<double>(nWork) * static_cast<double>(nWork - 1) / 2.0;
    double expectedSumSq = static_cast<double>(nWork - 1) * static_cast<double>(nWork) * static_cast<double>(2 * nWork - 1) / 6.0;

    EXPECT_DOUBLE_EQ(sum, expectedSum);
    EXPECT_DOUBLE_EQ(sumSq, expectedSumSq);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
