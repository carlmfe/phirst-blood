#include <gtest/gtest.h>
#include "test_utils.hpp"

#include "phirst/phirst.hpp"
#include "phirst/phase_space.hpp"
#include "phirst/backend/random.hpp"

using namespace phirst;

TEST(Integrands, Constant) {
    ConstantIntegrand c(4.2);
    HEPUtils::P4 mom[1];
    EXPECT_DOUBLE_EQ(c.evaluate(mom), 4.2);
}

TEST(Integrands, EggholderZeroForIdenticalMomenta) {
    EggholderIntegrand e(1e6);
    HEPUtils::P4 mom[3];
    mom[0] = HEPUtils::P4::mkXYZM(0.0, 0.0, 0.0, 0.0);
    mom[1] = mom[0];
    mom[2] = mom[0];
    double val = e.evaluate(mom);
    EXPECT_NEAR(val, 0.0, 1e-12);
}

TEST(Integrands, MandelstamSWorks) {
    const int NP = 2;
    HEPUtils::P4 mom[NP];
    mom[0] = HEPUtils::P4::mkXYZE(1.0, 0.0, 0.0, 2.0); // E consistent with p
    mom[1] = HEPUtils::P4::mkXYZE(0.0, 2.0, 0.0, 3.0);

    MandelstamSIntegrand<NP> msi(1.0);
    double s_over_scale = msi.evaluate(mom);

    HEPUtils::P4 P = mom[0] + mom[1];
    double expected = P.dot(P) / 1.0;
    EXPECT_NEAR(s_over_scale, expected, 1e-12);
}

TEST(Integrands, DrellYanAnalyticAndEvaluate) {
    double s = 100.0;
    double eq = 2.0 / 3.0;
    double alpha = 1.0 / 137.035999;

    double analytic = DrellYanIntegrand::analyticCrossSection(s, eq, alpha);
    // recompute manually for sanity
    constexpr double hbarc2 = 0.3893793656;
    double expected = 4.0 * math::pi * alpha * alpha * eq * eq / (3.0 * s) * hbarc2;
    EXPECT_NEAR(analytic, expected, 1e-15);

    // Evaluate at a simple kinematic point (should be finite and non-negative)
    DrellYanIntegrand dyn(eq, alpha);
    HEPUtils::P4 k1 = HEPUtils::P4::mkXYZE(0.0, 0.0, +5.0, 5.0);
    HEPUtils::P4 k2 = HEPUtils::P4::mkXYZE(0.0, 0.0, -5.0, 5.0);
    HEPUtils::P4 mom[2] = {k1, k2};
    double val = dyn.evaluate(mom);
    EXPECT_TRUE(isfinite(val));
}

TEST(Integrands, EggholderWithRealisticMomenta) {
    // Generate two distinct 3-particle phase-space points and verify finite, differing results.
    constexpr int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy = 200.0;

    RamboAlgorithm<NP> algo(masses);

    // Point A
    uint64_t rngA = initRngState(5489ULL, 42);
    double momA[NP][4];
    algo.generate(cmEnergy, rngA, momA);
    HEPUtils::P4 p4A[NP];
    for (int i = 0; i < NP; ++i)
        p4A[i] = HEPUtils::P4::mkXYZM(momA[i][1], momA[i][2], momA[i][3], 0.0);

    // Point B (different seed)
    uint64_t rngB = initRngState(12345ULL, 7);
    double momB[NP][4];
    algo.generate(cmEnergy, rngB, momB);
    HEPUtils::P4 p4B[NP];
    for (int i = 0; i < NP; ++i)
        p4B[i] = HEPUtils::P4::mkXYZM(momB[i][1], momB[i][2], momB[i][3], 0.0);

    EggholderIntegrand e(1e6);
    double valA = e.evaluate(p4A);
    double valB = e.evaluate(p4B);

    EXPECT_TRUE(isfinite(valA));
    EXPECT_TRUE(isfinite(valB));
    // Two random phase-space points give different integrand values
    EXPECT_NE(valA, valB);
}
