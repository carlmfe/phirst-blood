#include <gtest/gtest.h>

#include "phirst/phase_space.hpp"
#include "phirst/backend/random.hpp"

#include <cmath>
#include <limits>

using namespace phirst;

// =============================================================================
// Helper
// =============================================================================

template <int NP>
static void checkMomentumConservation(const double momenta[NP][4], double cmEnergy,
                                       double tol = 1e-10) {
    double sumE = 0, sumPx = 0, sumPy = 0, sumPz = 0;
    for (int i = 0; i < NP; ++i) {
        sumE  += momenta[i][0];
        sumPx += momenta[i][1];
        sumPy += momenta[i][2];
        sumPz += momenta[i][3];
    }
    EXPECT_NEAR(sumE,  cmEnergy, tol);
    EXPECT_NEAR(sumPx, 0.0,     tol);
    EXPECT_NEAR(sumPy, 0.0,     tol);
    EXPECT_NEAR(sumPz, 0.0,     tol);
}

template <int NP>
static void checkOnShell(const double momenta[NP][4], const double masses[NP], double tol = 1e-8) {
    for (int i = 0; i < NP; ++i) {
        double E  = momenta[i][0];
        double px = momenta[i][1], py = momenta[i][2], pz = momenta[i][3];
        double m2 = E*E - px*px - py*py - pz*pz;
        EXPECT_NEAR(m2, masses[i] * masses[i], tol) << "particle " << i;
        EXPECT_GT(E, 0.0) << "particle " << i << " has non-positive energy";
    }
}

// =============================================================================
// RamboAlgorithm – massless
// =============================================================================

TEST(PhaseSpaceRambo, TwoParticleMasslessConservation) {
    constexpr int NP = 2;
    double masses[NP] = {0.0, 0.0};
    double cmEnergy = 100.0;
    RamboAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 0);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
    checkMomentumConservation<NP>(mom, cmEnergy);
}

TEST(PhaseSpaceRambo, ThreeParticleMasslessConservation) {
    constexpr int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy = 200.0;
    RamboAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 1);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
    checkMomentumConservation<NP>(mom, cmEnergy);
}

TEST(PhaseSpaceRambo, FourParticleMasslessConservation) {
    constexpr int NP = 4;
    double masses[NP] = {0.0, 0.0, 0.0, 0.0};
    double cmEnergy = 500.0;
    RamboAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 2);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
    checkMomentumConservation<NP>(mom, cmEnergy);
}

// =============================================================================
// RamboAlgorithm – massive (Newton-Raphson conformal rescaling path)
// =============================================================================

TEST(PhaseSpaceRambo, TwoParticleMassiveConservationAndOnShell) {
    constexpr int NP = 2;
    double masses[NP] = {0.5, 0.5};  // 500 MeV each
    double cmEnergy = 10.0;          // well above threshold (1.0 GeV)
    RamboAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 3);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    checkMomentumConservation<NP>(mom, cmEnergy, 1e-8);
    checkOnShell<NP>(mom, masses, 1e-8);
}

TEST(PhaseSpaceRambo, ThreeParticleMassiveConservationAndOnShell) {
    constexpr int NP = 3;
    double masses[NP] = {1.0, 2.0, 3.0};
    double cmEnergy = 20.0;          // well above threshold (6.0 GeV)
    RamboAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 4);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    checkMomentumConservation<NP>(mom, cmEnergy, 1e-7);
    checkOnShell<NP>(mom, masses, 1e-7);
}

// =============================================================================
// RamboDietAlgorithm – random number count
// =============================================================================

TEST(PhaseSpaceDiet, RandomNumberCount) {
    // Diet uses 3*N-4 randoms (fewer than Rambo's 4*N)
    EXPECT_EQ(RamboDietAlgorithm<2>::nRandomNumbers, 2);
    EXPECT_EQ(RamboDietAlgorithm<3>::nRandomNumbers, 5);
    EXPECT_EQ(RamboDietAlgorithm<4>::nRandomNumbers, 8);
}

// =============================================================================
// RamboDietAlgorithm – massless
// =============================================================================

TEST(PhaseSpaceDiet, TwoParticleMasslessConservation) {
    constexpr int NP = 2;
    double masses[NP] = {0.0, 0.0};
    double cmEnergy = 100.0;
    RamboDietAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 10);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
    checkMomentumConservation<NP>(mom, cmEnergy);
}

// NOTE: The following two tests are expected to FAIL in the current source.
// RamboDietAlgorithm::boost() applies the boost in the wrong direction for N>=3
// (boostVec transforms from global to QPrev rest frame instead of QPrev rest
// to global frame). This exposes a suspected bug in the source code that should
// be referred to the C++ Development agent.

TEST(PhaseSpaceDiet, ThreeParticleMasslessConservation) {
    constexpr int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy = 200.0;
    RamboDietAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 11);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
    checkMomentumConservation<NP>(mom, cmEnergy);
}

TEST(PhaseSpaceDiet, FourParticleMasslessConservation) {
    constexpr int NP = 4;
    double masses[NP] = {0.0, 0.0, 0.0, 0.0};
    double cmEnergy = 500.0;
    RamboDietAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 12);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
    checkMomentumConservation<NP>(mom, cmEnergy);
}

// =============================================================================
// RamboDietAlgorithm – massive
// =============================================================================

TEST(PhaseSpaceDiet, TwoParticleMassiveConservationAndOnShell) {
    constexpr int NP = 2;
    double masses[NP] = {0.5, 0.5};
    double cmEnergy = 10.0;
    RamboDietAlgorithm<NP> algo(masses);

    uint64_t rng = initRngState(5489ULL, 13);
    double mom[NP][4];
    double logW = algo.generate(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    checkMomentumConservation<NP>(mom, cmEnergy, 1e-8);
    checkOnShell<NP>(mom, masses, 1e-8);
}

// =============================================================================
// PhaseSpaceGenerator wrapper with Diet algorithm as explicit template parameter
// =============================================================================

TEST(PhaseSpaceDiet, GeneratorWrapperInterface) {
    // Tests the PhaseSpaceGenerator wrapper interface with RamboDietAlgorithm.
    // Only checks compile-time properties and that generate() returns a finite
    // logWeight — NOT momentum conservation, which is separately tested.
    constexpr int NP = 3;
    double masses[NP] = {0.0, 0.0, 0.0};
    double cmEnergy = 91.2;

    PhaseSpaceGenerator<NP, RamboDietAlgorithm<NP>> gen(masses);
    EXPECT_EQ(gen.nRandomNumbers, 5);  // 3*3-4

    uint64_t rng = initRngState(5489ULL, 20);
    double mom[NP][4];
    double logW = gen(cmEnergy, rng, mom);

    EXPECT_TRUE(std::isfinite(logW));
    for (int i = 0; i < NP; ++i) EXPECT_GT(mom[i][0], 0.0);
}
