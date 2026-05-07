#pragma once
#ifndef PHIRST_INTEGRANDS_HPP
#define PHIRST_INTEGRANDS_HPP

/**
 * @file integrands.hpp
 * @brief Portable integrand implementations for Monte Carlo integration.
 * 
 * Each integrand must provide an `evaluate(const HEPUtils::P4 momenta[])` method
 * that computes the physics quantity from 4-momenta.
 */

#include "backend/config.hpp"
#include "backend/math.hpp"
#include "contrib/HEPUtils/Vectors.h"

#include <iostream>

namespace phirst {

// =============================================================================
// Eggholder integrand
// =============================================================================

/**
 * Toy integrand used for testing; depends on three final-state momenta.
 * Based on the "Eggholder" test function with oscillatory behavior.
 */
struct EggholderIntegrand {
    double lambdaSquared;

    PHIRST_HOST_DEVICE
    EggholderIntegrand(double lambda = 1000000.0) 
        : lambdaSquared(lambda) {}

    PHIRST_HOST_DEVICE
    auto evaluate(const HEPUtils::P4 momenta[]) const -> double {
        // Compute Lorentz invariants directly as s_ij = m_i^2 + m_j^2 - 2*p_i·p_j.
        // Do NOT use P4 subtraction: operator-= clamps the mass of space-like
        // differences to 0, turning the real negative invariant into floating-point
        // noise that gets amplified by the large phase-space weight.
        double s12 = momenta[0].m2() + momenta[1].m2() - 2.0 * momenta[0].dot(momenta[1]);
        double s13 = momenta[0].m2() + momenta[2].m2() - 2.0 * momenta[0].dot(momenta[2]);
        double s23 = momenta[1].m2() + momenta[2].m2() - 2.0 * momenta[1].dot(momenta[2]);

        const double arg1 = math::fabs((s12 - s23) / lambdaSquared);
        const double arg2 = math::fabs(s13 / lambdaSquared);
        return math::sin(math::sqrt(arg1)) * math::cos(math::sqrt(arg2));
    }
};

// =============================================================================
// Constant integrand
// =============================================================================

/**
 * Returns a constant value regardless of momenta.
 * Useful for sanity checks and verifying phase space integration.
 */
struct ConstantIntegrand {
    double value;

    PHIRST_HOST_DEVICE
    ConstantIntegrand(double v = 1.0) : value(v) {}

    PHIRST_HOST_DEVICE
    auto evaluate(const HEPUtils::P4 momenta[]) const -> double {
        (void)momenta;
        return value;
    }
};

// =============================================================================
// Drell-Yan integrand
// =============================================================================

/**
 * Leading-order Drell-Yan style matrix element for q qbar -> l+ l-.
 * This is a pedagogical toy model of the electroweak process.
 */
struct DrellYanIntegrand {
    double quarkCharge;
    double alphaEM;

    PHIRST_HOST_DEVICE
    DrellYanIntegrand(double eq = 2.0/3.0, double alpha = 1.0/137.035999)
        : quarkCharge(eq), alphaEM(alpha) {}

    PHIRST_HOST_DEVICE
    auto evaluate(const HEPUtils::P4 momenta[]) const -> double {
        HEPUtils::P4 Ptot = momenta[0] + momenta[1];
        double sqrtS = Ptot.m();
        double s = sqrtS * sqrtS;
        if (sqrtS <= 0.0) { return 0.0; }

        HEPUtils::P4 p1 = HEPUtils::P4::mkXYZM(0.0, 0.0, +sqrtS / 2.0, 0.0);

        HEPUtils::P4 k1 = momenta[0];
        HEPUtils::P4 k2 = momenta[1];

        double t = p1.m2() + k1.m2() - 2.0 * p1.dot(k1);
        double u = p1.m2() + k2.m2() - 2.0 * p1.dot(k2);

        double e4 = 16.0 * math::pi * math::pi * alphaEM * alphaEM;
        double eq2 = quarkCharge * quarkCharge;
        double Msq = 2.0 * e4 * eq2 * (t*t + u*u) / (s*s);

        double dsigma = Msq / (2.0 * s) / (4.0 * math::pi * math::pi);

        constexpr double hbarc2 = 0.3893793656;  // GeV^-2 to mb conversion
        return dsigma * hbarc2;
    }
    /**
     * Analytic cross-section for comparison.
     * @param s Center-of-mass energy squared (GeV^2).
     * @param eq Quark charge in units of e.
     * @param alpha Fine structure constant.
     * @return Cross-section in millibarns.
     */
    PHIRST_HOST_DEVICE
    static auto analyticCrossSection(double s, double eq, double alpha) -> double {
        constexpr double hbarc2 = 0.3893793656;
        return 4.0 * math::pi * alpha * alpha * eq * eq / (3.0 * s) * hbarc2;
    }
};

// =============================================================================
// Mandelstam S integrand
// =============================================================================

/**
 * Simple integrand that returns the Mandelstam s variable (invariant mass squared).
 * Useful for testing and debugging.
 */
template <int nParticles>
struct MandelstamSIntegrand {
    double scale;

    PHIRST_HOST_DEVICE
    MandelstamSIntegrand(double s = 1.0) : scale(s) {}

    PHIRST_HOST_DEVICE
    auto evaluate(const HEPUtils::P4 momenta[]) const -> double {
        HEPUtils::P4 Ptot = momenta[0];
        for (int i = 1; i < nParticles; ++i) {
            Ptot += momenta[i];
        }

        double s = Ptot.dot(Ptot);
        return s / scale;
    }
};

} // namespace phirst

#endif // PHIRST_INTEGRANDS_HPP
