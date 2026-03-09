#pragma once
#ifndef RAMBO_INTEGRANDS_HPP
#define RAMBO_INTEGRANDS_HPP

/**
 * @file integrands.hpp
 * @brief Portable integrand implementations for Monte Carlo integration.
 * 
 * Each integrand must provide an `evaluate(const double momenta[][4])` method
 * that computes the physics quantity from 4-momenta.
 */

#include "pal/backend.hpp"
#include "pal/math.hpp"

namespace rambo {

// =============================================================================
// Eggholder integrand
// =============================================================================

/**
 * Toy integrand used for testing; depends on three final-state momenta.
 * Based on the "Eggholder" test function with oscillatory behavior.
 */
struct EggholderIntegrand {
    double lambdaSquared;
    
    RAMBO_HOST_DEVICE EggholderIntegrand(double lambda = 1000000.0) 
        : lambdaSquared(lambda) {}
    
    RAMBO_HOST_DEVICE auto evaluate(const double momenta[][4]) const -> double {
        double s12 = 0.0, s13 = 0.0, s23 = 0.0;
        
        for (int mu = 0; mu < 4; ++mu) {
            const double d12 = momenta[0][mu] - momenta[1][mu];
            const double d13 = momenta[0][mu] - momenta[2][mu];
            const double d23 = momenta[1][mu] - momenta[2][mu];
            const double sign = (mu == 0) ? 1.0 : -1.0;
            s12 += sign * d12 * d12;
            s13 += sign * d13 * d13;
            s23 += sign * d23 * d23;
        }
        
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
    
    RAMBO_HOST_DEVICE ConstantIntegrand(double v = 1.0) : value(v) {}
    
    RAMBO_HOST_DEVICE auto evaluate(const double momenta[][4]) const -> double {
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
    
    RAMBO_HOST_DEVICE DrellYanIntegrand(double eq = 2.0/3.0, double alpha = 1.0/137.035999)
        : quarkCharge(eq), alphaEM(alpha) {}
    
    RAMBO_HOST_DEVICE auto evaluate(const double momenta[][4]) const -> double {
        double Ptot[4];
        for (int mu = 0; mu < 4; ++mu) {
            Ptot[mu] = momenta[0][mu] + momenta[1][mu];
        }
        
        double s = Ptot[0]*Ptot[0] - Ptot[1]*Ptot[1] 
                 - Ptot[2]*Ptot[2] - Ptot[3]*Ptot[3];
        
        if (s <= 0.0) return 0.0;
        
        double sqrtS = math::sqrt(s);
        double p1[4] = {sqrtS/2.0, 0.0, 0.0, +sqrtS/2.0};
        
        const double* k1 = momenta[0];
        const double* k2 = momenta[1];
        
        double t = -2.0 * (p1[0]*k1[0] - p1[1]*k1[1] - p1[2]*k1[2] - p1[3]*k1[3]);
        double u = -2.0 * (p1[0]*k2[0] - p1[1]*k2[1] - p1[2]*k2[2] - p1[3]*k2[3]);
        
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
    RAMBO_HOST_DEVICE static auto analyticCrossSection(double s, double eq, double alpha) -> double {
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
    
    RAMBO_HOST_DEVICE MandelstamSIntegrand(double s = 1.0) : scale(s) {}
    
    RAMBO_HOST_DEVICE auto evaluate(const double momenta[][4]) const -> double {
        double Ptot[4] = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < nParticles; ++i) {
            for (int mu = 0; mu < 4; ++mu) {
                Ptot[mu] += momenta[i][mu];
            }
        }
        
        double s = Ptot[0]*Ptot[0] - Ptot[1]*Ptot[1] 
                 - Ptot[2]*Ptot[2] - Ptot[3]*Ptot[3];
        
        return s / scale;
    }
};

} // namespace rambo

#endif // RAMBO_INTEGRANDS_HPP
