#pragma once
#ifndef RAMBO_BASE_INTEGRANDS_HPP
#define RAMBO_BASE_INTEGRANDS_HPP

#include <cmath>

namespace rambo {

// =============================================================================
// Eggholder integrand
// =============================================================================

// Toy integrand used for testing; depends on three final-state momenta.
struct EggholderIntegrand {
    double lambdaSquared;
    
    EggholderIntegrand(double lambda = 1000000.0) 
        : lambdaSquared(lambda) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
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
        
        const double arg1 = std::fabs((s12 - s23) / lambdaSquared);
        const double arg2 = std::fabs(s13 / lambdaSquared);
        return std::sin(std::sqrt(arg1)) * std::cos(std::sqrt(arg2));
    }
};

// =============================================================================
// Constant integrand
// =============================================================================

// Returns a constant value regardless of momenta (useful for sanity checks).
struct ConstantIntegrand {
    double value;
    
    ConstantIntegrand(double v = 1.0) : value(v) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        (void)momenta;
        return value;
    }
};

// =============================================================================
// Drell-Yan integrand
// =============================================================================

// Leading-order Drell-Yan style matrix element for q qbar -> l+ l- (toy model).
struct DrellYanIntegrand {
    double quarkCharge;
    double alphaEM;
    
    DrellYanIntegrand(double eq = 2.0/3.0, double alpha = 1.0/137.035999)
        : quarkCharge(eq), alphaEM(alpha) {}
    
    auto evaluate(const double momenta[][4]) const -> double {
        double Ptot[4];
        for (int mu = 0; mu < 4; ++mu) {
            Ptot[mu] = momenta[0][mu] + momenta[1][mu];
        }
        
        double s = Ptot[0]*Ptot[0] - Ptot[1]*Ptot[1] 
                 - Ptot[2]*Ptot[2] - Ptot[3]*Ptot[3];
        
        if (s <= 0.0) return 0.0;
        
        double sqrtS = std::sqrt(s);
        double p1[4] = {sqrtS/2.0, 0.0, 0.0, +sqrtS/2.0};
        
        const double* k1 = momenta[0];
        const double* k2 = momenta[1];
        
        double t = -2.0 * (p1[0]*k1[0] - p1[1]*k1[1] - p1[2]*k1[2] - p1[3]*k1[3]);
        double u = -2.0 * (p1[0]*k2[0] - p1[1]*k2[1] - p1[2]*k2[2] - p1[3]*k2[3]);
        
        constexpr double PI = 3.14159265358979323846;
        double e4 = 16.0 * PI * PI * alphaEM * alphaEM;
        double eq2 = quarkCharge * quarkCharge;
        double Msq = 2.0 * e4 * eq2 * (t*t + u*u) / (s*s);
        
        double dsigma = Msq / (2.0 * s) / (4.0 * PI * PI);
        
        constexpr double hbarc2 = 0.3893793656;
        return dsigma * hbarc2;
    }
    
    static auto analyticCrossSection(double s, double eq, double alpha) -> double {
        constexpr double PI = 3.14159265358979323846;
        constexpr double hbarc2 = 0.3893793656;
        return 4.0 * PI * alpha * alpha * eq * eq / (3.0 * s) * hbarc2;
    }
};

} // namespace rambo

#endif // RAMBO_BASE_INTEGRANDS_HPP
 