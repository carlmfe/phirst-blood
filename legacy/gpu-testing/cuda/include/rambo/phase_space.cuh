#pragma once
#ifndef RAMBO_CUDA_PHASE_SPACE_CUH
#define RAMBO_CUDA_PHASE_SPACE_CUH

#include <cmath>
#include <cstdint>

namespace rambo
{

    // =============================================================================
    // Random Number Generation (XorShift64 for CUDA)
    // =============================================================================

    __device__ __forceinline__ uint64_t xorshift64(uint64_t &state)
    {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }

    __device__ __forceinline__ auto uniformRandom(uint64_t &state) -> double
    {
        return (double)(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
    }

    // =============================================================================
    // RAMBO Algorithm Implementation
    // =============================================================================

    template <int nParticles>
    struct RamboAlgorithm
    {
        static constexpr double tolerance = 1e-14;
        static constexpr int maxIterations = 1000;
        static constexpr int nRandomNumbers = 4 * nParticles;

        // Pre-computed mathematical constants
        static constexpr double twoPi = 6.283185307179586476925286766559;
        static constexpr double logPiOver2 = 0.45158270528945486472619522989488;

        // Precomputed zCoeff[n-1] values for n = 1 to 10 (compile-time constant)
        static constexpr double zCoeffTable[10] = {
            0.0,                      // n = 1
            0.0,                      // n = 2
            -0.24156447527049046409,  // n = 3
            -1.58174123920909082130,  // n = 4
            -3.61506518370763618719,  // n = 5
            -6.15921475197217294095,  // n = 6
            -9.10882942834487252526,  // n = 7
            -12.39491634133878683599, // n = 8
            -15.96868532678448104889, // n = 9
            -19.79376874051108003982, // n = 10
        };
        static constexpr double zCoeffFinal = zCoeffTable[nParticles - 1];

        // Pre-computed quantities from masses
        double totalMass = 0.0;
        double totalMassSq = 0.0;
        int nMassive = 0;
        double massSq[nParticles] = {};

        __device__ __host__ RamboAlgorithm() = default;

        __device__ __host__ explicit RamboAlgorithm(const double *masses)
        {
            initializeMasses(masses);
        }

        __device__ __host__ void initializeMasses(const double *masses)
        {
            totalMass = 0.0;
            nMassive = 0;
            for (int i = 0; i < nParticles; ++i)
            {
                massSq[i] = masses[i] * masses[i];
                if (masses[i] != 0.0)
                    nMassive++;
                totalMass += fabs(masses[i]);
            }
            totalMassSq = totalMass * totalMass;
        }

        /**
         * Generic Newton-Raphson solver for finding roots of f(x) = 0.
         * @tparam FuncType Callable returning f(x).
         * @tparam DerivType Callable returning f'(x).
         * @tparam ClampType Callable to clamp x to valid bounds (optional).
         * @param x Initial guess; updated in-place with the solution.
         * @param f Function whose root is sought.
         * @param df Derivative of f.
         * @param tol Convergence tolerance on |f(x)|.
         * @param maxIter Maximum iterations.
         * @param clamp Function to clamp x after each step.
         * @return Number of iterations used.
         */
        template <typename FuncType, typename DerivType, typename ClampType>
        __device__ static auto newtonSolve(double &x, FuncType f, DerivType df,
                                           double tol, int maxIter, ClampType clamp) -> int
        {
            for (int iter = 0; iter <= maxIter; ++iter)
            {
                double fVal = f(x);
                if (fabs(fVal) <= tol)
                    return iter;
                double dfVal = df(x);
                if (dfVal == 0.0)
                    return maxIter + 1;
                x = x - fVal / dfVal;
                clamp(x);
            }
            return maxIter + 1;
        }

        /// Newton solver without clamping.
        template <typename FuncType, typename DerivType>
        __device__ static auto newtonSolve(double &x, FuncType f, DerivType df,
                                           double tol, int maxIter) -> int
        {
            return newtonSolve(x, f, df, tol, maxIter, [](double &) {});
        }

    public:
        __device__ auto generate(double cmEnergy, const double r[4 * nParticles],
                                 double momenta[][4]) const -> double
        {
            double q[nParticles][4];
            double p[nParticles][4];
            double totalMom[4];
            double boostVec[3];

            if (nParticles < 1 || nParticles > 10)
                return 0.0;

            for (int i = 0; i < nParticles; i++)
            {
                double cosTheta = 2.0 * r[4 * i] - 1.0;
                double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
                double phi = twoPi * r[4 * i + 1];

                q[i][0] = -log(r[4 * i + 2] * r[4 * i + 3]);
                q[i][3] = q[i][0] * cosTheta;
                q[i][2] = q[i][0] * sinTheta * cos(phi);
                q[i][1] = q[i][0] * sinTheta * sin(phi);
            }

            for (int mu = 0; mu < 4; mu++)
                totalMom[mu] = 0.0;
            for (int i = 0; i < nParticles; i++)
            {
                for (int mu = 0; mu < 4; mu++)
                {
                    totalMom[mu] += q[i][mu];
                }
            }

            double invariantMass = sqrt(totalMom[0] * totalMom[0] - totalMom[1] * totalMom[1] - totalMom[2] * totalMom[2] - totalMom[3] * totalMom[3]);
            for (int k = 0; k < 3; k++)
            {
                boostVec[k] = -totalMom[k + 1] / invariantMass;
            }
            double gamma = totalMom[0] / invariantMass;
            double boostFactor = 1.0 / (1.0 + gamma);
            double scaleFactor = cmEnergy / invariantMass;

            for (int i = 0; i < nParticles; ++i)
            {
                double bDotQ = boostVec[0] * q[i][1] + boostVec[1] * q[i][2] + boostVec[2] * q[i][3];
                for (int k = 1; k < 4; ++k)
                {
                    p[i][k] = scaleFactor * (q[i][k] + boostVec[k - 1] * (q[i][0] + boostFactor * bDotQ));
                }
                p[i][0] = scaleFactor * (gamma * q[i][0] + bDotQ);
            }

            double logWeight = logPiOver2;
            if (nParticles != 2)
            {
                logWeight += (2.0 * nParticles - 4.0) * log(cmEnergy) + zCoeffFinal;
            }

            if (nMassive == 0)
            {
                for (int i = 0; i < nParticles; ++i)
                {
                    for (int mu = 0; mu < 4; ++mu)
                    {
                        momenta[i][mu] = p[i][mu];
                    }
                }
                return logWeight;
            }

            // Make a conformal transformation to give particles mass
            double momSq[nParticles];
            double energies[nParticles];
            double virtMom[nParticles];

            double cmEnergySq = cmEnergy * cmEnergy;
            double x = sqrt(1.0 - totalMassSq / cmEnergySq);
            for (int i = 0; i < nParticles; ++i)
            {
                momSq[i] = p[i][0] * p[i][0];
            }

            double accuracyGoal = cmEnergy * tolerance;

            // Newton solve for conformal scaling factor x
            newtonSolve(x, [&](double xVal)
                        {
                    double f = -cmEnergy;
                    double x2 = xVal * xVal;
                    for (int i = 0; i < nParticles; ++i) {
                        energies[i] = sqrt(massSq[i] + x2 * momSq[i]);
                        f += energies[i];
                    }
                    return f; }, [&](double xVal)
                        {
                    double df = 0.0;
                    double x2 = xVal * xVal;
                    for (int i = 0; i < nParticles; ++i) {
                        double e = sqrt(massSq[i] + x2 * momSq[i]);
                        df += xVal * momSq[i] / e;
                    }
                    return df; }, accuracyGoal, maxIterations);

            for (int i = 0; i < nParticles; ++i)
            {
                virtMom[i] = x * p[i][0];
                for (int k = 1; k < 4; ++k)
                {
                    p[i][k] *= x;
                }
                p[i][0] = energies[i];
            }

            double weightProduct = 1.0;
            double weightSum = 0.0;
            for (int i = 0; i < nParticles; i++)
            {
                weightProduct *= virtMom[i] / energies[i];
                weightSum += (virtMom[i] * virtMom[i]) / energies[i];
            }
            double logWeightMassive = (2.0 * nParticles - 3.0) * log(x) + log(weightProduct / weightSum * cmEnergy);

            logWeight += logWeightMassive;

            for (int i = 0; i < nParticles; ++i)
            {
                for (int mu = 0; mu < 4; ++mu)
                {
                    momenta[i][mu] = p[i][mu];
                }
            }

            return logWeight;
        }

        __device__ auto generate(double cmEnergy, uint64_t &rngState,
                                 double momenta[][4]) const -> double
        {
            double r[4 * nParticles];
            for (int i = 0; i < 4 * nParticles; ++i)
            {
                r[i] = uniformRandom(rngState);
            }
            return generate(cmEnergy, r, momenta);
        }
    };

    // =============================================================================
    // RAMBO "Diet" variant (Plätzer) - CUDA device implementation
    // =============================================================================

    template <int nParticles>
    struct RamboDietAlgorithm
    {
        static constexpr double tolerance = 1e-14;
        static constexpr int maxIterations = 1000;
        static constexpr int nRandomNumbers = 3 * nParticles - 4;

        // Pre-computed mathematical constants
        static constexpr double twoPi = 6.283185307179586476925286766559;
        static constexpr double logPiOver2 = 0.45158270528945486472619522989488;

        // Precomputed zCoeff[n-1] values for n = 1 to 10 (compile-time constant)
        static constexpr double zCoeffTable[10] = {
            0.0,                      // n = 1
            0.0,                      // n = 2
            -0.24156447527049046409,  // n = 3
            -1.58174123920909082130,  // n = 4
            -3.61506518370763618719,  // n = 5
            -6.15921475197217294095,  // n = 6
            -9.10882942834487252526,  // n = 7
            -12.39491634133878683599, // n = 8
            -15.96868532678448104889, // n = 9
            -19.79376874051108003982  // n = 10
        };
        static constexpr double zCoeffFinal = zCoeffTable[nParticles - 1];

        // Pre-computed quantities from masses
        double totalMass = 0.0;
        double totalMassSq = 0.0;
        int nMassive = 0;
        double massSq[nParticles] = {};

        __device__ __host__ RamboDietAlgorithm() = default;

        __device__ __host__ explicit RamboDietAlgorithm(const double *masses)
        {
            initializeMasses(masses);
        }

        __device__ __host__ void initializeMasses(const double *masses)
        {
            totalMass = 0.0;
            nMassive = 0;
            for (int i = 0; i < nParticles; ++i)
            {
                massSq[i] = masses[i] * masses[i];
                if (masses[i] != 0.0)
                    nMassive++;
                totalMass += fabs(masses[i]);
            }
            totalMassSq = totalMass * totalMass;
        }

        /**
         * Generic Newton-Raphson solver for finding roots of f(x) = 0.
         * @tparam FuncType Callable returning f(x).
         * @tparam DerivType Callable returning f'(x).
         * @tparam ClampType Callable to clamp x to valid bounds (optional).
         * @param x Initial guess; updated in-place with the solution.
         * @param f Function whose root is sought.
         * @param df Derivative of f.
         * @param tol Convergence tolerance on |f(x)|.
         * @param maxIter Maximum iterations.
         * @param clamp Function to clamp x after each step.
         * @return Number of iterations used.
         */
        template <typename FuncType, typename DerivType, typename ClampType>
        __device__ static auto newtonSolve(double &x, FuncType f, DerivType df,
                                           double tol, int maxIter, ClampType clamp) -> int
        {
            for (int iter = 0; iter <= maxIter; ++iter)
            {
                double fVal = f(x);
                if (fabs(fVal) <= tol)
                    return iter;
                double dfVal = df(x);
                if (dfVal == 0.0)
                    return maxIter + 1;
                x = x - fVal / dfVal;
                clamp(x);
            }
            return maxIter + 1;
        }

        /// Newton solver without clamping.
        template <typename FuncType, typename DerivType>
        __device__ static auto newtonSolve(double &x, FuncType f, DerivType df,
                                           double tol, int maxIter) -> int
        {
            return newtonSolve(x, f, df, tol, maxIter, [](double &) {});
        }

    public:
        /**
         * Apply a Lorentz boost to a 4-vector `p` in-place.
         * @param p Array `[E, px, py, pz]` (modified in-place).
         * @param boostVec 3-component velocity vector `(beta_x, beta_y, beta_z)`.
         */
        __device__ auto boost(double p[4], const double *boostVec) const -> void
        {
            double b2 = boostVec[0] * boostVec[0] + boostVec[1] * boostVec[1] + boostVec[2] * boostVec[2];
            if (b2 >= 1.0 || b2 <= 0.0)
                return; // Invalid boost; leave 'p' unchanged.
            double gamma = 1.0 / sqrt(1.0 - b2);
            double bDotP = boostVec[0] * p[1] + boostVec[1] * p[2] + boostVec[2] * p[3];
            double factor = (gamma - 1.0) * bDotP / b2 - gamma * p[0];

            p[0] = gamma * (p[0] - bDotP);
            for (int k = 1; k < 4; ++k)
            {
                p[k] += boostVec[k - 1] * factor;
            }
        }

        /**
         * Evaluate the equation whose root is `u` (used by Newton iterations).
         * f(u) = r - m*u^(m-1) + (m-1)*u^m = 0
         * @param u Variable to evaluate.
         * @param r Random parameter from RNG.
         * @param m Dimension parameter (nParticles - index).
         * @return Value of the equation at `u`.
         */
        __device__ static auto uEquation(double u, double r, int m) -> double
        {
            return r - m * pow(u, m - 1) + (m - 1) * pow(u, m);
        }

        /**
         * Derivative of `uEquation` with respect to `u`.
         * @param u Point at which to compute derivative.
         * @param m Dimension parameter (nParticles - index).
         * @return d/du uEquation(u)
         */
        __device__ static auto dUEquation(double u, int m) -> double
        {
            return m * (m - 1) * (-pow(u, m - 2) + pow(u, m - 1));
        }

        /**
         * Newton solve for the intermediate variable `u` used in the Diet variant.
         * @param u Initial guess on input; updated with the solution on output.
         * @param r Uniform random number driving the equation.
         * @param index Current particle index (affects equation dimension).
         */
        __device__ inline auto solveForU(double &u, double r, int index) const -> void
        {
            const int m = nParticles - index;
            // Initial guess: u ~ r^(1/(m-1))
            u = pow(r, 1.0 / static_cast<double>(m - 1));

            // Clamp to valid range (0, 1)
            auto clampU = [](double &u)
            {
                if (u <= 0.0)
                    u = 1e-12;
                if (u >= 1.0)
                    u = 1.0 - 1e-12;
            };
            clampU(u);

            newtonSolve(u, [r, m](double u)
                        { return uEquation(u, r, m); }, [m](double u)
                        { return dUEquation(u, m); }, tolerance, maxIterations, clampU);
        }

        /**
         * Diet-variant generate using pre-computed masses.
         * @param cmEnergy Center-of-mass energy.
         * @param r Array of `3*nParticles-4` uniform random numbers.
         * @param momenta Output array `[nParticles][4]` for generated 4-momenta.
         * @return Natural logarithm of the phase-space weight.
         *
         * Note: Uses pre-computed mass quantities. Call initializeMasses() first if masses changed.
         */
        __device__ auto generate(double cmEnergy, const double r[3 * nParticles - 4],
                                 double momenta[nParticles][4]) const -> double
        {
            if (totalMass > cmEnergy)
                return -INFINITY;

            double p[nParticles][4];

            // === Generate phase space for massless particles first ===
            // N = 2 case is simple, with no intermediate masses
            if constexpr (nParticles == 2)
            {
                double cosTheta = 2.0 * r[0] - 1.0;
                double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
                double phi = twoPi * r[1];
                double q = 0.5 * cmEnergy;

                p[0][0] = q;
                p[0][1] = q * sinTheta * cos(phi);
                p[0][2] = q * sinTheta * sin(phi);
                p[0][3] = q * cosTheta;

                p[1][0] = q;
                p[1][1] = -p[0][1];
                p[1][2] = -p[0][2];
                p[1][3] = -p[0][3];
            }
            else
            {
                double QPrev[4]{cmEnergy, 0.0, 0.0, 0.0};
                double QCurr[4]{cmEnergy, 0.0, 0.0, 0.0};
                double MPrev = cmEnergy;
                double MCurr = MPrev;
                double boostVec[3];
                double cosTheta, sinTheta, phi, q;

                // u determines intermediate mass ratios – only needed for N > 3
                [[maybe_unused]] double u[nParticles > 3 ? nParticles - 2 : 1];

                for (int i = 1; i < nParticles; ++i)
                {
                    if constexpr (nParticles == 3)
                    {
                        MCurr = (i == 1) ? (1.0 - sqrt(1.0 - r[0])) * MPrev : 0.0;
                    }
                    else
                    {
                        if (i < nParticles - 1)
                        {
                            solveForU(u[i - 1], r[i - 1], i);
                            MCurr = u[i - 1] * MPrev;
                        }
                        else
                        {
                            MCurr = 0.0;
                        }
                    }

                    cosTheta = 2.0 * r[nParticles - 4 + 2 * i] - 1.0;
                    sinTheta = sqrt(1.0 - cosTheta * cosTheta);
                    phi = twoPi * r[nParticles - 3 + 2 * i];
                    q = 0.5 * (MPrev * MPrev - MCurr * MCurr) / MPrev;

                    p[i - 1][0] = q;
                    p[i - 1][1] = q * sinTheta * cos(phi);
                    p[i - 1][2] = q * sinTheta * sin(phi);
                    p[i - 1][3] = q * cosTheta;

                    QCurr[0] = sqrt(q * q + MCurr * MCurr);
                    QCurr[1] = -p[i - 1][1];
                    QCurr[2] = -p[i - 1][2];
                    QCurr[3] = -p[i - 1][3];

                    if constexpr (nParticles > 2)
                    {
                        if (i > 1)
                        {
                            boostVec[0] = QPrev[1] / QPrev[0];
                            boostVec[1] = QPrev[2] / QPrev[0];
                            boostVec[2] = QPrev[3] / QPrev[0];
                            boost(p[i - 1], boostVec);
                            boost(QCurr, boostVec);
                        }

                        MPrev = MCurr;
                        for (int k = 0; k < 4; ++k)
                        {
                            QPrev[k] = QCurr[k];
                        }
                    }
                }

                // Last particle
                for (int k = 0; k < 4; ++k)
                {
                    p[nParticles - 1][k] = QCurr[k];
                }
            }

            // === Compute weight for massless configuration ===
            double logWeight = logPiOver2;
            if constexpr (nParticles > 2)
            {
                logWeight += (2.0 * nParticles - 4.0) * log(cmEnergy) + zCoeffFinal;
            }

            // === If all particles ARE massless, we are done now ===
            if (nMassive == 0)
            {
                for (int i = 0; i < nParticles; ++i)
                {
                    for (int mu = 0; mu < 4; ++mu)
                    {
                        momenta[i][mu] = p[i][mu];
                    }
                }
                return logWeight;
            }

            // Make a conformal transformation to give particles mass
            double momSq[nParticles];
            double energies[nParticles];
            double virtMom[nParticles];

            double cmEnergySq = cmEnergy * cmEnergy;
            double x = sqrt(1.0 - totalMassSq / cmEnergySq);
            for (int i = 0; i < nParticles; ++i)
            {
                momSq[i] = p[i][0] * p[i][0];
            }

            double accuracyGoal = cmEnergy * tolerance;

            // Newton solve for conformal scaling factor x
            newtonSolve(x, [&](double xVal)
                        {
                    double f = -cmEnergy;
                    double x2 = xVal * xVal;
                    for (int i = 0; i < nParticles; ++i) {
                        energies[i] = sqrt(massSq[i] + x2 * momSq[i]);
                        f += energies[i];
                    }
                    return f; }, [&](double xVal)
                        {
                    double df = 0.0;
                    double x2 = xVal * xVal;
                    for (int i = 0; i < nParticles; ++i) {
                        double e = sqrt(massSq[i] + x2 * momSq[i]);
                        df += xVal * momSq[i] / e;
                    }
                    return df; }, accuracyGoal, maxIterations);

            for (int i = 0; i < nParticles; ++i)
            {
                virtMom[i] = x * p[i][0];
                for (int k = 1; k < 4; ++k)
                {
                    p[i][k] *= x;
                }
                p[i][0] = energies[i];
            }

            double weightProduct = 1.0;
            double weightSum = 0.0;
            for (int i = 0; i < nParticles; ++i)
            {
                weightProduct *= virtMom[i] / energies[i];
                weightSum += (virtMom[i] * virtMom[i]) / energies[i];
            }
            double logWeightMassive = (2.0 * nParticles - 3.0) * log(x) + log(weightProduct / weightSum * cmEnergy);

            logWeight += logWeightMassive;

            for (int i = 0; i < nParticles; ++i)
            {
                for (int mu = 0; mu < 4; ++mu)
                {
                    momenta[i][mu] = p[i][mu];
                }
            }

            return logWeight;
        }

        __device__ auto generate(double cmEnergy, uint64_t &rngState,
                                 double momenta[nParticles][4]) const -> double
        {
            constexpr int numRandoms = 3 * nParticles - 4;
            double r[numRandoms > 0 ? numRandoms : 1];
            for (int i = 0; i < numRandoms; i++)
            {
                r[i] = uniformRandom(rngState);
            }
            return generate(cmEnergy, r, momenta);
        }
    };

    // =============================================================================
    // Phase Space Generator (Wrapper)
    // =============================================================================

    template <int nParticles, typename Algorithm = RamboAlgorithm<nParticles>>
    struct PhaseSpaceGenerator
    {
        double cmEnergy;
        Algorithm algorithm;

        static constexpr int nRandomNumbers = Algorithm::nRandomNumbers;

        __device__ __host__ PhaseSpaceGenerator(double energy, const double *masses)
            : cmEnergy(energy), algorithm(masses) {}

        __device__ auto operator()(uint64_t &rngState, double momenta[][4]) const -> double
        {
            return algorithm.generate(cmEnergy, rngState, momenta);
        }

        __device__ auto operator()(const double *r, double momenta[][4]) const -> double
        {
            return algorithm.generate(cmEnergy, r, momenta);
        }
    };

    template <int nParticles>
    using DefaultPhaseSpaceGenerator = PhaseSpaceGenerator<nParticles, RamboAlgorithm<nParticles>>;

} // namespace rambo

#endif // RAMBO_CUDA_PHASE_SPACE_CUH
