#include "phirst/backend/numba_bridge.cuh"

#include "phirst/backend/math.hpp"
#include "phirst/backend/random.hpp"

namespace {

constexpr int kMaxNumbaParticles = 10;
constexpr double kMinLogArg = 1.0e-300;

__device__ auto z_coeff(int nParticles) -> double {
    switch (nParticles) {
        case 3:
            return -0.24156447527049046409;
        case 4:
            return -1.58174123920909082130;
        case 5:
            return -3.61506518370763618719;
        case 6:
            return -6.15921475197217294095;
        case 7:
            return -9.10882942834487252526;
        case 8:
            return -12.39491634133878683599;
        case 9:
            return -15.96868532678448104889;
        case 10:
            return -19.79376874051108003982;
        default:
            return 0.0;
    }
}

__device__ auto massless_rambo_generate(
    double cmEnergy,
    int nParticles,
    uint64_t& rng,
    double* momentaFlat) -> double {
    if (nParticles < 2 || nParticles > kMaxNumbaParticles) {
        return 0.0;
    }

    double q[kMaxNumbaParticles * 4] = {};
    double totalMom[4] = {};

    for (int i = 0; i < nParticles; ++i) {
        double cosTheta = 2.0 * phirst::uniformRandom(rng) - 1.0;
        double sinTheta = phirst::math::sqrt(phirst::math::fmax(0.0, 1.0 - cosTheta * cosTheta));
        double phi = phirst::math::twoPi * phirst::uniformRandom(rng);
        double e = -phirst::math::log(phirst::math::fmax(
            phirst::uniformRandom(rng) * phirst::uniformRandom(rng), kMinLogArg));

        q[4 * i] = e;
        q[4 * i + 1] = e * sinTheta * phirst::math::sin(phi);
        q[4 * i + 2] = e * sinTheta * phirst::math::cos(phi);
        q[4 * i + 3] = e * cosTheta;

        totalMom[0] += q[4 * i];
        totalMom[1] += q[4 * i + 1];
        totalMom[2] += q[4 * i + 2];
        totalMom[3] += q[4 * i + 3];
    }

    double invariantMassSq = totalMom[0] * totalMom[0]
        - totalMom[1] * totalMom[1]
        - totalMom[2] * totalMom[2]
        - totalMom[3] * totalMom[3];
    double invariantMass = phirst::math::sqrt(phirst::math::fmax(0.0, invariantMassSq));
    if (invariantMass <= 0.0) {
        return 0.0;
    }

    double boostVec[3] = {
        -totalMom[1] / invariantMass,
        -totalMom[2] / invariantMass,
        -totalMom[3] / invariantMass,
    };
    double gamma = totalMom[0] / invariantMass;
    double boostFactor = 1.0 / (1.0 + gamma);
    double scaleFactor = cmEnergy / invariantMass;

    for (int i = 0; i < nParticles; ++i) {
        double q0 = q[4 * i];
        double qx = q[4 * i + 1];
        double qy = q[4 * i + 2];
        double qz = q[4 * i + 3];
        double bDotQ = boostVec[0] * qx + boostVec[1] * qy + boostVec[2] * qz;

        momentaFlat[4 * i] = scaleFactor * (gamma * q0 + bDotQ);
        momentaFlat[4 * i + 1] = scaleFactor * (qx + boostVec[0] * (q0 + boostFactor * bDotQ));
        momentaFlat[4 * i + 2] = scaleFactor * (qy + boostVec[1] * (q0 + boostFactor * bDotQ));
        momentaFlat[4 * i + 3] = scaleFactor * (qz + boostVec[2] * (q0 + boostFactor * bDotQ));
    }

    double logWeight = phirst::math::logPiOver2;
    if (nParticles != 2) {
        logWeight += (2.0 * static_cast<double>(nParticles) - 4.0) * phirst::math::log(cmEnergy)
            + z_coeff(nParticles);
    }
    return logWeight;
}

}  // namespace

extern "C" __global__ void phirst_numba_mc_kernel(
    double cmEnergy,
    const double* masses,
    int nParticles,
    int64_t nEvents,
    uint64_t baseSeed,
    double* sumOut,
    double* sum2Out) {
    (void)masses;

    int64_t workIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; workIdx < nEvents; workIdx += stride) {
        uint64_t rng = phirst::initRngState(baseSeed, workIdx);
        double momentaFlat[kMaxNumbaParticles * 4] = {};
        double logWeight = massless_rambo_generate(cmEnergy, nParticles, rng, momentaFlat);
        double integrandValue = phirst_user_integrand(momentaFlat, nParticles);
        double weightedValue = phirst::math::exp(logWeight) * integrandValue;

        atomicAdd(sumOut, weightedValue);
        atomicAdd(sum2Out, weightedValue * weightedValue);
    }
}

void phirst_numba_mc_launch(
    double cmEnergy,
    const double* d_masses,
    int nParticles,
    int64_t nEvents,
    uint64_t baseSeed,
    double* d_sum,
    double* d_sum2,
    int numBlocks,
    int blockSize) {
    phirst_numba_mc_kernel<<<numBlocks, blockSize>>>(
        cmEnergy,
        d_masses,
        nParticles,
        nEvents,
        baseSeed,
        d_sum,
        d_sum2);
}
