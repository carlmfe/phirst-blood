#include "phirst/backend/numba_bridge.cuh"

#include "phirst/phase_space.hpp"

namespace {

// One PhaseSpaceGenerator<N> is constructed per thread (outside the event loop)
// so mass pre-computation is done once rather than once per event.
template <int N>
__device__ void runEvents(
    double cmEnergy,
    const double* masses,
    int64_t workStart,
    int64_t stride,
    int64_t nEvents,
    uint64_t baseSeed,
    double* sumOut,
    double* sum2Out)
{
    phirst::PhaseSpaceGenerator<N> gen(masses);
    for (int64_t idx = workStart; idx < nEvents; idx += stride) {
        uint64_t rng = phirst::initRngState(baseSeed, idx);
        double momenta[N][4];
        double logWeight = gen(cmEnergy, rng, momenta);
        if (!isfinite(logWeight)) { continue; }
        double integrandValue = 0.0;
        int rc = phirst_user_integrand(&integrandValue, &momenta[0][0], N);
        (void)rc;
        double weightedValue = phirst::math::exp(logWeight) * integrandValue;
        atomicAdd(sumOut, weightedValue);
        atomicAdd(sum2Out, weightedValue * weightedValue);
    }
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
    int64_t workIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    switch (nParticles) {
        case 2:  runEvents<2>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 3:  runEvents<3>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 4:  runEvents<4>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 5:  runEvents<5>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 6:  runEvents<6>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 7:  runEvents<7>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 8:  runEvents<8>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 9:  runEvents<9>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        case 10: runEvents<10>(cmEnergy, masses, workIdx, stride, nEvents, baseSeed, sumOut, sum2Out); break;
        default: break;
    }
}
