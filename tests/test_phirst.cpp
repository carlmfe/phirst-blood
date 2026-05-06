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

// =============================================================================
// MathFunctions: remaining math.hpp coverage
// =============================================================================

TEST(MathFunctions, LogExp) {
    EXPECT_NEAR(math::log(math::e), 1.0, 1e-15);
    EXPECT_NEAR(math::log(1.0), 0.0, 1e-15);
    EXPECT_NEAR(math::exp(0.0), 1.0, 1e-15);
    EXPECT_NEAR(math::exp(1.0), math::e, 1e-15);
    EXPECT_NEAR(math::log(math::exp(3.7)), 3.7, 1e-14);
}

TEST(MathFunctions, PowDouble) {
    EXPECT_NEAR(math::pow(2.0, 3.0), 8.0, 1e-14);
    EXPECT_NEAR(math::pow(4.0, 0.5), 2.0, 1e-14);
}

TEST(MathFunctions, PowInt) {
    EXPECT_NEAR(math::pow(2.0, 3), 8.0, 1e-14);
    EXPECT_NEAR(math::pow(5.0, 0), 1.0, 1e-14);
}

TEST(MathFunctions, Fabs) {
    EXPECT_DOUBLE_EQ(math::fabs(-5.7), 5.7);
    EXPECT_DOUBLE_EQ(math::fabs(5.7), 5.7);
    EXPECT_DOUBLE_EQ(math::fabs(0.0), 0.0);
}

TEST(MathFunctions, FmaxFmin) {
    EXPECT_DOUBLE_EQ(math::fmax(1.0, 2.0), 2.0);
    EXPECT_DOUBLE_EQ(math::fmax(-3.0, -1.0), -1.0);
    EXPECT_DOUBLE_EQ(math::fmin(1.0, 2.0), 1.0);
    EXPECT_DOUBLE_EQ(math::fmin(-3.0, -1.0), -3.0);
}

TEST(MathFunctions, Trigonometric) {
    EXPECT_NEAR(math::cos(0.0), 1.0, 1e-15);
    EXPECT_NEAR(math::cos(math::pi), -1.0, 1e-14);
    EXPECT_NEAR(math::tan(math::pi / 4.0), 1.0, 1e-14);
    EXPECT_NEAR(math::asin(1.0), math::halfPi, 1e-14);
    EXPECT_NEAR(math::acos(1.0), 0.0, 1e-15);
    EXPECT_NEAR(math::acos(0.0), math::halfPi, 1e-14);
    EXPECT_NEAR(math::atan(1.0), math::pi / 4.0, 1e-14);
    EXPECT_NEAR(math::atan2(1.0, 1.0), math::pi / 4.0, 1e-14);
    EXPECT_NEAR(math::atan2(1.0, 0.0), math::halfPi, 1e-14);
    // Pythagorean identity
    double angle = 1.23;
    EXPECT_NEAR(math::sin(angle) * math::sin(angle) + math::cos(angle) * math::cos(angle), 1.0, 1e-14);
}

TEST(MathFunctions, Hyperbolic) {
    EXPECT_NEAR(math::cosh(0.0), 1.0, 1e-15);
    EXPECT_NEAR(math::sinh(0.0), 0.0, 1e-15);
    // cosh^2 - sinh^2 = 1
    double x = 1.5;
    EXPECT_NEAR(math::cosh(x) * math::cosh(x) - math::sinh(x) * math::sinh(x), 1.0, 1e-13);
}

TEST(MathFunctions, CopySign) {
    EXPECT_DOUBLE_EQ(math::copysign(3.0, -1.0), -3.0);
    EXPECT_DOUBLE_EQ(math::copysign(-3.0, 1.0), 3.0);
    EXPECT_DOUBLE_EQ(math::copysign(3.0, 1.0), 3.0);
}

TEST(MathFunctions, Fmod) {
    EXPECT_NEAR(math::fmod(5.3, 2.0), 1.3, 1e-14);
    EXPECT_NEAR(math::fmod(-1.0, 3.0), -1.0, 1e-14);
    EXPECT_NEAR(math::fmod(6.0, 3.0), 0.0, 1e-14);
}

TEST(MathFunctions, Constants) {
    EXPECT_NEAR(math::twoPi, 2.0 * math::pi, 1e-15);
    EXPECT_NEAR(math::halfPi, 0.5 * math::pi, 1e-15);
    EXPECT_NEAR(math::logPiOver2, std::log(math::pi / 2.0), 1e-14);
    EXPECT_NEAR(math::e, std::exp(1.0), 1e-15);
    EXPECT_NEAR(math::ln2, std::log(2.0), 1e-15);
}

// =============================================================================
// Parallel infrastructure: DeviceBuffer, deep_copy, fill_buffer, atomic_add,
// host_reduce, GridConfig
//
// DeviceBuffer on GPU backends (CUDA, HIP, SYCL, Kokkos) is true device memory
// with no host-side operator[]. Tests that index DeviceBuffer directly or call
// __device__-only functions (atomic_add) are guarded to serial only.
// =============================================================================

#if defined(PHIRST_BACKEND_SERIAL)
TEST(Parallel, DeviceBufferAllocAndAccess) {
    phirst::DeviceBuffer<double> buf(5);
    EXPECT_EQ(buf.size(), 5);
    // Verify zero-initialization
    for (int64_t i = 0; i < 5; ++i) EXPECT_DOUBLE_EQ(buf[i], 0.0);
    // Write and read back
    for (int64_t i = 0; i < 5; ++i) buf[i] = static_cast<double>(i * 2);
    EXPECT_DOUBLE_EQ(buf[0], 0.0);
    EXPECT_DOUBLE_EQ(buf[4], 8.0);
}

TEST(Parallel, DeviceBufferConstAccess) {
    phirst::DeviceBuffer<int> buf(3);
    buf[0] = 7; buf[1] = 14; buf[2] = 21;
    const phirst::DeviceBuffer<int>& cref = buf;
    EXPECT_EQ(cref[0], 7);
    EXPECT_EQ(cref[2], 21);
}

TEST(Parallel, DeviceBufferMove) {
    phirst::DeviceBuffer<int> a(3);
    a[0] = 10; a[1] = 20; a[2] = 30;
    phirst::DeviceBuffer<int> b(std::move(a));
    EXPECT_EQ(b.size(), 3);
    EXPECT_EQ(b[0], 10);
    EXPECT_EQ(b[1], 20);
    EXPECT_EQ(a.size(), 0);  // moved-from state
}
#endif // PHIRST_BACKEND_SERIAL

TEST(Parallel, DeepCopyBothDirections) {
    const int64_t n = 4;
    double host[n] = {1.1, 2.2, 3.3, 4.4};
    phirst::DeviceBuffer<double> dev(n);
    phirst::deep_copy(dev, host, n);

    double out[n] = {};
    phirst::deep_copy(out, dev, n);
    for (int64_t i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(out[i], host[i]);
}

#if defined(PHIRST_BACKEND_SERIAL)
TEST(Parallel, FillBuffer) {
    phirst::DeviceBuffer<double> buf(6);
    phirst::fill_buffer(buf, 3.14);
    for (int64_t i = 0; i < buf.size(); ++i) EXPECT_DOUBLE_EQ(buf[i], 3.14);
}

TEST(Parallel, AtomicAdd) {
    double val = 10.0;
    phirst::atomic_add(phirst::KernelAcc{}, &val, 5.0);
    EXPECT_DOUBLE_EQ(val, 15.0);

    int ival = 3;
    phirst::atomic_add(phirst::KernelAcc{}, &ival, 7);
    EXPECT_EQ(ival, 10);
}
#endif // PHIRST_BACKEND_SERIAL

TEST(Parallel, HostReduce) {
    double data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double sum = phirst::host_reduce(data, 5);
    EXPECT_DOUBLE_EQ(sum, 15.0);

    double empty[1] = {};
    EXPECT_DOUBLE_EQ(phirst::host_reduce(empty, 0), 0.0);
}

TEST(Parallel, GridConfigCompute) {
    auto cfg = phirst::GridConfig::compute(1000, 256, 1024);
    EXPECT_EQ(cfg.blockSize, 256);
    EXPECT_EQ(cfg.numBlocks, 4);           // ceil(1000/256) = 4
    EXPECT_EQ(cfg.totalThreads, 4 * 256);

    // Clamping: nWork large enough to exceed maxBlks
    auto cfg2 = phirst::GridConfig::compute(1000000, 256, 4);
    EXPECT_EQ(cfg2.numBlocks, 4);          // clamped to maxBlks
    EXPECT_EQ(cfg2.totalThreads, 4 * 256);

    // Exact fit
    auto cfg3 = phirst::GridConfig::compute(512, 256, 1024);
    EXPECT_EQ(cfg3.numBlocks, 2);
    EXPECT_EQ(cfg3.totalThreads, 512);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
