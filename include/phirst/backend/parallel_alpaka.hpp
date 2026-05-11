#pragma once
#ifndef PHIRST_PARALLEL_ALPAKA_HPP
#define PHIRST_PARALLEL_ALPAKA_HPP

#include "config.hpp"
#include <cstdint>
#include <vector>
#include <optional>
#include <alpaka/alpaka.hpp>

namespace phirst {
namespace alpaka_impl {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using AlpakaTag = alpaka::TagGpuCudaRt;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using AlpakaTag = alpaka::TagGpuHipRt;
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    using AlpakaTag = alpaka::TagGpuSyclIntel;
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    using AlpakaTag = alpaka::TagCpuOmp2Blocks;
#else
    using AlpakaTag = alpaka::TagCpuSerial;
#endif

using AlpakaDim = alpaka::DimInt<1>;
using AlpakaIdx = std::size_t;
using AlpakaAcc = alpaka::TagToAcc<AlpakaTag, AlpakaDim, AlpakaIdx>;
using AlpakaDevAcc = alpaka::Dev<AlpakaAcc>;
using AlpakaDevHost = alpaka::DevCpu;
using AlpakaQueue = alpaka::Queue<AlpakaAcc, alpaka::Blocking>;

// DeviceBuffer
template <typename T>
class DeviceBuffer {
public:
    using BufType = alpaka::Buf<AlpakaAcc, T, AlpakaDim, AlpakaIdx>;

    DeviceBuffer() : size_(0) {}
    explicit DeviceBuffer(int64_t n) : size_(n) {
        auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
        buf_.emplace(alpaka::allocBuf<T, AlpakaIdx>(devAcc, static_cast<AlpakaIdx>(n)));
    }

    T* data() { return buf_ ? alpaka::getPtrNative(*buf_) : nullptr; }
    const T* data() const { return buf_ ? alpaka::getPtrNative(*buf_) : nullptr; }
    int64_t size() const { return size_; }
    BufType& buffer() { return *buf_; }
    const BufType& buffer() const { return *buf_; }
private:
    std::optional<BufType> buf_;
    int64_t size_;
};

// deep_copy
template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
    auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    AlpakaIdx extent = static_cast<AlpakaIdx>(n);
    auto hostBuf = alpaka::allocBuf<T, AlpakaIdx>(devHost, extent);
    T* hostPtr = alpaka::getPtrNative(hostBuf);
    for (int64_t i = 0; i < n; ++i) hostPtr[i] = hostSrc[i];
    alpaka::memcpy(queue, dest.buffer(), hostBuf, extent);
    alpaka::wait(queue);
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
    auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    AlpakaIdx extent = static_cast<AlpakaIdx>(n);
    auto hostBuf = alpaka::allocBuf<T, AlpakaIdx>(devHost, extent);
    alpaka::memcpy(queue, hostBuf, const_cast<DeviceBuffer<T>&>(src).buffer(), extent);
    alpaka::wait(queue);
    T* hostPtr = alpaka::getPtrNative(hostBuf);
    for (int64_t i = 0; i < n; ++i) hostDest[i] = hostPtr[i];
}

// fill_buffer
template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    if (value == T{}) {
        alpaka::memset(queue, buf.buffer(), 0);
    } else {
        auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
        AlpakaIdx extent = static_cast<AlpakaIdx>(buf.size());
        auto hostBuf = alpaka::allocBuf<T, AlpakaIdx>(devHost, extent);
        T* hostPtr = alpaka::getPtrNative(hostBuf);
        for (int64_t i = 0; i < buf.size(); ++i) hostPtr[i] = value;
        alpaka::memcpy(queue, buf.buffer(), hostBuf, extent);
    }
    alpaka::wait(queue);
}

// fence
inline void fence() {
    // Alpaka uses blocking queue generally
}

// atomic_add
template <typename TAcc, typename T>
ALPAKA_FN_ACC void atomic_add(TAcc const& acc, T* ptr, T val) {
    alpaka::atomicAdd(acc, ptr, val, alpaka::hierarchy::Grids{});
}

// run_single_thread
template <typename WorkFunctor>
struct AlpakaSingleThreadKernel {
    WorkFunctor work;

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc) const {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];

        if (threadIdx == 0) {
            work(acc);
        }
    }
};

template <typename WorkFunctor>
void run_single_thread(const WorkFunctor& work) {
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);

    AlpakaSingleThreadKernel<WorkFunctor> kernel{work};

    alpaka::Vec<AlpakaDim, AlpakaIdx> const blocksPerGrid{1};
    alpaka::Vec<AlpakaDim, AlpakaIdx> const threadsPerBlock{1};
    alpaka::Vec<AlpakaDim, AlpakaIdx> const elementsPerThread{1};
    auto const workDiv = alpaka::WorkDivMembers<AlpakaDim, AlpakaIdx>(blocksPerGrid, threadsPerBlock, elementsPerThread);

    alpaka::exec<AlpakaAcc>(queue, workDiv, kernel);
    alpaka::wait(queue);
}

// grid_stride_reduce
template <typename WorkFunctor, typename T>
struct AlpakaGridStrideKernel {
    WorkFunctor work;
    T* globalResult1;
    T* globalResult2;
    int64_t nWork;
    int64_t totalThreads;

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc) const {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];

        T localAcc1 = T{};
        T localAcc2 = T{};
        for (int64_t i = static_cast<int64_t>(threadIdx); i < nWork; i += totalThreads) {
            work(acc, i, localAcc1, localAcc2);
        }

        alpaka::atomicAdd(acc, globalResult1, localAcc1, alpaka::hierarchy::Grids{});
        alpaka::atomicAdd(acc, globalResult2, localAcc2, alpaka::hierarchy::Grids{});
    }
};

template <typename WorkFunctor, typename T>
void grid_stride_reduce(int64_t nWork, const WorkFunctor& work, T& result1, T& result2) {
    auto cfg = GridConfig::compute(nWork);
    DeviceBuffer<T> deviceResult1(1);
    DeviceBuffer<T> deviceResult2(1);
    fill_buffer(deviceResult1, T{});
    fill_buffer(deviceResult2, T{});

    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);

    AlpakaGridStrideKernel<WorkFunctor, T> kernel{work, deviceResult1.data(), deviceResult2.data(), nWork, cfg.totalThreads};

    alpaka::Vec<AlpakaDim, AlpakaIdx> const blocksPerGrid{static_cast<AlpakaIdx>(cfg.numBlocks)};
    alpaka::Vec<AlpakaDim, AlpakaIdx> const threadsPerBlock{static_cast<AlpakaIdx>(cfg.blockSize)};
    alpaka::Vec<AlpakaDim, AlpakaIdx> const elementsPerThread{1};
    auto const workDiv = alpaka::WorkDivMembers<AlpakaDim, AlpakaIdx>(blocksPerGrid, threadsPerBlock, elementsPerThread);

    alpaka::exec<AlpakaAcc>(queue, workDiv, kernel);
    alpaka::wait(queue);

    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
}

} // namespace alpaka_impl
} // namespace phirst

#endif // PHIRST_PARALLEL_ALPAKA_HPP
