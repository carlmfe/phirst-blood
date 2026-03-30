#pragma once
#ifndef PHIRST_PARALLEL_KOKKOS_HPP
#define PHIRST_PARALLEL_KOKKOS_HPP

#include "config.hpp"
#include <cstdint>
#include <vector>

namespace phirst {
namespace kokkos_impl {

// Execution Space Tag
struct HostSpace {};
struct DeviceSpace {};
using SpaceTag = DeviceSpace;

// Dummy Accelerator
struct KernelAcc {};

// Memory Space Aliases
using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemorySpace = Kokkos::HostSpace;

// DeviceBuffer
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : size_(0) {}
    explicit DeviceBuffer(int64_t n) : view_("buffer", n), size_(n) {}

    T* data() { return view_.data(); }
    const T* data() const { return view_.data(); }
    int64_t size() const { return size_; }
    Kokkos::View<T*, DefaultMemorySpace>& view() { return view_; }
    const Kokkos::View<T*, DefaultMemorySpace>& view() const { return view_; }
private:
    Kokkos::View<T*, DefaultMemorySpace> view_;
    int64_t size_;
};

// deep_copy
template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
    auto hostView = Kokkos::View<const T*, HostMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hostSrc, n);
    auto destSubview = Kokkos::subview(dest.view(), std::make_pair(int64_t(0), n));
    Kokkos::deep_copy(destSubview, hostView);
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
    auto hostView = Kokkos::View<T*, HostMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hostDest, n);
    auto srcSubview = Kokkos::subview(src.view(), std::make_pair(int64_t(0), n));
    Kokkos::deep_copy(hostView, srcSubview);
}

// fill_buffer
template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
    Kokkos::deep_copy(buf.view(), value);
}

// fence
inline void fence() {
    Kokkos::fence();
}

// atomic_add
template <typename Acc, typename T>
KOKKOS_INLINE_FUNCTION void atomic_add(const Acc&, T* ptr, T val) { Kokkos::atomic_add(ptr, val); }

// run_single_thread
template <typename WorkFunctor>
void run_single_thread(const WorkFunctor& work) {
    Kokkos::parallel_for("single_thread_exec", 1,
        KOKKOS_LAMBDA(const int) {
            work(KernelAcc{});
        }
    );
    Kokkos::fence();
}

// grid_stride_reduce
template <typename WorkFunctor, typename T>
void grid_stride_reduce(int64_t nWork, const WorkFunctor& work, T& result1, T& result2) {
    auto cfg = GridConfig::compute(nWork);
    DeviceBuffer<T> deviceResult1(1);
    DeviceBuffer<T> deviceResult2(1);
    fill_buffer(deviceResult1, T{});
    fill_buffer(deviceResult2, T{});

    const int64_t totalThreads = cfg.totalThreads;
    auto view1 = deviceResult1.view();
    auto view2 = deviceResult2.view();

    Kokkos::parallel_for("grid_stride_reduce", totalThreads,
        KOKKOS_LAMBDA(const int64_t threadIdx) {
            T localAcc1 = T{};
            T localAcc2 = T{};
            for (int64_t idx = threadIdx; idx < nWork; idx += totalThreads) {
                work(KernelAcc{}, idx, localAcc1, localAcc2);
            }
            atomic_add(KernelAcc{}, view1.data(), localAcc1);
            atomic_add(KernelAcc{}, view2.data(), localAcc2);
        }
    );
    Kokkos::fence();

    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
}

} // namespace kokkos_impl
} // namespace phirst

#endif // PHIRST_PARALLEL_KOKKOS_HPP
