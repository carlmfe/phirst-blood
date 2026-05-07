#pragma once
#ifndef PHIRST_PARALLEL_SYCL_HPP
#define PHIRST_PARALLEL_SYCL_HPP

#include "config.hpp"
#include <cstdint>
#include <sycl/sycl.hpp>

namespace phirst {
namespace sycl_impl {

// Dummy Accelerator
struct KernelAcc {};

// SYCL Queue Helper
inline sycl::queue& get_default_queue() {
    static sycl::queue q{sycl::default_selector_v};
    return q;
}

// DeviceBuffer
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}
    explicit DeviceBuffer(int64_t n) : size_(n) {
        data_ = sycl::malloc_device<T>(n, get_default_queue());
    }
    ~DeviceBuffer() {
        if (data_) { sycl::free(data_, get_default_queue()); }
    }

    DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr; o.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (data_) { sycl::free(data_, get_default_queue()); }
            data_ = o.data_; size_ = o.size_;
            o.data_ = nullptr; o.size_ = 0;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* data() { return data_; }
    [[nodiscard]] const T* data() const { return data_; }
    [[nodiscard]] int64_t size() const { return size_; }
private:
    T* data_;
    int64_t size_;
};

// deep_copy
template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
    get_default_queue().memcpy(dest.data(), hostSrc, n * sizeof(T)).wait();
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
    get_default_queue().memcpy(hostDest, src.data(), n * sizeof(T)).wait();
}

// fill_buffer
template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
    if (value == T{}) {
        get_default_queue().memset(buf.data(), 0, buf.size() * sizeof(T)).wait();
    } else {
        get_default_queue().parallel_for(sycl::range<1>(buf.size()), [=, ptr = buf.data()](sycl::id<1> i) {
            ptr[i] = value;
        }).wait();
    }
}

// fence
inline void fence() {
    // SYCL uses .wait() on individual operations or queue.wait()
    get_default_queue().wait();
}

// atomic_add
template <typename Acc, typename T>
inline void atomic_add(const Acc&, T* ptr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*(ptr));
    ref.fetch_add(val);
}

// run_single_thread
template <typename WorkFunctor>
void run_single_thread(const WorkFunctor& work) {
    get_default_queue().submit([&](sycl::handler& h) {
        h.single_task([=]() {
            work(KernelAcc{});
        });
    }).wait();
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
    size_t globalSize = static_cast<size_t>(cfg.totalThreads);
    size_t localSize = static_cast<size_t>(cfg.blockSize);
    auto* ptr1 = deviceResult1.data();
    auto* ptr2 = deviceResult2.data();

    get_default_queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>{globalSize, localSize},
            [=](sycl::nd_item<1> item) {
                int64_t threadIdx = static_cast<int64_t>(item.get_global_id(0));
                T localAcc1 = T{};
                T localAcc2 = T{};
                for (int64_t idx = threadIdx; idx < nWork; idx += totalThreads) {
                    work(KernelAcc{}, idx, localAcc1, localAcc2);
                }
                atomic_add(KernelAcc{}, ptr1, localAcc1);
                atomic_add(KernelAcc{}, ptr2, localAcc2);
            }
        );
    });
    get_default_queue().wait();

    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
}

} // namespace sycl_impl
} // namespace phirst

#endif // PHIRST_PARALLEL_SYCL_HPP
