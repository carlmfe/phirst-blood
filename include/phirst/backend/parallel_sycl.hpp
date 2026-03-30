#pragma once
#ifndef PHIRST_PARALLEL_SYCL_HPP
#define PHIRST_PARALLEL_SYCL_HPP

#include "config.hpp"
#include <cstdint>
#include <vector>
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
    DeviceBuffer() : data_(nullptr), size_(0), queue_(nullptr) {}
    explicit DeviceBuffer(int64_t n) : size_(n) {
        queue_ = new sycl::queue{sycl::default_selector_v};
        data_ = sycl::malloc_device<T>(n, *queue_);
    }
    ~DeviceBuffer() { if (data_ && queue_) { sycl::free(data_, *queue_); delete queue_; } }

    DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), size_(o.size_), queue_(o.queue_) {
        o.data_ = nullptr; o.size_ = 0; o.queue_ = nullptr;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (data_ && queue_) { sycl::free(data_, *queue_); delete queue_; }
            data_ = o.data_; size_ = o.size_; queue_ = o.queue_;
            o.data_ = nullptr; o.size_ = 0; o.queue_ = nullptr;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    int64_t size() const { return size_; }
    sycl::queue& queue() { return *queue_; }
private:
    T* data_;
    int64_t size_;
    sycl::queue* queue_;
};

// deep_copy
template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
    dest.queue().memcpy(dest.data(), hostSrc, n * sizeof(T)).wait();
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
    auto& queue = const_cast<DeviceBuffer<T>&>(src).queue();
    queue.memcpy(hostDest, src.data(), n * sizeof(T)).wait();
}

// fill_buffer
template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
    if (value == T{}) {
        buf.queue().memset(buf.data(), 0, buf.size() * sizeof(T)).wait();
    } else {
        buf.queue().parallel_for(sycl::range<1>(buf.size()), [=, ptr = buf.data()](sycl::id<1> i) {
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

    deviceResult1.queue().submit([&](sycl::handler& h) {
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
    deviceResult1.queue().wait();

    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
}

} // namespace sycl_impl
} // namespace phirst

#endif // PHIRST_PARALLEL_SYCL_HPP
