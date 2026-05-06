#pragma once
#ifndef PHIRST_PARALLEL_SERIAL_HPP
#define PHIRST_PARALLEL_SERIAL_HPP

#include "config.hpp"
#include <cstdint>
#include <vector>

// Forward declare GridConfig, KernelAcc if needed, though they might be better placed
// in a common types file if used universally. For now, we define what Serial needs.

namespace phirst::serial_impl {

// Execution Space Tag
struct HostSpace {};
using SpaceTag = HostSpace;

// Dummy Accelerator
struct KernelAcc {};

// DeviceBuffer
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}
    explicit DeviceBuffer(int64_t n) : data_(new T[n]()), size_(n) {}
    ~DeviceBuffer() { delete[] data_; }

    DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), size_(o.size_) { o.data_ = nullptr; o.size_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { delete[] data_; data_ = o.data_; size_ = o.size_; o.data_ = nullptr; o.size_ = 0; }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* data() { return data_; }
    [[nodiscard]] const T* data() const { return data_; }
    [[nodiscard]] int64_t size() const { return size_; }
    T& operator[](int64_t i) { return data_[i]; }
    const T& operator[](int64_t i) const { return data_[i]; }
private:
    T* data_;
    int64_t size_;
};

// deep_copy
template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
    for (int64_t i = 0; i < n; ++i) { dest[i] = hostSrc[i]; }
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) { hostDest[i] = src[i]; }
}

// fill_buffer
template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
    for (int64_t i = 0; i < buf.size(); ++i) { buf[i] = value; }
}

// fence
inline void fence() {
    // No-op for serial
}

// atomic_add
template <typename Acc, typename T>
inline void atomic_add(const Acc& /*acc*/, T* ptr, T val) {
    *ptr += val;
}

// run_single_thread
template <typename WorkFunctor>
void run_single_thread(const WorkFunctor& work) {
    work(KernelAcc{});
}

// launch_reduction_kernel & grid_stride_reduce combined logically
template <typename WorkFunctor, typename T>
void grid_stride_reduce(int64_t nWork, const WorkFunctor& work, T& result1, T& result2) {
    result1 = T{};
    result2 = T{};
    for (int64_t i = 0; i < nWork; ++i) {
        work(KernelAcc{}, i, result1, result2);
    }
}

} // namespace phirst::serial_impl

#endif // PHIRST_PARALLEL_SERIAL_HPP
