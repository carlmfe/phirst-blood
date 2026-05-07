#pragma once
#ifndef PHIRST_PARALLEL_CUDA_HPP
#define PHIRST_PARALLEL_CUDA_HPP

#include "config.hpp"
#include <cstdint>
#include <vector>

namespace phirst {
namespace cuda_impl {

// Dummy Accelerator
struct KernelAcc {};

// DeviceBuffer
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}
    explicit DeviceBuffer(int64_t n) : size_(n) {
        cudaMalloc(&data_, n * sizeof(T));
        cudaMemset(data_, 0, n * sizeof(T));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }

    DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), size_(o.size_) { o.data_ = nullptr; o.size_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { if (data_) cudaFree(data_); data_ = o.data_; size_ = o.size_; o.data_ = nullptr; o.size_ = 0; }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    int64_t size() const { return size_; }
private:
    T* data_;
    int64_t size_;
};

// deep_copy
template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
    cudaMemcpy(dest.data(), hostSrc, n * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
    cudaMemcpy(hostDest, src.data(), n * sizeof(T), cudaMemcpyDeviceToHost);
}

// fill_buffer
template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
    if (value == T{}) {
        cudaMemset(buf.data(), 0, buf.size() * sizeof(T));
    } else {
        std::vector<T> hostBuf(buf.size(), value);
        cudaMemcpy(buf.data(), hostBuf.data(), buf.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
}

// fence
inline void fence() {
    cudaDeviceSynchronize();
}

// atomic_add
template <typename Acc, typename T>
__device__ void atomic_add(const Acc&, T* ptr, T val) { atomicAdd(ptr, val); }

template <typename Acc>
__device__ inline void atomic_add(const Acc&, double* ptr, double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicAdd(ptr, val);
#else
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(ptr);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// run_single_thread
template <typename WorkFunctor>
__global__ void cuda_single_thread_kernel(WorkFunctor work) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        work(KernelAcc{});
    }
}

template <typename WorkFunctor>
void run_single_thread(const WorkFunctor& work) {
    cuda_single_thread_kernel<<<1, 1>>>(work);
    cudaDeviceSynchronize();
}

// grid_stride_reduce
template <typename WorkFunctor, typename T>
__global__ void cuda_grid_stride_kernel(WorkFunctor work, T* globalResult1, T* globalResult2, int64_t nWork, int64_t totalThreads) {
    int64_t threadIdx_global = blockIdx.x * blockDim.x + threadIdx.x;
    T localAcc1 = T{};
    T localAcc2 = T{};
    for (int64_t i = threadIdx_global; i < nWork; i += totalThreads) {
        work(KernelAcc{}, i, localAcc1, localAcc2);
    }
    atomic_add(KernelAcc{}, globalResult1, localAcc1);
    atomic_add(KernelAcc{}, globalResult2, localAcc2);
}

template <typename WorkFunctor, typename T>
void grid_stride_reduce(int64_t nWork, const WorkFunctor& work, T& result1, T& result2) {
    auto cfg = GridConfig::compute(nWork);
    DeviceBuffer<T> deviceResult1(1);
    DeviceBuffer<T> deviceResult2(1);
    fill_buffer(deviceResult1, T{});
    fill_buffer(deviceResult2, T{});

    cuda_grid_stride_kernel<<<cfg.numBlocks, cfg.blockSize>>>(work, deviceResult1.data(), deviceResult2.data(), nWork, cfg.totalThreads);
    cudaDeviceSynchronize();

    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
}

} // namespace cuda_impl
} // namespace phirst

#endif // PHIRST_PARALLEL_CUDA_HPP
