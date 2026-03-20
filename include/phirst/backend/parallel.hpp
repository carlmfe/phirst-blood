#pragma once
#ifndef PHIRST_PARALLEL_HPP
#define PHIRST_PARALLEL_HPP

/**
 * @file parallel.hpp
 * @brief Portable parallel execution abstractions.
 * 
 * Provides low-level, backend-agnostic primitives for parallel execution.
 * All backends provide the same set of functions with consistent signatures.
 * 
 * Key abstractions:
 * - DeviceBuffer<T>: Manages device memory allocation
 * - deep_copy(): Memory transfer between host and device
 * - fill_buffer(): Initialize buffer with value
 * - fence(): Synchronization barrier
 * - GridConfig: Thread grid configuration helper
 * - seed_for_thread(): RNG seeding helper
 * - grid_stride_reduce(): Generic parallel reduction primitive
 */

#include "config.hpp"
#include "math.hpp"
#include <cstdint>
#include <vector>

// Backend-specific includes
#if defined(PHIRST_BACKEND_SYCL)
    #include <sycl/sycl.hpp>
#elif defined(PHIRST_BACKEND_ALPAKA)
    #include <alpaka/alpaka.hpp>
    #include <optional>
#elif defined(PHIRST_BACKEND_KOKKOS)
    // Kokkos included via backend.hpp
#elif defined(PHIRST_BACKEND_CUDA)
    // CUDA included via compilation
#endif

namespace phirst {

// =============================================================================
// Execution Space Tags
// =============================================================================

struct HostSpace {};
struct DeviceSpace {};

#if defined(PHIRST_BACKEND_SERIAL)
    using DefaultSpace = HostSpace;
#else
    using DefaultSpace = DeviceSpace;
#endif

// =============================================================================
// Backend-Specific Type Aliases
// =============================================================================

#if defined(PHIRST_BACKEND_KOKKOS)
    using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using HostMemorySpace = Kokkos::HostSpace;
#elif defined(PHIRST_BACKEND_ALPAKA)
    #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        using AlpakaTag = alpaka::TagGpuCudaRt;
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
#endif

// =============================================================================
// Grid Configuration
// =============================================================================

struct GridConfig {
    int64_t totalThreads;
    int64_t blockSize;
    int64_t numBlocks;
    
    static GridConfig compute(int64_t nWork, int64_t blockSz = 256, int64_t maxBlks = 1024) {
        GridConfig cfg;
        cfg.blockSize = blockSz;
        cfg.numBlocks = (nWork + blockSz - 1) / blockSz;
        if (cfg.numBlocks > maxBlks) cfg.numBlocks = maxBlks;
        cfg.totalThreads = cfg.numBlocks * cfg.blockSize;
        return cfg;
    }
};

// =============================================================================
// RNG Seeding Helper
// =============================================================================

PHIRST_HOST_DEVICE
inline uint64_t seed_for_thread(uint64_t baseSeed, int64_t threadIdx) {
    uint64_t seed = baseSeed ^ (static_cast<uint64_t>(threadIdx) * 2685821657736338717ULL);
    return (seed == 0) ? baseSeed + 1 : seed;
}

// =============================================================================
// DeviceBuffer<T> - Memory Management
// =============================================================================
// Note: DeviceBuffer requires separate class definitions per backend due to
// fundamentally different underlying storage types (pointers, Views, Buffers).

#if defined(PHIRST_BACKEND_SERIAL)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}
    explicit DeviceBuffer(int64_t n) : size_(n) { data_ = new T[n](); }
    ~DeviceBuffer() { delete[] data_; }
    
    DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), size_(o.size_) { o.data_ = nullptr; o.size_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { delete[] data_; data_ = o.data_; size_ = o.size_; o.data_ = nullptr; o.size_ = 0; }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    int64_t size() const { return size_; }
    T& operator[](int64_t i) { return data_[i]; }
    const T& operator[](int64_t i) const { return data_[i]; }
private:
    T* data_;
    int64_t size_;
};

#elif defined(PHIRST_BACKEND_KOKKOS)

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

#elif defined(PHIRST_BACKEND_SYCL)

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

#elif defined(PHIRST_BACKEND_ALPAKA)

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

#elif defined(PHIRST_BACKEND_CUDA)

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

#endif // DeviceBuffer

// =============================================================================
// deep_copy - Host/Device Memory Transfer
// =============================================================================

template <typename T>
void deep_copy(DeviceBuffer<T>& dest, const T* hostSrc, int64_t n) {
#if defined(PHIRST_BACKEND_SERIAL)
    for (int64_t i = 0; i < n; ++i) dest[i] = hostSrc[i];
    
#elif defined(PHIRST_BACKEND_KOKKOS)
    auto hostView = Kokkos::View<const T*, Kokkos::HostSpace, 
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hostSrc, n);
    auto destSubview = Kokkos::subview(dest.view(), std::make_pair(int64_t(0), n));
    Kokkos::deep_copy(destSubview, hostView);
    
#elif defined(PHIRST_BACKEND_SYCL)
    dest.queue().memcpy(dest.data(), hostSrc, n * sizeof(T)).wait();
    
#elif defined(PHIRST_BACKEND_ALPAKA)
    auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    AlpakaIdx extent = static_cast<AlpakaIdx>(n);
    auto hostBuf = alpaka::allocBuf<T, AlpakaIdx>(devHost, extent);
    for (int64_t i = 0; i < n; ++i) hostBuf[i] = hostSrc[i];
    alpaka::memcpy(queue, dest.buffer(), hostBuf, extent);
    alpaka::wait(queue);
    
#elif defined(PHIRST_BACKEND_CUDA)
    cudaMemcpy(dest.data(), hostSrc, n * sizeof(T), cudaMemcpyHostToDevice);
#endif
}

template <typename T>
void deep_copy(T* hostDest, const DeviceBuffer<T>& src, int64_t n) {
#if defined(PHIRST_BACKEND_SERIAL)
    for (int64_t i = 0; i < n; ++i) hostDest[i] = src[i];
    
#elif defined(PHIRST_BACKEND_KOKKOS)
    auto hostView = Kokkos::View<T*, Kokkos::HostSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hostDest, n);
    auto srcSubview = Kokkos::subview(src.view(), std::make_pair(int64_t(0), n));
    Kokkos::deep_copy(hostView, srcSubview);
    
#elif defined(PHIRSTT_BACKEND_SYCL)
    auto& queue = const_cast<DeviceBuffer<T>&>(src).queue();
    queue.memcpy(hostDest, src.data(), n * sizeof(T)).wait();
    
#elif defined(PHIRSTT_BACKEND_ALPAKA)
    auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    AlpakaIdx extent = static_cast<AlpakaIdx>(n);
    auto hostBuf = alpaka::allocBuf<T, AlpakaIdx>(devHost, extent);
    alpaka::memcpy(queue, hostBuf, const_cast<DeviceBuffer<T>&>(src).buffer(), extent);
    alpaka::wait(queue);
    for (int64_t i = 0; i < n; ++i) hostDest[i] = hostBuf[i];
    
#elif defined(PHIRSTT_BACKEND_CUDA)
    cudaMemcpy(hostDest, src.data(), n * sizeof(T), cudaMemcpyDeviceToHost);
#endif
}

// =============================================================================
// fill_buffer - Initialize Buffer
// =============================================================================

template <typename T>
void fill_buffer(DeviceBuffer<T>& buf, T value) {
#if defined(PHIRSTT_BACKEND_SERIAL)
    for (int64_t i = 0; i < buf.size(); ++i) buf[i] = value;
    
#elif defined(PHIRSTTTTT_BACKEND_KOKKOS)
    Kokkos::deep_copy(buf.view(), value);
    
#elif defined(PHIRSTTTT_BACKEND_SYCL)
    if (value == T{}) {
        buf.queue().memset(buf.data(), 0, buf.size() * sizeof(T)).wait();
    } else {
        buf.queue().parallel_for(sycl::range<1>(buf.size()), [=, ptr = buf.data()](sycl::id<1> i) {
            ptr[i] = value;
        }).wait();
    }
    
#elif defined(PHIRST_BACKEND_ALPAKA)
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    if (value == T{}) {
        alpaka::memset(queue, buf.buffer(), 0);
    } else {
        auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
        AlpakaIdx extent = static_cast<AlpakaIdx>(buf.size());
        auto hostBuf = alpaka::allocBuf<T, AlpakaIdx>(devHost, extent);
        for (int64_t i = 0; i < buf.size(); ++i) hostBuf[i] = value;
        alpaka::memcpy(queue, buf.buffer(), hostBuf, extent);
    }
    alpaka::wait(queue);
    
#elif defined(PHIRST_BACKEND_CUDA)
    if (value == T{}) {
        cudaMemset(buf.data(), 0, buf.size() * sizeof(T));
    } else {
        std::vector<T> hostBuf(buf.size(), value);
        cudaMemcpy(buf.data(), hostBuf.data(), buf.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
#endif
}

// =============================================================================
// fence - Synchronization Barrier
// =============================================================================

inline void fence() {
#if defined(PHIRST_BACKEND_SERIAL)
    // No-op
#elif defined(PHIRST_BACKEND_KOKKOS)
    Kokkos::fence();
#elif defined(PHIRST_BACKEND_SYCL)
    // SYCL uses .wait() on individual operations
#elif defined(PHIRST_BACKEND_ALPAKA)
    // Alpaka uses blocking queue
#elif defined(PHIRST_BACKEND_CUDA)
    cudaDeviceSynchronize();
#endif
}

// =============================================================================
// host_reduce - Host-side Reduction Helper
// =============================================================================

template <typename T>
T host_reduce(const T* data, int64_t n) {
    T sum = T{};
    for (int64_t i = 0; i < n; ++i) sum += data[i];
    return sum;
}

// =============================================================================
// atomic_add - Portable Atomic Addition
// =============================================================================
// Device-callable atomic add wrapper. 
// Note: Alpaka requires the accelerator object, so uses alpaka::atomicAdd directly.
// Note: SYCL requires atomic_ref, which is verbose - use PHIRST_SYCL_ATOMIC_ADD macro.

#if defined(PHIRST_BACKEND_SERIAL)
template <typename T>
inline void atomic_add(T* ptr, T val) { *ptr += val; }

#elif defined(PHIRST_BACKEND_KOKKOS)
template <typename T>
KOKKOS_INLINE_FUNCTION void atomic_add(T* ptr, T val) { Kokkos::atomic_add(ptr, val); }

#elif defined(PHIRST_BACKEND_CUDA)
template <typename T>
__device__ void atomic_add(T* ptr, T val) { atomicAdd(ptr, val); }

#elif defined(PHIRST_BACKEND_SYCL)
// SYCL atomic_ref is verbose; provide macro for use in device code
#define PHIRST_SYCL_ATOMIC_ADD(ptr, val) do { \
    sycl::atomic_ref<decltype(val), sycl::memory_order::relaxed, \
        sycl::memory_scope::device, \
        sycl::access::address_space::global_space> ref(*(ptr)); \
    ref.fetch_add(val); \
} while(0)

#elif defined(PHIRST_BACKEND_ALPAKA)
// Alpaka needs acc object; use alpaka::atomicAdd(acc, ptr, val, hierarchy::Grids{}) directly
#endif

// =============================================================================
// Grid-Stride Kernel Bodies
// =============================================================================
// Internal grid-stride loop + atomic reduction logic.
// These are defined at namespace scope for backends requiring kernel structs.

#if defined(PHIRST_BACKEND_ALPAKA)

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
            work(i, localAcc1, localAcc2);
        }
        
        alpaka::atomicAdd(acc, globalResult1, localAcc1, alpaka::hierarchy::Grids{});
        alpaka::atomicAdd(acc, globalResult2, localAcc2, alpaka::hierarchy::Grids{});
    }
};

#elif defined(PHIRST_BACKEND_CUDA)

template <typename WorkFunctor, typename T>
__global__ void cuda_grid_stride_kernel(WorkFunctor work, T* globalResult1, T* globalResult2, 
                                         int64_t nWork, int64_t totalThreads) {
    int64_t threadIdx_global = blockIdx.x * blockDim.x + threadIdx.x;
    
    T localAcc1 = T{};
    T localAcc2 = T{};
    for (int64_t i = threadIdx_global; i < nWork; i += totalThreads) {
        work(i, localAcc1, localAcc2);
    }
    
    atomic_add(globalResult1, localAcc1);
    atomic_add(globalResult2, localAcc2);
}

#endif

// =============================================================================
// launch_reduction_kernel - Backend-Specific Kernel Launch
// =============================================================================
// Launches the grid-stride reduction kernel. This is the ONLY function that
// differs fundamentally between backends (kernel launch syntax).

template <typename WorkFunctor, typename T>
void launch_reduction_kernel(const GridConfig& cfg, int64_t nWork, const WorkFunctor& work,
                              T* ptr1, T* ptr2
#if defined(PHIRST_BACKEND_SYCL)
                              , sycl::queue& queue
#endif
                              ) {
#if defined(PHIRST_BACKEND_SERIAL)
    // Sequential - no parallelism, direct accumulation
    for (int64_t i = 0; i < nWork; ++i) {
        work(i, *ptr1, *ptr2);
    }
    
#elif defined(PHIRST_BACKEND_KOKKOS)
    const int64_t totalThreads = cfg.totalThreads;
    Kokkos::parallel_for("grid_stride_reduce", cfg.totalThreads,
        KOKKOS_LAMBDA(const int64_t threadIdx) {
            T localAcc1 = T{};
            T localAcc2 = T{};
            for (int64_t idx = threadIdx; idx < nWork; idx += totalThreads) {
                work(idx, localAcc1, localAcc2);
            }
            atomic_add(ptr1, localAcc1);
            atomic_add(ptr2, localAcc2);
        }
    );
    Kokkos::fence();
    
#elif defined(PHIRST_BACKEND_SYCL)
    const int64_t totalThreads = cfg.totalThreads;
    size_t globalSize = static_cast<size_t>(cfg.totalThreads);
    size_t localSize = static_cast<size_t>(cfg.blockSize);
    
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>{globalSize, localSize},
            [=](sycl::nd_item<1> item) {
                int64_t threadIdx = static_cast<int64_t>(item.get_global_id(0));
                T localAcc1 = T{};
                T localAcc2 = T{};
                for (int64_t idx = threadIdx; idx < nWork; idx += totalThreads) {
                    work(idx, localAcc1, localAcc2);
                }
                PHIRST_SYCL_ATOMIC_ADD(ptr1, localAcc1);
                PHIRST_SYCL_ATOMIC_ADD(ptr2, localAcc2);
            }
        );
    });
    queue.wait();
    
#elif defined(PHIRST_BACKEND_ALPAKA)
    auto devAcc = alpaka::getDevByIdx(alpaka::Platform<AlpakaAcc>{}, 0);
    AlpakaQueue queue(devAcc);
    
    AlpakaGridStrideKernel<WorkFunctor, T> kernel{work, ptr1, ptr2, nWork, cfg.totalThreads};
    
    alpaka::Vec<AlpakaDim, AlpakaIdx> const blocksPerGrid{static_cast<AlpakaIdx>(cfg.numBlocks)};
    alpaka::Vec<AlpakaDim, AlpakaIdx> const threadsPerBlock{static_cast<AlpakaIdx>(cfg.blockSize)};
    alpaka::Vec<AlpakaDim, AlpakaIdx> const elementsPerThread{1};
    auto const workDiv = alpaka::WorkDivMembers<AlpakaDim, AlpakaIdx>(blocksPerGrid, threadsPerBlock, elementsPerThread);
    
    alpaka::exec<AlpakaAcc>(queue, workDiv, kernel);
    alpaka::wait(queue);
    
#elif defined(PHIRST_BACKEND_CUDA)
    cuda_grid_stride_kernel<<<cfg.numBlocks, cfg.blockSize>>>(
        work, ptr1, ptr2, nWork, cfg.totalThreads
    );
    cudaDeviceSynchronize();
#endif
}

// =============================================================================
// grid_stride_reduce - Generic Parallel Reduction Primitive
// =============================================================================
// 
// Encapsulates the common GPU reduction pattern:
// - Allocate device result buffers (shared)
// - Initialize to zero (shared)
// - Launch grid-stride kernel with atomics (backend-specific)
// - Copy results back (shared)
//
// WorkFunctor signature: void operator()(int64_t workIdx, T& acc1, T& acc2) const
// =============================================================================

template <typename WorkFunctor, typename T>
void grid_stride_reduce(int64_t nWork, const WorkFunctor& work, T& result1, T& result2) {
    auto cfg = GridConfig::compute(nWork);
    
#if defined(PHIRST_BACKEND_SERIAL)
    // Serial: no buffers needed, accumulate directly
    result1 = T{};
    result2 = T{};
    launch_reduction_kernel(cfg, nWork, work, &result1, &result2);
    
#elif defined(PHIRST_BACKEND_SYCL)
    // SYCL: needs queue passed to kernel launcher
    DeviceBuffer<T> deviceResult1(1);
    DeviceBuffer<T> deviceResult2(1);
    fill_buffer(deviceResult1, T{});
    fill_buffer(deviceResult2, T{});
    
    launch_reduction_kernel(cfg, nWork, work, deviceResult1.data(), deviceResult2.data(),
                            deviceResult1.queue());
    
    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
    
#else
    // Kokkos, Alpaka, CUDA: standard pattern
    DeviceBuffer<T> deviceResult1(1);
    DeviceBuffer<T> deviceResult2(1);
    fill_buffer(deviceResult1, T{});
    fill_buffer(deviceResult2, T{});
    
    launch_reduction_kernel(cfg, nWork, work, deviceResult1.data(), deviceResult2.data());
    
    deep_copy(&result1, deviceResult1, 1);
    deep_copy(&result2, deviceResult2, 1);
#endif
}

} // namespace phirst

#endif // PHIRST_PARALLEL_HPP
