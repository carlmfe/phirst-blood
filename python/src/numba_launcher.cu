#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef PHIRST_NUMBA_BRIDGE_OBJECT_FILENAME
#define PHIRST_NUMBA_BRIDGE_OBJECT_FILENAME "phirst_numba_bridge_device.o"
#endif

namespace {

struct CubinCache {
    std::mutex mutex;
    std::unordered_map<size_t, std::vector<char>> cubins;
};

auto cubin_cache() -> CubinCache& {
    static CubinCache cache;
    return cache;
}

auto ptx_key(const char* ptxBytes, size_t ptxSize) -> size_t {
    return std::hash<std::string_view>{}(std::string_view(ptxBytes, ptxSize));
}

void report_driver_error(const char* what, CUresult status) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    std::fprintf(stderr,
                 "phirst numba bridge: %s failed (%s): %s\n",
                 what,
                 name != nullptr ? name : "unknown",
                 message != nullptr ? message : "no error string");
}

void report_linker_log(const char* action, const char* errorLog, const char* infoLog) {
    if (errorLog != nullptr && errorLog[0] != '\0') {
        std::fprintf(stderr, "phirst numba bridge: %s error log:\n%s\n", action, errorLog);
    }
    if (infoLog != nullptr && infoLog[0] != '\0') {
        std::fprintf(stderr, "phirst numba bridge: %s info log:\n%s\n", action, infoLog);
    }
}

#define PHIRST_CU_TRY(call, action) \
    do { \
        status = (call); \
        if (status != CUDA_SUCCESS) { \
            report_driver_error(action, status); \
            result = 1; \
            goto cleanup; \
        } \
    } while (false)

auto bridge_object_path() -> std::string {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&bridge_object_path), &info) == 0 || info.dli_fname == nullptr) {
        return PHIRST_NUMBA_BRIDGE_OBJECT_FILENAME;
    }
    std::filesystem::path libPath(info.dli_fname);
    return (libPath.parent_path() / PHIRST_NUMBA_BRIDGE_OBJECT_FILENAME).string();
}

auto masses_are_massless(const double* massesHost, int nParticles) -> bool {
    for (int i = 0; i < nParticles; ++i) {
        if (std::fabs(massesHost[i]) > 0.0) {
            return false;
        }
    }
    return true;
}

void compute_statistics(int64_t nEvents, double sum, double sum2, double* meanOut, double* errorOut) {
    if (meanOut == nullptr || errorOut == nullptr) {
        return;
    }
    if (nEvents <= 0) {
        *meanOut = 0.0;
        *errorOut = 0.0;
        return;
    }

    double n = static_cast<double>(nEvents);
    *meanOut = sum / n;
    double variance = (sum2 / n) - (*meanOut * *meanOut);
    *errorOut = std::sqrt(std::fabs(variance) / n);
}

}  // namespace

extern "C" int phirst_link_and_launch(
    const char* ptx_bytes,
    size_t ptx_size,
    double cmEnergy,
    const double* masses_host,
    int nParticles,
    int64_t nEvents,
    uint64_t seed,
    double* mean_out,
    double* error_out) {
    if (ptx_bytes == nullptr || ptx_size == 0 || masses_host == nullptr || mean_out == nullptr || error_out == nullptr) {
        return 1;
    }
    if (nParticles < 2 || nParticles > 10) {
        std::fprintf(stderr, "phirst numba bridge: nParticles must be in [2, 10]\n");
        return 1;
    }
    if (!masses_are_massless(masses_host, nParticles)) {
        std::fprintf(stderr, "phirst numba bridge: only massless particles are supported\n");
        return 1;
    }

    int result = 0;
    CUresult status = CUDA_SUCCESS;
    CUcontext context = nullptr;
    bool createdContext = false;
    CUlinkState linkState = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr dMasses = 0;
    CUdeviceptr dSum = 0;
    CUdeviceptr dSum2 = 0;
    std::vector<char> localCubin;
    std::vector<char>* cubinBytes = nullptr;
    double zero = 0.0;
    double sum = 0.0;
    double sum2 = 0.0;

    PHIRST_CU_TRY(cuInit(0), "cuInit");
    PHIRST_CU_TRY(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (context == nullptr) {
        CUdevice device = 0;
        PHIRST_CU_TRY(cuDeviceGet(&device, 0), "cuDeviceGet");
        PHIRST_CU_TRY(cuCtxCreate(&context, 0, device), "cuCtxCreate");
        createdContext = true;
    }

    {
        auto key = ptx_key(ptx_bytes, ptx_size);
        auto& cache = cubin_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto it = cache.cubins.find(key);
        if (it != cache.cubins.end()) {
            cubinBytes = &it->second;
        }
    }

    if (cubinBytes == nullptr) {
        char infoLog[8192] = {};
        char errorLog[8192] = {};
        CUjit_option options[] = {
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_TARGET_FROM_CUCONTEXT,
        };
        void* optionValues[] = {
            infoLog,
            reinterpret_cast<void*>(static_cast<intptr_t>(sizeof(infoLog))),
            errorLog,
            reinterpret_cast<void*>(static_cast<intptr_t>(sizeof(errorLog))),
            nullptr,
        };

        PHIRST_CU_TRY(cuLinkCreate(5, options, optionValues, &linkState), "cuLinkCreate");

        const std::string bridgeObject = bridge_object_path();
        PHIRST_CU_TRY(
            cuLinkAddFile(linkState, CU_JIT_INPUT_OBJECT, bridgeObject.c_str(), 0, nullptr, nullptr),
            "cuLinkAddFile");
        PHIRST_CU_TRY(
            cuLinkAddData(linkState,
                          CU_JIT_INPUT_PTX,
                          const_cast<char*>(ptx_bytes),
                          ptx_size,
                          const_cast<char*>("phirst_numba_user.ptx"),
                          0,
                          nullptr,
                          nullptr),
            "cuLinkAddData");

        void* linkedCubin = nullptr;
        size_t linkedCubinSize = 0;
        status = cuLinkComplete(linkState, &linkedCubin, &linkedCubinSize);
        if (status != CUDA_SUCCESS) {
            report_linker_log("cuLinkComplete", errorLog, infoLog);
            report_driver_error("cuLinkComplete", status);
            result = 1;
            goto cleanup;
        }
        report_linker_log("cuLinkComplete", errorLog, infoLog);

        localCubin.assign(static_cast<const char*>(linkedCubin),
                          static_cast<const char*>(linkedCubin) + linkedCubinSize);

        {
            auto key = ptx_key(ptx_bytes, ptx_size);
            auto& cache = cubin_cache();
            std::lock_guard<std::mutex> lock(cache.mutex);
            auto [it, inserted] = cache.cubins.emplace(key, localCubin);
            if (!inserted) {
                it->second = localCubin;
            }
            cubinBytes = &it->second;
        }
    }

    if (cubinBytes == nullptr) {
        result = 1;
        goto cleanup;
    }

    PHIRST_CU_TRY(cuModuleLoadData(&module, cubinBytes->data()), "cuModuleLoadData");
    PHIRST_CU_TRY(cuModuleGetFunction(&kernel, module, "phirst_numba_mc_kernel"), "cuModuleGetFunction");

    if (nEvents > 0) {
        PHIRST_CU_TRY(cuMemAlloc(&dMasses, static_cast<size_t>(nParticles) * sizeof(double)), "cuMemAlloc masses");
        PHIRST_CU_TRY(cuMemAlloc(&dSum, sizeof(double)), "cuMemAlloc sum");
        PHIRST_CU_TRY(cuMemAlloc(&dSum2, sizeof(double)), "cuMemAlloc sum2");

        PHIRST_CU_TRY(cuMemcpyHtoD(dMasses, masses_host, static_cast<size_t>(nParticles) * sizeof(double)), "cuMemcpyHtoD masses");
        PHIRST_CU_TRY(cuMemcpyHtoD(dSum, &zero, sizeof(double)), "cuMemcpyHtoD sum");
        PHIRST_CU_TRY(cuMemcpyHtoD(dSum2, &zero, sizeof(double)), "cuMemcpyHtoD sum2");

        int blockSize = 256;
        int64_t maxBlocks = 1024;
        int numBlocks = static_cast<int>((nEvents + blockSize - 1) / blockSize);
        if (numBlocks < 1) {
            numBlocks = 1;
        }
        if (numBlocks > maxBlocks) {
            numBlocks = static_cast<int>(maxBlocks);
        }

        void* args[] = {
            &cmEnergy,
            &dMasses,
            &nParticles,
            &nEvents,
            &seed,
            &dSum,
            &dSum2,
        };
        PHIRST_CU_TRY(cuLaunchKernel(kernel,
                                     static_cast<unsigned int>(numBlocks),
                                     1,
                                     1,
                                     static_cast<unsigned int>(blockSize),
                                     1,
                                     1,
                                     0,
                                     nullptr,
                                     args,
                                     nullptr),
                      "cuLaunchKernel");
        PHIRST_CU_TRY(cuCtxSynchronize(), "cuCtxSynchronize");
        PHIRST_CU_TRY(cuMemcpyDtoH(&sum, dSum, sizeof(double)), "cuMemcpyDtoH sum");
        PHIRST_CU_TRY(cuMemcpyDtoH(&sum2, dSum2, sizeof(double)), "cuMemcpyDtoH sum2");
    }

    compute_statistics(nEvents, sum, sum2, mean_out, error_out);

cleanup:
    if (dSum2 != 0) {
        cuMemFree(dSum2);
    }
    if (dSum != 0) {
        cuMemFree(dSum);
    }
    if (dMasses != 0) {
        cuMemFree(dMasses);
    }
    if (module != nullptr) {
        cuModuleUnload(module);
    }
    if (linkState != nullptr) {
        cuLinkDestroy(linkState);
    }
    if (createdContext && context != nullptr) {
        cuCtxDestroy(context);
    }
    return result;
}
