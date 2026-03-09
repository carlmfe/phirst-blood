#pragma once
#ifndef RAMBO_SYCL_INTEGRATOR_HPP
#define RAMBO_SYCL_INTEGRATOR_HPP

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

#include "phase_space.hpp"

namespace rambo {

// =============================================================================
// Result structure
// =============================================================================

struct IntegrationResult {
    double mean = 0.0;
    double error = 0.0;
    double sum = 0.0;
    double sumSquared = 0.0;
    int64_t nEvents = 0;
    
    void computeStatistics() {
        mean = sum / static_cast<double>(nEvents);
        double variance = (sumSquared / static_cast<double>(nEvents)) - (mean * mean);
        error = std::sqrt(std::fabs(variance) / static_cast<double>(nEvents));
    }
};

// =============================================================================
// Integrator Class
// =============================================================================

template <typename Integrand, int NumParticles, typename Algorithm = RamboAlgorithm<NumParticles>>
class RamboIntegrator {
public:
    RamboIntegrator(int64_t nEvents, const Integrand& integrand)
        : nEvents_(nEvents), integrand_(integrand) {}
    
    void run(double cmEnergy, const double* hostMasses,
             double& mean, double& error,
             uint64_t seed = 5489ULL) {
        
        IntegrationResult result;
        result.nEvents = nEvents_;
        
        sycl::queue queue{sycl::gpu_selector_v};
        
        double* deviceMasses = sycl::malloc_device<double>(NumParticles, queue);
        double* deviceSum = sycl::malloc_device<double>(1, queue);
        double* deviceSum2 = sycl::malloc_device<double>(1, queue);
        
        queue.memcpy(deviceMasses, hostMasses, sizeof(double) * NumParticles);
        queue.memset(deviceSum, 0, sizeof(double));
        queue.memset(deviceSum2, 0, sizeof(double));
        queue.wait();
        
        launchMonteCarloKernel(queue, cmEnergy, deviceMasses, deviceSum, deviceSum2, seed);
        
        queue.memcpy(&result.sum, deviceSum, sizeof(double));
        queue.memcpy(&result.sumSquared, deviceSum2, sizeof(double));
        queue.wait();
        
        sycl::free(deviceMasses, queue);
        sycl::free(deviceSum, queue);
        sycl::free(deviceSum2, queue);
        
        result.computeStatistics();
        
        mean = result.mean;
        error = result.error;
    }
    
private:
    int64_t nEvents_;
    Integrand integrand_;
    
    void launchMonteCarloKernel(sycl::queue& queue, double cmEnergy, 
                                 double* deviceMasses, double* deviceSum, 
                                 double* deviceSum2, uint64_t baseSeed) {
        
        auto device = queue.get_device();
        size_t maxWorkGroupSize = device.get_info<sycl::info::device::max_work_group_size>();
        size_t workGroupSize = std::min(maxWorkGroupSize, size_t(256));
        
        // Cap total threads to reduce atomic contention - each thread processes multiple events
        size_t maxBlocks = 1024;  // Reasonable cap for grid-stride efficiency
        size_t numBlocks = std::min(
            static_cast<size_t>((nEvents_ + workGroupSize - 1) / workGroupSize),
            maxBlocks
        );
        size_t globalSize = numBlocks * workGroupSize;
        
        Integrand integrand = integrand_;
        int64_t nEvents = nEvents_;
        
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>{globalSize, workGroupSize},
                [=](sycl::nd_item<1> item) {
                    int64_t idx = item.get_global_id(0);
                    
                    uint64_t rngState = baseSeed ^ (static_cast<uint64_t>(idx) * 2685821657736338717ULL);
                    if (rngState == 0) rngState = baseSeed + 1;
                    
                    double localSum = 0.0;
                    double localSum2 = 0.0;
                    
                    // Create generator once per thread - pre-computes mass-dependent quantities
                    PhaseSpaceGenerator<NumParticles, Algorithm> rambo(cmEnergy, deviceMasses);
                    
                    int64_t gridSize = static_cast<int64_t>(globalSize);
                    while (idx < nEvents) {
                        double momenta[NumParticles][4];
                        double logWeight = rambo(rngState, momenta);
                        
                        double fx = integrand.evaluate(momenta);
                        double weightedValue = fx * sycl::exp(logWeight);
                        
                        localSum += weightedValue;
                        localSum2 += weightedValue * weightedValue;
                        
                        idx += gridSize;
                    }
                    
                    auto sumRef = sycl::atomic_ref<double, 
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*deviceSum);
                    sumRef.fetch_add(localSum);
                    
                    auto sum2Ref = sycl::atomic_ref<double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(*deviceSum2);
                    sum2Ref.fetch_add(localSum2);
                });
        });
        
        queue.wait();
    }
};

} // namespace rambo

#endif // RAMBO_SYCL_INTEGRATOR_HPP
