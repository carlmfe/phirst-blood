// RAMBO Monte Carlo Integrator using Alpaka 2.0.0
// Example application demonstrating the rambo::alpaka library
//
// Build with different backends:
//   cmake -DALPAKA_BACKEND=CUDA ..   # NVIDIA GPU
//   cmake -DALPAKA_BACKEND=CPU ..    # CPU serial
//   cmake -DALPAKA_BACKEND=OMP ..    # OpenMP parallel

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>

#include <rambo/rambo.hpp>

// =============================================================================
// Backend Selection (compile-time)
// =============================================================================
#if defined(ALPAKA_USE_CUDA)
    using DefaultAccTag = alpaka::TagGpuCudaRt;
    constexpr const char* BACKEND_NAME = "CUDA GPU (NVIDIA)";
#elif defined(ALPAKA_USE_HIP)
    using DefaultAccTag = alpaka::TagGpuHipRt;
    constexpr const char* BACKEND_NAME = "HIP GPU (AMD)";
#elif defined(ALPAKA_USE_SYCL)
    using DefaultAccTag = alpaka::TagGpuSyclIntel;
    constexpr const char* BACKEND_NAME = "SYCL GPU (Intel)";
#elif defined(ALPAKA_USE_CPU_OMP)
    using DefaultAccTag = alpaka::TagCpuOmp2Blocks;
    constexpr const char* BACKEND_NAME = "CPU OpenMP";
#elif defined(ALPAKA_USE_CPU_SERIAL)
    using DefaultAccTag = alpaka::TagCpuSerial;
    constexpr const char* BACKEND_NAME = "CPU Serial";
#else
    using DefaultAccTag = std::tuple_element_t<0, alpaka::EnabledAccTags>;
    constexpr const char* BACKEND_NAME = "Auto-detected";
#endif

// =============================================================================
// Benchmark helper
// =============================================================================
template <typename AccTag, rambo::IntegrandConcept Integrand, int nParticles, typename Algorithm = rambo::RamboAlgorithm<nParticles>>
void runBenchmark(const std::string& backendName, 
                  int64_t nEvents, 
                  double cmEnergy, 
                  const double* masses,
                  const Integrand& integrand,
                  uint64_t seed) {
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Backend: " << backendName << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    double mean = 0.0;
    double error = 0.0;
    
    // Warmup run (smaller)
    {
        rambo::RamboIntegrator<AccTag, Integrand, nParticles, Algorithm> warmup(
            std::min(nEvents / 10, int64_t(10000)), integrand);
        warmup.run(cmEnergy, masses, mean, error, seed);
    }
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    rambo::RamboIntegrator<AccTag, Integrand, nParticles, Algorithm> integrator(nEvents, integrand);
    integrator.run(cmEnergy, masses, mean, error, seed);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double timeMs = duration.count() / 1000.0;
    
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Error: " << error << std::endl;
    std::cout << "  Time: " << timeMs << " ms" << std::endl;
    std::cout << "  Throughput: " << (nEvents / timeMs * 1000.0) << " events/sec" << std::endl;
    std::cout << std::endl;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Parse command line arguments
    const int64_t nEvents = (argc > 1) ? std::stoll(argv[1]) : 100000;
    const uint64_t seed = (argc > 2) ? std::stoull(argv[2]) : 5489ULL;
    const double cmEnergy = 91.2;  // Z-pole energy in GeV (classic Drell-Yan)
    constexpr int nParticles = 2;   // e+ e- final state
    
    std::cout << "======================================" << std::endl;
    std::cout << "RAMBO Monte Carlo Integrator (Alpaka)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Compiled backend: " << BACKEND_NAME << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV" << std::endl;
    std::cout << "Number of particles: " << nParticles << std::endl;
    std::cout << std::endl;
    
    // Print available accelerators
    std::cout << "Available accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    std::cout << std::endl;
    
    // Set up particle masses for e+ e- (electron mass in GeV)
    constexpr double electronMass = 0.000511;
    double masses[nParticles] = {electronMass, electronMass};
    
    // Create Drell-Yan integrand (up-quark charge = 2/3)
    const double quarkCharge = 2.0 / 3.0;
    const double alphaEM = 1.0 / 137.035999;
    rambo::DrellYanIntegrand integrand(quarkCharge, alphaEM);
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Drell-Yan Process: q qbar -> gamma* -> e+ e-" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Quark charge (e_q): " << quarkCharge << std::endl;
    std::cout << "Fine structure constant (alpha): " << alphaEM << std::endl;
    std::cout << std::endl;
    
    // Run benchmark with compiled backend
    runBenchmark<DefaultAccTag, rambo::DrellYanIntegrand, nParticles, rambo::RamboAlgorithm<nParticles>>(
        BACKEND_NAME, nEvents, cmEnergy, masses, integrand, seed);
    
    // Analytic verification
    std::cout << "========================================" << std::endl;
    std::cout << "Analytic Verification" << std::endl;
    std::cout << "========================================" << std::endl;
    double s = cmEnergy * cmEnergy;
    double analyticSigma = rambo::DrellYanIntegrand::analyticCrossSection(s, quarkCharge, alphaEM);
    std::cout << std::scientific;
    std::cout << "Analytic cross-section:" << std::endl;
    std::cout << "  sigma = 4*pi*alpha^2*e_q^2 / (3*s) * hbarc^2" << std::endl;
    std::cout << "  s = " << s << " GeV^2" << std::endl;
    std::cout << "  sigma = " << analyticSigma << " mb" << std::endl;
    std::cout << "  sigma = " << analyticSigma * 1e6 << " nb" << std::endl;
    std::cout << "  sigma = " << analyticSigma * 1e9 << " pb" << std::endl;
    std::cout << std::endl;
    
    std::cout << "======================================" << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}
