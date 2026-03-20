// RAMBO Monte Carlo Integrator using pure CUDA
// Example application demonstrating the rambo::cuda library
//
// Build:
//   mkdir build && cd build
//   cmake ..
//   make

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>

#include <rambo/rambo.cuh>

// =============================================================================
// Benchmark helper
// =============================================================================
template <typename Integrand, int nParticles, typename Algorithm = rambo::RamboAlgorithm<nParticles>>
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
        rambo::RamboIntegrator<Integrand, nParticles, Algorithm> warmup(
            std::min(nEvents / 10, int64_t(10000)), integrand);
        warmup.run(cmEnergy, masses, mean, error, seed);
    }
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    rambo::RamboIntegrator<Integrand, nParticles, Algorithm> integrator(nEvents, integrand);
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
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "======================================" << std::endl;
    std::cout << "RAMBO Monte Carlo Integrator (CUDA)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Compiled backend: CUDA GPU" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV" << std::endl;
    std::cout << "Number of particles: " << nParticles << std::endl;
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
    
    // Run benchmark using RAMBO on diet phase-space
    runBenchmark<rambo::DrellYanIntegrand, nParticles, rambo::RamboAlgorithm<nParticles>>(
        "CUDA GPU", nEvents, cmEnergy, masses, integrand, seed);
    
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
