// =============================================================================
// RAMBO Monte Carlo Integration - Unified Multi-Backend Implementation
// =============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <string>

#include <phirst/phirst.hpp>

// =============================================================================
// Benchmark helper
// =============================================================================
template <typename Integrand, int nParticles, typename Algorithm = phirst::RamboDietAlgorithm<nParticles>>
void runBenchmark(const std::string& backendName, 
                  int64_t nEvents, 
                  double cmEnergy, 
                  const double* masses,
                  const Integrand& integrand,
                  uint64_t seed,
                  bool useVegas) {
    
    std::cout << "----------------------------------------\n";
    std::cout << "Backend: " << backendName;
    if (useVegas) {
        std::cout << " (VEGAS)";
    } else {
        std::cout << " (Flat MC)";
    }
    std::cout << '\n';
    std::cout << "----------------------------------------\n";
    
    double mean = 0.0;
    double error = 0.0;
    
    // Warmup run
    {
        phirst::RamboIntegrator<Integrand, nParticles, Algorithm> warmup(
            std::min(nEvents / 10, int64_t(10000)), integrand);
        if (useVegas) {
            warmup.runVegas(cmEnergy, masses, mean, error, seed);
        } else {
            warmup.run(cmEnergy, masses, mean, error, seed);
        }
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    phirst::RamboIntegrator<Integrand, nParticles, Algorithm> integrator(nEvents, integrand);
    if (useVegas) {
        integrator.runVegas(cmEnergy, masses, mean, error, seed);
    } else {
        integrator.run(cmEnergy, masses, mean, error, seed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double timeMs = static_cast<double>(duration.count()) / 1000.0;
    
    std::cout << "  Mean: " << mean << '\n';
    std::cout << "  Error: " << error << '\n';
    std::cout << "  Time: " << timeMs << " ms\n";
    std::cout << "  Throughput: " << (static_cast<double>(nEvents) / timeMs * 1000.0) << " events/sec\n";
    std::cout << '\n';
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // -------------------------------------------------------------------------
    // Backend Initialization
    // -------------------------------------------------------------------------
#if defined(PHIRST_BACKEND_KOKKOS)
    Kokkos::initialize(argc, argv);
    {
#endif

    const int64_t nEvents = (argc > 1) ? std::stoll(argv[1]) : 1000000;
    const uint64_t seed = (argc > 2) ? std::stoull(argv[2]) : 5489ULL;
    const bool useVegas = (argc > 3) ? (std::stoi(argv[3]) != 0) : true;

    const double cmEnergy = 91.2;
    constexpr int nParticles = 2;
    
    std::cout << "========================================\n";
    std::cout << "RAMBO Monte Carlo Integrator : Drell-Yan\n";
    std::cout << "========================================\n";
    std::cout << "Library version: " << phirst::VERSION_MAJOR << "."
              << phirst::VERSION_MINOR << "." << phirst::VERSION_PATCH << '\n';
    std::cout << "Compiled backend: " << phirst::BACKEND_NAME << '\n';
    std::cout << "Number of events: " << nEvents << '\n';
    std::cout << "Random seed: " << seed << '\n';
    std::cout << "Center-of-mass energy: " << cmEnergy << " GeV\n";
    std::cout << "Number of particles: " << nParticles << '\n';
    std::cout << "Integration Mode: " << (useVegas ? "VEGAS" : "Flat MC") << '\n';
    std::cout << '\n';
    
    constexpr double electronMass = 0.000511;
    double masses[nParticles] = {electronMass, electronMass};
    
    const double quarkCharge = 2.0 / 3.0;
    const double alphaEM = 1.0 / 137.035999;
    phirst::DrellYanIntegrand integrand(quarkCharge, alphaEM);
    
    std::cout << "----------------------------------------\n";
    std::cout << "Drell-Yan Process: q qbar -> gamma* -> e+ e-\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Quark charge (e_q): " << quarkCharge << '\n';
    std::cout << "Fine structure constant (alpha): " << alphaEM << '\n';
    std::cout << '\n';
    
    runBenchmark<phirst::DrellYanIntegrand, nParticles, phirst::RamboDietAlgorithm<nParticles>>(
        phirst::BACKEND_NAME, nEvents, cmEnergy, masses, integrand, seed, useVegas);
    
    // Analytic verification
    std::cout << "========================================\n";
    std::cout << "Analytic Verification\n";
    std::cout << "========================================\n";
    double s = cmEnergy * cmEnergy;
    double analyticSigma = phirst::DrellYanIntegrand::analyticCrossSection(s, quarkCharge, alphaEM);
    
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Analytic cross-section:\n";
    std::cout << "  sigma = 4*pi*alpha^2*e_q^2 / (3*s) * hbarc^2\n";
    std::cout << "  s = " << s << " GeV^2\n";
    std::cout << "  sigma = " << analyticSigma << " mb\n";
    std::cout << "  sigma = " << analyticSigma * 1e6 << " nb\n";
    std::cout << "  sigma = " << analyticSigma * 1e9 << " pb\n";
    std::cout << '\n';
    
    std::cout << "======================================\n";
    std::cout << "Benchmark complete.\n";
    std::cout << "======================================\n";

    // -------------------------------------------------------------------------
    // Backend Finalization
    // -------------------------------------------------------------------------
#if defined(PHIRST_BACKEND_KOKKOS)
    }
    Kokkos::finalize();
#endif

    return 0;
}
