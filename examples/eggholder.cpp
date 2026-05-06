// =============================================================================
// RAMBO Monte Carlo Integration - Eggholder Test Function
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
    // Parse an optional 3rd argument to select VEGAS vs flat MC. Default to VEGAS.
    const bool useVegas = (argc > 3) ? (std::stoi(argv[3]) != 0) : true;

    const double cmEnergy = 10000.0;
    // The EggholderIntegrand provided requires exactly 3 particles
    constexpr int nParticles = 3;

    std::cout << "========================================\n";
    std::cout << "RAMBO Monte Carlo Integrator : Eggholder Test\n";
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

    double masses[nParticles] = {0.0, 0.0, 0.0};

    const double lambda = 1000000.0;
    phirst::EggholderIntegrand integrand(lambda);

    std::cout << "----------------------------------------\n";
    std::cout << "Eggholder Integrand Parameters:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Lambda squared: " << lambda << '\n';
    std::cout << '\n';

    runBenchmark<phirst::EggholderIntegrand, nParticles, phirst::RamboDietAlgorithm<nParticles>>(
        phirst::BACKEND_NAME, nEvents, cmEnergy, masses, integrand, seed, useVegas);

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
