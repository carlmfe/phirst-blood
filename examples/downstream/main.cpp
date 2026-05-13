// Minimal downstream example: Monte Carlo integration of a constant integrand.
// Demonstrates that phirst headers and backend macros are available after
// find_package(phirst REQUIRED) + target_link_libraries(... phirst::<backend>).

#include <phirst/integrator.hpp>
#include <phirst/integrands.hpp>
#include <phirst/phase_space.hpp>

#include <iostream>

int main() {
    constexpr int    nParticles = 2;
    constexpr double cmEnergy   = 91.2;  // Z-boson mass [GeV]
    constexpr double masses[]   = {0.0, 0.0};
    constexpr int    nEvents    = 100000;
    constexpr int    seed       = 5489;

    phirst::ConstantIntegrand integrand{1.0};
    phirst::RamboIntegrator<phirst::ConstantIntegrand, nParticles>
        integrator(nEvents, integrand);

    double mean = 0.0, error = 0.0;
    integrator.run(cmEnergy, masses, mean, error, seed);

    std::cout << "Mean:  " << mean  << "\n"
              << "Error: " << error << "\n";
    return 0;
}
