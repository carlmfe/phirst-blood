#include <iostream>
#include <cstdlib>
#include <cstdint>


#include "rng.h"
#include "rambo_omp.h"
#include "integrator.h"



int main(int argc, char* argv[]) {
  
  const int64_t nEvents = (argc > 1) ? std::stoll(argv[1]) : 100000;
  const double energy = 1000.0;
  const int nParticles = 2;
  double* masses_h = new double[nParticles];
  for (int i = 0; i < nParticles; ++i) {
    masses_h[i] = 0.0;
  }

  double mean, error;

  integrator_2particle_omp(nEvents, energy, masses_h, mean, error);

  std::cout << "Mean: " << mean << ", Error: " << error << std::endl;
  delete[] masses_h;

  return 0;
}
