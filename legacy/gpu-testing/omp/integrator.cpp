#include "integrator.h"

#include <vector>
#include <cmath>
#include "rng.h"
#include "rambo_omp.h"

using namespace std;

void integrator_omp(int64_t nEvents, double energy, const double* masses_d, const int nParticles, double &mean, double &error) {
  // Host array of RNG states; we'll map it to device and use there
  XorShift64State *states_d = new XorShift64State[nEvents];

  double sum_h = 0.0;
  double sum2_h = 0.0;

  // Initialize states on the device in parallel using splitmix seeding
  const uint64_t base_seed = 123456789ULL;
  #pragma omp target map(tofrom: states_d[0:nEvents])
  {
    #pragma omp teams distribute parallel for simd
    for (int i = 0; i < nEvents; ++i) {
      uint64_t s = base_seed + (uint64_t)i * 2ULL + 1ULL;
      // call splitting + seed
      xorshift64_seed(states_d[i], s);
    }
  }

  #pragma omp target map(to: states_d[0:nEvents]) map(to: masses_d[0:nParticles]) map(tofrom: sum_h, sum2_h)
  {
    #pragma omp teams distribute parallel for reduction(+:sum_h,sum2_h)
    for (int i = 0; i < nEvents; ++i) {
      XorShift64State rng = states_d[i];
      double weight;
      double momenta_local[MAX_PARTICLES][4];
      // Pass the raw data pointer and particle count
      rambo_device(energy, masses_d, nParticles, rng, momenta_local, weight);
      double integrand_value = integrand_device(nParticles, momenta_local);
      double wval = integrand_value * std::exp(weight); // note: weight is log-weight in many RAMBO implementations
      sum_h += wval;
      sum2_h += wval * wval;
    }
  }
  mean = sum_h / nEvents;
  double variance = (sum2_h / nEvents) - (mean*mean);
  error = sqrt(variance / nEvents);
  delete[] states_d;
}

void integrator_2particle_omp(int64_t nEvents, double energy, const double* masses_d, double &mean, double &error) {
  // Host array of RNG states; we'll map it to device and use there
  XorShift64State *states_d = new XorShift64State[nEvents];

  double sum_h = 0.0;
  double sum2_h = 0.0;

  // Initialize states on the device in parallel using splitmix seeding
  const uint64_t base_seed = 123456789ULL;
  #pragma omp target map(tofrom: states_d[0:nEvents])
  {
    #pragma omp teams distribute parallel for simd
    for (int i = 0; i < nEvents; ++i) {
      uint64_t s = base_seed + (uint64_t)i * 2ULL + 1ULL;
      // call splitting + seed
      xorshift64_seed(states_d[i], s);
    }
  }

  #pragma omp target map(to: states_d[0:nEvents]) map(to: masses_d[0:2]) map(tofrom: sum_h, sum2_h)
  {
    #pragma omp teams distribute parallel for reduction(+:sum_h,sum2_h)
    for (int i = 0; i < nEvents; ++i) {
      XorShift64State rng = states_d[i];
      double weight;
      double momenta_local[2][4];
      // Pass the raw data pointer and particle count
      rambo_device(energy, masses_d, 2, rng, momenta_local, weight);
      double integrand_value = integrand_2particle_device(momenta_local);
      double wval = integrand_value * std::exp(weight); // note: weight is log-weight in many RAMBO implementations
      sum_h += wval;
      sum2_h += wval * wval;
    }
  }
  mean = sum_h / nEvents;
  double variance = (sum2_h / nEvents) - (mean*mean);
  error = sqrt(variance / nEvents);
  delete[] states_d;
}


#pragma omp declare target
double integrand_device(int n, const double momenta_out[][4]) {
  double result = 0.0;
  for (int i = 0; i < n; ++i) {
    result += momenta_out[i][0]; // example: sum of energies
  }
  return result;
}
#pragma omp end declare target

#pragma omp declare target
double integrand_2particle_device(const double momenta[2][4]) {
  // Example specialized integrand for 2-particle case

  double tot_momentum[4];
  for (int k = 0; k < 4; k++) {
    tot_momentum[k] = momenta[0][k] + momenta[1][k];
  }
  double s = tot_momentum[0] * tot_momentum[0] - tot_momentum[1] * tot_momentum[1]
              - tot_momentum[2] * tot_momentum[2] - tot_momentum[3] * tot_momentum[3];
  double m12 = momenta[0][0] * momenta[0][0] - momenta[0][1] * momenta[0][1]
              - momenta[0][2] * momenta[0][2] - momenta[0][3] * momenta[0][3];
  double m22 = momenta[1][0] * momenta[1][0] - momenta[1][1] * momenta[1][1]
              - momenta[1][2] * momenta[1][2] - momenta[1][3] * momenta[1][3];
  double t = m12 - sqrt(s) * (momenta[0][0] - momenta[0][3]);
  double u = m22 - sqrt(s) * (momenta[1][0] - momenta[1][3]);

  return (t * t + u * u + 4 * s * m22 - 2 * m22 * m22) / pow(s - 1000.0, 2);
}
#pragma omp end declare target
