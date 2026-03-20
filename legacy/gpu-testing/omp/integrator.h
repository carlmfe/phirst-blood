#pragma once
#include <cstdint>

void integrator_omp(int64_t nEvents, double energy, const double* masses_d, const int nParticles, double &mean, double &error);

void integrator_2particle_omp(int64_t nEvents, double energy, const double* masses_d, double &mean, double &error);

#pragma omp declare target
double integrand_device(int n, const double momenta_out[][4]);
#pragma omp end declare target

#pragma omp declare target
double integrand_2particle_device(const double momenta_out[4][4]);
#pragma omp end declare target
