#pragma once

#include <vector>
#include "rng.h"

using namespace std;

#pragma omp declare target
const int MAX_PARTICLES = 10;
#pragma omp end declare target

#pragma omp declare target
void rambo_device(double energy, const double* masses, int n,
                  XorShift64State &rng,
                  double momenta_out[MAX_PARTICLES][4],
                  double &weight);
#pragma omp end declare target
