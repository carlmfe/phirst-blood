#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <stdlib.h>

#include "rng.h"
#include "rambo_omp.h"

using namespace std;


#pragma omp declare target
void rambo_device(double energy, const double* masses, int n,
                  XorShift64State &rng,
                  double momenta_out[MAX_PARTICLES][4],
                  double &weight) {
  /**********************************************************************
   *                       rambo                                         *
   *    ra(ndom)  m(omenta)  b(eautifully)  o(rganized)                  *
   *                                                                     *
   *    a democratic multi-particle phase space generator                *
   *    authors:  s.d. ellis,  r. kleiss,  w.j. stirling                 *
   *    this is version 1.0 -  written by r. kleiss                      *
   *    -- adjusted by hans kuijf, weights are logarithmic (20-08-90)    *
   *                                                                     *
   *    n  = number of particles                                         *
   *    energy = total centre-of-mass energy                             *
   *    masses = particle masses ( dim=nexternal-nincoming )             *
   *    p  = particle momenta ( dim=(4,nexternal-nincoming) )            *
   *    weight = weight of the event                                     *
   ***********************************************************************/
  // Sanity checks: device-safe (no cout/exit)
  if (n < 1 || n > MAX_PARTICLES) {
    // invalid; mark weight with sentinel and return
    weight = -INFINITY;
    for (int i=0;i<MAX_PARTICLES;i++)
      for (int k=0;k<4;k++) momenta_out[i][k] = 0.0;
    return;
  }

  // Local (stack) arrays sized to MAX_PARTICLES
  double q[MAX_PARTICLES][4];
  double p[MAX_PARTICLES][4];
  double z[MAX_PARTICLES];
  double r[4];
  double b[3];
  double p2[MAX_PARTICLES];
  double xm2[MAX_PARTICLES];
  double e[MAX_PARTICLES];
  double v[MAX_PARTICLES];

  // local constants
  const double acc = 1e-14;
  const int itmax = 1000;
  const double twopi = 8.0 * atan(1.0);
  const double po2log = log(twopi / 4.0);

  // copy masses and compute total mass & nm (non-zero masses)
  double masssum = 0.0;
  int nm = 0;
  for (int i = 0; i < n; ++i) {
    masssum += fabs(masses[i]);
    if (masses[i] != 0.0) ++nm;
  }
  if (masssum > energy) {
    // not enough energy
    weight = -INFINITY;
    for (int i=0;i<n;i++)
      for (int k=0;k<4;k++) momenta_out[i][k] = 0.0;
    return;
  }

  // Generate massless momenta in infinite phase space
  for (int i = 0; i < n; ++i) {
    double r1 = xorshift64_rand(rng);
    double c = 2.0 * r1 - 1.0;
    double s = sqrt(1.0 - c * c);
    double f = twopi * xorshift64_rand(rng);
    r1 = xorshift64_rand(rng);
    double r2 = xorshift64_rand(rng);
    q[i][0] = -log(r1 * r2);
    q[i][3] = q[i][0] * c;
    q[i][2] = q[i][0] * s * cos(f);
    q[i][1] = q[i][0] * s * sin(f);
  }

  // Conformal transformation params
  r[0] = r[1] = r[2] = r[3] = 0.0;
  for (int i = 0; i < n; ++i) {
    r[0] += q[i][0];
    r[1] += q[i][1];
    r[2] += q[i][2];
    r[3] += q[i][3];
  }
  double rmas = sqrt(r[0]*r[0] - r[1]*r[1] - r[2]*r[2] - r[3]*r[3]);
  b[0] = -r[1] / rmas;
  b[1] = -r[2] / rmas;
  b[2] = -r[3] / rmas;
  double g = r[0] / rmas;
  double a = 1.0 / (1.0 + g);
  double x = energy / rmas;

  // Transform q -> p
  for (int i = 0; i < n; ++i) {
    double bq = b[0]*q[i][1] + b[1]*q[i][2] + b[2]*q[i][3];
    for (int k = 1; k < 4; ++k) {
      p[i][k] = x * ( q[i][k] + b[k-1] * ( q[i][0] + a * bq ) );
    }
    p[i][0] = x * ( g * q[i][0] + bq );
  }

  // Build z[] for weight (small cost on device)
  if (n > 1) {
    // original code used 1-based indexing; we replicate logic
    for (int k = 0; k < n; ++k) z[k] = 0.0;
    if (n >= 2) z[1] = po2log;
    for (int k = 2; k < n; ++k) {
      z[k] = z[k-1] + po2log - 2.0 * log(double(k-1));
    }
    for (int k = 2; k < n; ++k) {
      z[k] = z[k] - log(double(k));
    }
  }

  // weight for massless phase space (log weight)
  if (n == 2) {
    weight = po2log;
  } else {
    // z[n-1] corresponds to original z[n-1] since we filled z[1..n-1]
    weight = (2.0 * n - 4.0) * log(energy) + z[n-1];
  }

  // If no masses, write out and return
  if (nm == 0) {
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 4; ++k) momenta_out[i][k] = p[i][k];
    }
    return;
  }

  // Massive rescaling: compute p2 and xm2
  for (int i = 0; i < n; ++i) {
    xm2[i] = masses[i] * masses[i];
    p2[i] = p[i][0] * p[i][0];
  }

  // Initial xmax / x
  double xmt = 0.0;
  for (int i = 0; i < n; ++i) xmt += fabs(masses[i]);
  double xmax = sqrt(1.0 - (xmt * xmt) / (energy * energy));
  x = xmax;

  // Solve for x using a Newton-like iteration
  int iter = 0;
  double accu = energy * acc;
  while (true) {
    double f0 = -energy;
    double g0 = 0.0;
    for (int i = 0; i < n; ++i) {
      e[i] = sqrt(xm2[i] + x * x * p2[i]);
      f0 += e[i];
      g0 += p2[i] / e[i];
    }
    if (fabs(f0) <= accu) break;
    ++iter;
    if (iter > itmax) break;
    x = x - f0 / (x * g0);
    if (!(x > 0.0)) { // numerical safety
      weight = -INFINITY;
      for (int i=0;i<n;i++)
        for (int k=0;k<4;k++) momenta_out[i][k] = 0.0;
      return;
    }
  }

  // finalize massive four-momenta & compute v[i] and e[i] as energies
  double weight2 = 1.0;
  double weight3 = 0.0;
  for (int i = 0; i < n; ++i) {
    v[i] = x * p[i][0];
    double px = x * p[i][1];
    double py = x * p[i][2];
    double pz = x * p[i][3];
    double Ei = sqrt(px*px + py*py + pz*pz + xm2[i]);
    e[i] = Ei;
    momenta_out[i][0] = Ei;
    momenta_out[i][1] = px;
    momenta_out[i][2] = py;
    momenta_out[i][3] = pz;

    weight2 *= v[i] / e[i];
    weight3 += (v[i] * v[i]) / e[i];
  }

  double weightm = (2.0 * n - 3.0) * log(x) + log(weight2 / weight3 * energy);
  weight = weight + weightm;

  return;
}
#pragma omp end declare target
