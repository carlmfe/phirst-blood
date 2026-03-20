#pragma once
#include <cstdint>

class Random
{
  public:
  double ranmar();
  void rmarin(int ij, int kl);
  
  private:
  double ranu[98];
  double ranc, rancd, rancm;
  int iranmr, jranmr;
};

double rn(int idummy);

// Device-friendly xorshift RNG types and functions
#pragma omp declare target
struct XorShift64State {
  uint64_t s;
};

// splitmix64 for seeding (good mixing)
static inline uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// initialize state from a seed (64-bit)
static inline void xorshift64_seed(XorShift64State &st, uint64_t seed) {
  uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
  st.s = splitmix64(z) | 1ULL; // never zero
}

// xorshift64 (Marsaglia) next value
static inline uint64_t xorshift64_next(XorShift64State &st) {
  uint64_t x = st.s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  st.s = x;
  return x;
}

// return double in [0,1)
static inline double xorshift64_rand(XorShift64State &st) {
  uint64_t v = xorshift64_next(st);
  // Convert to double in [0,1), using 53 bits
  return (double)(v >> 11) * (1.0 / 9007199254740992.0);
}
#pragma omp end declare target
