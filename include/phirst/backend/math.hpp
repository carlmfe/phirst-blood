#pragma once
#ifndef PHIRST_MATH_HPP
#define PHIRST_MATH_HPP

/**
 * @file math.hpp
 * @brief Portable math function wrappers for all backends.
 * 
 * Provides unified access to mathematical functions that work correctly
 * on both host and device code across all supported backends.
 * 
 * Usage:
 *   #include <phirst/math.hpp>
 *   double x = phirst::math::sqrt(2.0);
 *   double y = phirst::math::log(x);
 */

#include "config.hpp"

namespace phirst {
namespace math {

// =============================================================================
// Basic Math Functions
// =============================================================================

/**
 * Square root.
 */
PHIRST_INLINE auto sqrt(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::sqrt(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

/**
 * Natural logarithm.
 */
PHIRST_INLINE auto log(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::log(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::log(x);
#else
    return std::log(x);
#endif
}

/**
 * Exponential function.
 */
PHIRST_INLINE auto exp(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::exp(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::exp(x);
#else
    return std::exp(x);
#endif
}

/**
 * Power function.
 */
PHIRST_INLINE auto pow(double base, double exp) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::pow(base, exp);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::pow(base, exp);
#else
    return std::pow(base, exp);
#endif
}

/**
 * Integer power (more efficient for integer exponents).
 */
PHIRST_INLINE auto pow(double base, int exp) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::pown(base, exp);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::pow(base, static_cast<double>(exp));
#else
    return std::pow(base, exp);
#endif
}

/**
 * Absolute value.
 */
PHIRST_INLINE auto fabs(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::fabs(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::fabs(x);
#else
    return std::fabs(x);
#endif
}

/**
 * Maximum of two values.
 */
PHIRST_INLINE auto fmax(double x, double y) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::fmax(x, y);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::fmax(x, y);
#else
    return std::fmax(x, y);
#endif
}

/**
 * Minimum of two values.
 */
PHIRST_INLINE auto fmin(double x, double y) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::fmin(x, y);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::fmin(x, y);
#else
    return std::fmin(x, y);
#endif
}

// =============================================================================
// Trigonometric Functions
// =============================================================================

/**
 * Sine.
 */
PHIRST_INLINE auto sin(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::sin(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::sin(x);
#else
    return std::sin(x);
#endif
}

/**
 * Cosine.
 */
PHIRST_INLINE auto cos(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::cos(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::cos(x);
#else
    return std::cos(x);
#endif
}

/**
 * Tangent.
 */
PHIRST_INLINE auto tan(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::tan(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::tan(x);
#else
    return std::tan(x);
#endif
}

/**
 * Arc sine.
 */
PHIRST_INLINE auto asin(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::asin(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::asin(x);
#else
    return std::asin(x);
#endif
}

/**
 * Arc cosine.
 */
PHIRST_INLINE auto acos(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::acos(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::acos(x);
#else
    return std::acos(x);
#endif
}

/**
 * Arc tangent.
 */
PHIRST_INLINE auto atan(double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::atan(x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::atan(x);
#else
    return std::atan(x);
#endif
}

/**
 * Two-argument arc tangent.
 */
PHIRST_INLINE auto atan2(double y, double x) -> double {
#if defined(PHIRST_BACKEND_SYCL)
    return sycl::atan2(y, x);
#elif defined(PHIRST_BACKEND_KOKKOS)
    return Kokkos::atan2(y, x);
#else
    return std::atan2(y, x);
#endif
}

// =============================================================================
// Constants
// =============================================================================

/// Mathematical constant pi
inline constexpr double pi = 3.14159265358979323846264338327950288;

/// 2 * pi
inline constexpr double twoPi = 6.28318530717958647692528676655900577;

/// pi / 2
inline constexpr double halfPi = 1.57079632679489661923132169163975144;

/// log(pi / 2)
inline constexpr double logPiOver2 = 0.45158270528945486472619522989488;

/// Euler's number e
inline constexpr double e = 2.71828182845904523536028747135266250;

/// Natural log of 2
inline constexpr double ln2 = 0.69314718055994530941723212145817657;

} // namespace math
} // namespace phirst

#endif // PHIRST_MATH_HPP
