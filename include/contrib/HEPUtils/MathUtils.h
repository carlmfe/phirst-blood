// -*- C++ -*-
//
// This file is part of HEPUtils -- https://gitlab.com/hepcedar/heputils/
// Copyright (C) 2013-2023 Andy Buckley <andy.buckley@cern.ch>
//
// Embedding of HEPUtils code in other projects is permitted provided this
// notice is retained and the HEPUtils namespace and include path are changed.
//
#pragma once

#if __cplusplus <= 199711L
#error "This library needs at least a C++11 compliant compiler: are you using -std=c++11?"
#endif

#include <phirst/backend/config.hpp>
#include <phirst/backend/math.hpp>
#include <type_traits>
#if defined(PHIRST_BACKEND_SERIAL)
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstddef>
#endif

/// @file Convenience maths functions for HEP MC physics
/// @author Andy Buckley <andy.buckley@cern.ch>

namespace HEPUtils {


  /// @defgroup mathutils Mathematical utilities
  /// @{

  /// @defgroup mathutils_numhelp Numerical helpers
  /// @{

  /// Convenience function for getting the sign of a number
  template <typename N1>
  PHIRST_HOST_DEVICE
  constexpr typename std::enable_if<std::is_arithmetic<N1>::value, int>::type
  sign(const N1& val) {
    // if (val == 0) return 0;
    // return (val < 0) ? -1 : 1;
    return (val == 0) ? 0 : (val < 0) ? -1 : 1;
  }

  /// Convenience function for squaring (better than repeating long expressions/calcs or using intermediates)
  template <typename N1>
  PHIRST_HOST_DEVICE
  constexpr typename std::enable_if<std::is_arithmetic<N1>::value, N1>::type
  sqr(const N1 val) {
    return val * val;
  }

  /// Convenience function for adding two numbers in quadrature
  template <typename N1, typename N2>
  PHIRST_HOST_DEVICE
  inline typename std::enable_if<std::is_arithmetic<N1>::value && std::is_arithmetic<N2>::value, typename std::common_type<N1,N2>::type>::type
  add_quad(const N1& a, const N2& b) {
    return phirst::math::sqrt(a*a + b*b);
  }

  /// Convenience function for adding three numbers in quadrature
  template <typename N1, typename N2, typename N3>
  PHIRST_HOST_DEVICE
  inline typename std::enable_if<std::is_arithmetic<N1>::value && std::is_arithmetic<N2>::value && std::is_arithmetic<N3>::value, typename std::common_type<N1,N2,N3>::type>::type
  add_quad(const N1& a, const N2& b, const N3& c) {
    return phirst::math::sqrt(a*a + b*b + c*c);
  }

  /// @}


  /// @defgroup mathutils_rangehelp Range helpers
  /// @{

  /// @brief Boolean function to determine if @a value is within the given range
  ///
  /// @note The interval is closed (inclusive) at the low end, and open (exclusive) at the high end.
  template <typename N1, typename N2, typename N3>
  PHIRST_HOST_DEVICE
  constexpr typename std::enable_if<std::is_arithmetic<N1>::value && std::is_arithmetic<N2>::value && std::is_arithmetic<N3>::value, bool>::type
  in_range(const N1& val, const N2& low, const N3& high) {
    return val >= low && val < high;
  }

  /// @brief Boolean function to determine if @a value is within the given range
  ///
  /// @note The interval is closed at both ends.
  template <typename N1, typename N2, typename N3>
  PHIRST_HOST_DEVICE
  constexpr typename std::enable_if<std::is_arithmetic<N1>::value && std::is_arithmetic<N2>::value && std::is_arithmetic<N3>::value, bool>::type
  in_closed_range(const N1& val, const N2& low, const N3& high) {
    return val >= low && val <= high;
  }

  /// @brief Boolean function to determine if @a value is within the given range
  ///
  /// @note The interval is open at both ends.
  template <typename N1, typename N2, typename N3>
  PHIRST_HOST_DEVICE
  constexpr typename std::enable_if<std::is_arithmetic<N1>::value && std::is_arithmetic<N2>::value && std::is_arithmetic<N3>::value, bool>::type
  in_open_range(const N1& val, const N2& low, const N3& high) {
    return val > low && val < high;
  }


#if defined(PHIRST_BACKEND_SERIAL)
  /// Make a list of @a nbins + 1 values linearly spaced between @a start and @a end inclusive.
  inline std::vector<double> linspace(size_t nbins, double start, double end, bool include_end=true) {
    assert(end >= start);
    assert(nbins > 0);
    std::vector<double> rtn;
    const double interval = (end-start)/static_cast<double>(nbins);
    // Add all edges except the last
    for (size_t i = 0; i < nbins; ++i) {
      rtn.push_back(start + i*interval);
    }
    // Add the last edge
    assert(rtn.size() == nbins);
    if (include_end) rtn.push_back(end); // exact end, not result of n * interval
    return rtn;
  }


  /// Make a list of @a nbins + 1 values exponentially spaced between @a start and @a end inclusive.
  inline std::vector<double> logspace(size_t nbins, double start, double end, double offset=0, bool include_end=true) {
    assert(end >= start);
    assert(start+offset > 0);
    assert(nbins > 0);
    const double logstart = std::log(start+offset);
    const double logend = std::log(end+offset);
    const std::vector<double> logvals = linspace(nbins, logstart, logend, false);
    assert(logvals.size() == nbins);
    std::vector<double> rtn; rtn.reserve(nbins+1);
    rtn.push_back(start); //< exact start, not exp(log(start))
    for (size_t i = 1; i < logvals.size(); ++i) {
      rtn.push_back(std::exp(logvals[i]));
    }
    assert(rtn.size() == nbins);
    if (include_end) rtn.push_back(end); //< exact end, not exp(n * loginterval)
    return rtn;
  }

  /// @}

#endif // PHIRST_BACKEND_SERIAL

  /// @defgroup mathutils_physics Physics maths utils
  /// @{

  PHIRST_HOST_DEVICE
  inline double deltaphi(double a, double b) {
    double rtn = a - b;
    rtn = phirst::math::fmod(rtn, 2*phirst::math::pi);
    if (rtn == 0) return 0;
    if (rtn >  phirst::math::pi) rtn -= 2*phirst::math::pi;
    if (rtn <= -phirst::math::pi) rtn += 2*phirst::math::pi;
    return phirst::math::fabs(rtn);
  }

  /// @}


  /// @defgroup mathutils_random Random numbers and sampling
  /// @{

#if defined(PHIRST_BACKEND_SERIAL)
  inline double rand01() {
    return rand() / (double)RAND_MAX;
  }
#endif

  /// @}


}
