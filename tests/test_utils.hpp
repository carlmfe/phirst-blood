#pragma once
// Portable isfinite for all PHIRST backends.
// DPC++ SYCL exposes isfinite only in sycl::, not in std:: or the global namespace.
// Include this header (before any test code) and use the unqualified isfinite.
#include <cmath>
#if defined(PHIRST_BACKEND_SYCL)
#include <sycl/sycl.hpp>
using sycl::isfinite;
#else
using std::isfinite;
#endif
