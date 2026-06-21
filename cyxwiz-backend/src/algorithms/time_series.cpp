// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
} // namespace cyxwiz




