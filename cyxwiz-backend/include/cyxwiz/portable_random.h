// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Seeded draws that are the same on every compiler and OS (TOFIX205). The
// standard fixes std::mt19937 / std::mt19937_64's raw output, but not
// std::shuffle or the std::*_distribution classes: MSVC, libstdc++ and libc++
// turn one seed into different orders and values. Every draw here is built
// only on the raw output. Integers, shuffles and uniform values are
// bit-identical everywhere; normal, gamma, binomial and Poisson values use
// std::log / std::exp / std::cos, which may differ in the last ulp between C
// runtimes. Use these instead of std::shuffle and std::*_distribution.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace cyxwiz::portable {

namespace detail {

template <class Engine>
constexpr int EngineBits() {
    using R = typename Engine::result_type;
    static_assert(Engine::min() == 0, "portable draws need an engine whose output starts at 0");
    static_assert(Engine::max() == std::numeric_limits<uint32_t>::max() ||
                      (sizeof(R) >= 8 && Engine::max() == std::numeric_limits<uint64_t>::max()),
                  "portable draws need a full 32- or 64-bit engine (std::mt19937, std::mt19937_64)");
    return Engine::max() == std::numeric_limits<uint32_t>::max() ? 32 : 64;
}

constexpr double kTwoPi = 6.283185307179586476925286766559;

}  // namespace detail

// 64 random bits: one draw of a 64-bit engine, or two of a 32-bit engine (the
// first is the high half).
template <class Engine>
uint64_t Bits64(Engine& rng) {
    if constexpr (detail::EngineBits<Engine>() == 64) {
        return static_cast<uint64_t>(rng());
    } else {
        const uint64_t high = static_cast<uint32_t>(rng());
        return (high << 32) | static_cast<uint32_t>(rng());
    }
}

// Uniform in [0, bound), unbiased; bound must be above 0. A 32-bit engine with
// a 32-bit bound uses Lemire's method with rejection (libsvm's and
// scikit-learn's bounded_rand_int, so their shuffles are reproduced); anything
// else rejects 64-bit words above the largest multiple of the bound.
template <class Engine>
uint64_t Below(Engine& rng, uint64_t bound) {
    if constexpr (detail::EngineBits<Engine>() == 32) {
        if (bound <= std::numeric_limits<uint32_t>::max()) {
            const auto range = static_cast<uint32_t>(bound);
            uint32_t x = static_cast<uint32_t>(rng());
            uint64_t m = static_cast<uint64_t>(x) * range;
            uint32_t low = static_cast<uint32_t>(m);
            if (low < range) {
                uint32_t threshold = 0u - range;
                if (threshold >= range) {
                    threshold -= range;
                    if (threshold >= range) threshold %= range;
                }
                while (low < threshold) {
                    x = static_cast<uint32_t>(rng());
                    m = static_cast<uint64_t>(x) * range;
                    low = static_cast<uint32_t>(m);
                }
            }
            return m >> 32;
        }
    }
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    const uint64_t limit = max - max % bound;
    uint64_t value = Bits64(rng);
    while (value >= limit) value = Bits64(rng);
    return value % bound;
}

// Uniform integer in [low, high], both included.
template <class Int, class Engine>
Int UniformInt(Engine& rng, Int low, Int high) {
    static_assert(std::is_integral_v<Int>, "UniformInt draws integers");
    const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);
    const uint64_t offset = span == std::numeric_limits<uint64_t>::max() ? Bits64(rng) : Below(rng, span + 1);
    return static_cast<Int>(static_cast<uint64_t>(low) + offset);
}

// Fisher-Yates from the back: every order equally likely.
template <class RandomIt, class Engine>
void Shuffle(RandomIt first, RandomIt last, Engine& rng) {
    const auto n = static_cast<uint64_t>(std::distance(first, last));
    for (uint64_t i = n; i > 1; --i) {
        const uint64_t j = Below(rng, i);
        using std::swap;
        swap(first[static_cast<std::ptrdiff_t>(i - 1)], first[static_cast<std::ptrdiff_t>(j)]);
    }
}

// Uniform double in [0, 1) with 53 random bits.
template <class Engine>
double Uniform01(Engine& rng) {
    return static_cast<double>(Bits64(rng) >> 11) * 0x1.0p-53;
}

// Uniform double in [low, high).
template <class Engine>
double Uniform(Engine& rng, double low, double high) {
    const double value = low + (high - low) * Uniform01(rng);
    return value < high ? value : std::nextafter(high, low);
}

// Uniform float in [low, high).
template <class Engine>
float UniformFloat(Engine& rng, float low, float high) {
    const float value = static_cast<float>(low + (static_cast<double>(high) - low) * Uniform01(rng));
    return value < high ? value : std::nextafter(high, low);
}

// True with probability p.
template <class Engine>
bool Bernoulli(Engine& rng, double p) {
    return Uniform01(rng) < p;
}

// Normal(mean, sd) by Box-Muller: two uniform draws per value, no state kept.
template <class Engine>
double Normal(Engine& rng, double mean = 0.0, double sd = 1.0) {
    const double u1 = 1.0 - Uniform01(rng);  // (0, 1]
    const double u2 = Uniform01(rng);
    return mean + sd * std::sqrt(-2.0 * std::log(u1)) * std::cos(detail::kTwoPi * u2);
}

// Exponential with rate lambda (mean 1 / lambda).
template <class Engine>
double Exponential(Engine& rng, double lambda = 1.0) {
    return -std::log(1.0 - Uniform01(rng)) / lambda;
}

// Gamma(shape, scale) by Marsaglia and Tsang; a shape below 1 is boosted by
// one and scaled by U^(1 / shape).
template <class Engine>
double Gamma(Engine& rng, double shape, double scale = 1.0) {
    if (!(shape > 0.0)) return 0.0;
    if (shape < 1.0) {
        const double boosted = Gamma(rng, shape + 1.0, 1.0);
        return scale * boosted * std::pow(1.0 - Uniform01(rng), 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = Normal(rng);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = 1.0 - Uniform01(rng);
        if (u < 1.0 - 0.0331 * x * x * x * x) return scale * d * v;
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return scale * d * v;
    }
}

// Beta(a, b) from two gammas.
template <class Engine>
double Beta(Engine& rng, double a, double b) {
    const double x = Gamma(rng, a);
    const double y = Gamma(rng, b);
    return x + y > 0.0 ? x / (x + y) : 0.5;
}

// Binomial(n, p): inversion while n * min(p, 1 - p) is below 10, Hormann's
// BTRS (transformed rejection with squeeze) above.
template <class Engine>
uint64_t Binomial(Engine& rng, uint64_t n, double p) {
    if (n == 0 || !(p > 0.0)) return 0;
    if (p >= 1.0) return n;
    if (p > 0.5) return n - Binomial(rng, n, 1.0 - p);
    const double q = 1.0 - p;
    const double np = static_cast<double>(n) * p;
    if (np < 10.0) {
        const double s = p / q;
        const double a = (static_cast<double>(n) + 1.0) * s;
        for (;;) {
            double r = std::pow(q, static_cast<double>(n));
            double u = Uniform01(rng);
            uint64_t k = 0;
            bool done = true;
            while (u > r) {
                u -= r;
                ++k;
                if (k > n) {
                    done = false;  // rounding left u above the total: draw again
                    break;
                }
                r *= a / static_cast<double>(k) - s;
            }
            if (done) return k;
        }
    }
    const double nd = static_cast<double>(n);
    const double spq = std::sqrt(np * q);
    const double b = 1.15 + 2.53 * spq;
    const double a = -0.0873 + 0.0248 * b + 0.01 * p;
    const double c = np + 0.5;
    const double vr = 0.92 - 4.2 / b;
    const double alpha = (2.83 + 5.1 / b) * spq;
    const double lpq = std::log(p / q);
    const double m = std::floor((nd + 1.0) * p);
    const double h = std::lgamma(m + 1.0) + std::lgamma(nd - m + 1.0);
    for (;;) {
        const double u = Uniform01(rng) - 0.5;
        double v = Uniform01(rng);
        const double us = 0.5 - std::fabs(u);
        const double k = std::floor((2.0 * a / us + b) * u + c);
        if (k < 0.0 || k > nd) continue;
        if (us >= 0.07 && v <= vr) return static_cast<uint64_t>(k);
        v = std::log(v * alpha / (a / (us * us) + b));
        if (v <= h - std::lgamma(k + 1.0) - std::lgamma(nd - k + 1.0) + (k - m) * lpq) return static_cast<uint64_t>(k);
    }
}

// Poisson(mean): Knuth's product of uniforms below a mean of 10, Hormann's
// PTRS above.
template <class Engine>
uint64_t Poisson(Engine& rng, double mean) {
    if (!(mean > 0.0)) return 0;
    if (mean < 10.0) {
        const double limit = std::exp(-mean);
        uint64_t k = 0;
        double product = 1.0 - Uniform01(rng);
        while (product > limit) {
            ++k;
            product *= 1.0 - Uniform01(rng);
        }
        return k;
    }
    const double slam = std::sqrt(mean);
    const double loglam = std::log(mean);
    const double b = 0.931 + 2.53 * slam;
    const double a = -0.059 + 0.02483 * b;
    const double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    const double vr = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
        const double u = Uniform01(rng) - 0.5;
        const double v = Uniform01(rng);
        const double us = 0.5 - std::fabs(u);
        const double k = std::floor((2.0 * a / us + b) * u + mean + 0.43);
        if (us >= 0.07 && v <= vr) return static_cast<uint64_t>(k);
        if (k < 0.0 || (us < 0.013 && v > us)) continue;
        if (std::log(v) + std::log(invalpha) - std::log(a / (us * us) + b) <= -mean + k * loglam - std::lgamma(k + 1.0))
            return static_cast<uint64_t>(k);
    }
}

// An index drawn with probability weights[i] / sum(weights). Negative weights
// count as 0; returns weights.size() when no weight is above 0.
template <class Engine, class Weight>
size_t Discrete(Engine& rng, const std::vector<Weight>& weights) {
    double total = 0.0;
    for (const Weight w : weights)
        if (static_cast<double>(w) > 0.0) total += static_cast<double>(w);
    if (!(total > 0.0)) return weights.size();
    const double target = Uniform01(rng) * total;
    double running = 0.0;
    size_t last = weights.size();
    for (size_t i = 0; i < weights.size(); ++i) {
        const double w = static_cast<double>(weights[i]);
        if (!(w > 0.0)) continue;
        running += w;
        last = i;
        if (target < running) return i;
    }
    return last;  // rounding put the target at the total: the last positive weight
}

}  // namespace cyxwiz::portable
