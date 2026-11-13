// cyxwiz::portable draws: the same numbers on every compiler and OS, and the
// right distribution. The pinned values below were printed on Windows (MSVC)
// and pass unchanged with GCC on Linux; run with --print to print them again.
#include "cyxwiz/portable_random.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace pr = cyxwiz::portable;

int checks = 0;
bool print_pins = false;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void PinInts(const char* name, const std::vector<uint64_t>& got, const std::vector<uint64_t>& want) {
    if (print_pins) {
        std::printf("%s:", name);
        for (const uint64_t v : got) std::printf(" %llu,", static_cast<unsigned long long>(v));
        std::printf("\n");
        return;
    }
    Check(got == want, std::string(name) + ": the pinned draws");
}

void PinDoubles(const char* name, const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    if (print_pins) {
        std::printf("%s:", name);
        for (const double v : got) std::printf(" %.17g,", v);
        std::printf("\n");
        return;
    }
    Check(got.size() == want.size(), std::string(name) + ": count");
    for (size_t i = 0; i < got.size(); ++i)
        Check(tolerance == 0.0 ? got[i] == want[i] : std::abs(got[i] - want[i]) <= tolerance * (1.0 + std::abs(want[i])),
              std::string(name) + " [" + std::to_string(i) + "]: " + std::to_string(got[i]));
}

// Mean and variance of n draws within 5 standard errors of theory.
template <class Draw>
void Moments(const char* name, Draw draw, double mean, double variance, int n = 200000) {
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = draw();
        sum += v;
        sum_sq += v * v;
    }
    const double m = sum / n;
    const double var = sum_sq / n - m * m;
    Check(std::abs(m - mean) <= 5.0 * std::sqrt(variance / n), std::string(name) + ": mean " + std::to_string(m));
    // The variance of the sample variance needs the 4th moment; 2 % covers
    // every distribution here at this n.
    Check(std::abs(var - variance) <= 0.02 * variance + 1e-12, std::string(name) + ": variance " + std::to_string(var));
}

}  // namespace

int main(int argc, char** argv) {
    print_pins = argc > 1 && std::strcmp(argv[1], "--print") == 0;

    // The engines themselves: the 10,000th output is fixed by the standard.
    {
        std::mt19937 a;
        a.discard(9999);
        Check(a() == 4123659995u, "std::mt19937's 10,000th output");
        std::mt19937_64 b;
        b.discard(9999);
        Check(b() == 9981545732273789042ull, "std::mt19937_64's 10,000th output");
    }

    // ---- Pinned draws, seed 42
    {
        std::mt19937 r(42);
        std::vector<uint64_t> v;
        for (int i = 0; i < 8; ++i) v.push_back(pr::Bits64(r));
        PinInts("bits64 mt19937", v, {6909045637428952499ull, 17537583593393853710ull, 13502904847239337031ull, 11043299886329703444ull, 2878035897379592313ull, 2877591057541362902ull, 1071453510346823114ull, 15978129738215320948ull});
    }
    {
        std::mt19937 r32(42);
        std::mt19937_64 r64(42);
        std::vector<uint64_t> a, b;
        for (const uint64_t bound : {2ull, 3ull, 10ull, 1000ull, 4294967295ull, 4294967296ull, 1ull << 40, 18446744073709551557ull}) {
            a.push_back(pr::Below(r32, bound));
            b.push_back(pr::Below(r64, bound));
        }
        PinInts("below mt19937", a, {0ull, 2ull, 9ull, 183ull, 3143890025ull, 2571218620ull, 86569440870ull, 8224162778172047826ull});
        PinInts("below mt19937_64", b, {0ull, 2ull, 0ull, 662ull, 879934436ull, 2625589116ull, 109082584096ull, 6878563960102566144ull});
    }
    {
        std::mt19937 r(42);
        std::vector<uint64_t> order(12);
        std::iota(order.begin(), order.end(), 0);
        pr::Shuffle(order.begin(), order.end(), r);
        PinInts("shuffle 12", order, {11ull, 6ull, 10ull, 0ull, 2ull, 3ull, 7ull, 5ull, 1ull, 9ull, 8ull, 4ull});
        std::vector<uint64_t> ints;
        for (int i = 0; i < 6; ++i) ints.push_back(static_cast<uint64_t>(pr::UniformInt<int>(r, -5, 5) + 5));
        PinInts("uniform int -5..5 (+5)", ints, {1ull, 0ull, 5ull, 9ull, 3ull, 6ull});
    }
    {
        std::mt19937_64 r(42);
        std::vector<double> u, f;
        for (int i = 0; i < 6; ++i) u.push_back(pr::Uniform01(r));
        for (int i = 0; i < 6; ++i) f.push_back(pr::UniformFloat(r, -1.0f, 1.0f));
        PinDoubles("uniform01", u, {0.75515553295453897, 0.63903139385469743, 0.7521452007480266, 0.13627268363243705, 0.90326896642837828, 0.094068311762837031}, 0.0);
        PinDoubles("uniform float -1..1", f, {0.1491406112909317, -0.25422459840774536, -0.45225179195404053, -0.21945823729038239, -0.97523444890975952, 0.047411177307367325}, 0.0);
    }
    {
        // Values through std::log / std::exp / std::cos: equal to the last ulps.
        std::mt19937 r(42);
        std::vector<double> normal, expo, gamma_small, gamma_big, beta;
        for (int i = 0; i < 6; ++i) normal.push_back(pr::Normal(r));
        for (int i = 0; i < 4; ++i) expo.push_back(pr::Exponential(r, 2.0));
        for (int i = 0; i < 4; ++i) gamma_small.push_back(pr::Gamma(r, 0.4));
        for (int i = 0; i < 4; ++i) gamma_big.push_back(pr::Gamma(r, 3.5, 2.0));
        for (int i = 0; i < 4; ++i) beta.push_back(pr::Beta(r, 0.2, 0.2));
        PinDoubles("normal", normal, {0.92269958975747912, -1.320868982685788, 0.32436967755230517, 0.23068854957769014, -0.35304918699163018, 0.20032308317556141}, 1e-12);
        PinDoubles("exponential 2", expo, {0.89321475910521975, 0.1193438133598993, 0.10033949440918559, 0.10130571153845699}, 1e-12);
        PinDoubles("gamma 0.4", gamma_small, {0.17435133808693115, 0.72185399410966056, 0.21981154756828894, 1.8901764494996374}, 1e-12);
        PinDoubles("gamma 3.5 x2", gamma_big, {7.654472460807705, 4.3963958965235976, 2.5193967236525672, 11.329394068587542}, 1e-12);
        PinDoubles("beta 0.2 0.2", beta, {0.94411333930403096, 8.3384392950559922e-05, 0.99881572359457382, 0.99689017206768238}, 1e-12);
        std::vector<uint64_t> binom_small, binom_big, pois_small, pois_big, discrete;
        for (int i = 0; i < 6; ++i) binom_small.push_back(pr::Binomial(r, 20, 0.3));
        for (int i = 0; i < 6; ++i) binom_big.push_back(pr::Binomial(r, 1000000, 0.37));
        for (int i = 0; i < 6; ++i) pois_small.push_back(pr::Poisson(r, 3.5));
        for (int i = 0; i < 6; ++i) pois_big.push_back(pr::Poisson(r, 250.0));
        for (int i = 0; i < 8; ++i) discrete.push_back(pr::Discrete(r, std::vector<double>{0.1, 0.0, 0.6, 0.3}));
        PinInts("binomial 20 0.3", binom_small, {7ull, 3ull, 5ull, 4ull, 8ull, 7ull});
        PinInts("binomial 1e6 0.37", binom_big, {369762ull, 369732ull, 370332ull, 370681ull, 369343ull, 370385ull});
        PinInts("poisson 3.5", pois_small, {5ull, 4ull, 7ull, 2ull, 2ull, 2ull});
        PinInts("poisson 250", pois_big, {241ull, 237ull, 266ull, 246ull, 228ull, 283ull});
        PinInts("discrete", discrete, {2ull, 3ull, 2ull, 3ull, 3ull, 2ull, 2ull, 2ull});
    }
    if (print_pins) return 0;

    // ---- Ranges and edges
    {
        std::mt19937 r(7);
        for (int i = 0; i < 10000; ++i) {
            const double u = pr::Uniform01(r);
            Check(u >= 0.0 && u < 1.0, "uniform01 in [0, 1)");
            const float f = pr::UniformFloat(r, 0.0f, 1e-30f);
            Check(f >= 0.0f && f < 1e-30f, "a narrow float range stays below its top");
            Check(pr::Below(r, 7) < 7, "below 7");
        }
        Check(pr::Below(r, 1) == 0, "below 1 is 0");
        Check(pr::UniformInt<int64_t>(r, 5, 5) == 5, "a one-value range");
        const uint64_t full = pr::UniformInt<uint64_t>(r, 0, std::numeric_limits<uint64_t>::max());
        (void)full;  // the full 64-bit range draws without overflow
        Check(pr::Binomial(r, 0, 0.5) == 0 && pr::Binomial(r, 10, 0.0) == 0 && pr::Binomial(r, 10, 1.0) == 10, "binomial edges");
        Check(pr::Poisson(r, 0.0) == 0, "poisson of mean 0");
        Check(pr::Discrete(r, std::vector<float>{0.0f, 0.0f}) == 2, "no weight above 0");
        Check(pr::Discrete(r, std::vector<int>{0, 5, 0}) == 1, "one weight");
        std::vector<int> empty;
        pr::Shuffle(empty.begin(), empty.end(), r);
        std::vector<int> one{9};
        pr::Shuffle(one.begin(), one.end(), r);
        Check(one[0] == 9, "a one-item shuffle");
    }

    // ---- Distributions against theory
    {
        std::mt19937 r(11);
        Moments("uniform01", [&] { return pr::Uniform01(r); }, 0.5, 1.0 / 12.0);
        Moments("normal 3, 2", [&] { return pr::Normal(r, 3.0, 2.0); }, 3.0, 4.0);
        Moments("exponential 0.5", [&] { return pr::Exponential(r, 0.5); }, 2.0, 4.0);
        Moments("gamma 0.4", [&] { return pr::Gamma(r, 0.4); }, 0.4, 0.4);
        Moments("gamma 3.5 x2", [&] { return pr::Gamma(r, 3.5, 2.0); }, 7.0, 14.0);
        Moments("beta 2 5", [&] { return pr::Beta(r, 2.0, 5.0); }, 2.0 / 7.0, 10.0 / (49.0 * 8.0));
        Moments("binomial 20 0.3", [&] { return static_cast<double>(pr::Binomial(r, 20, 0.3)); }, 6.0, 4.2);
        Moments("binomial 500 0.8", [&] { return static_cast<double>(pr::Binomial(r, 500, 0.8)); }, 400.0, 80.0);
        Moments("binomial 1e6 0.37", [&] { return static_cast<double>(pr::Binomial(r, 1000000, 0.37)); }, 370000.0, 233100.0);
        Moments("poisson 3.5", [&] { return static_cast<double>(pr::Poisson(r, 3.5)); }, 3.5, 3.5);
        Moments("poisson 250", [&] { return static_cast<double>(pr::Poisson(r, 250.0)); }, 250.0, 250.0);
        Moments("uniform int 0..9", [&] { return static_cast<double>(pr::UniformInt<int>(r, 0, 9)); }, 4.5, 8.25);
        // A bound that is not a power of two: the low and high halves equally often.
        int64_t low_half = 0;
        const uint64_t bound = 3000000000ull;
        const int n = 200000;
        for (int i = 0; i < n; ++i) low_half += pr::Below(r, bound) < bound / 2 ? 1 : 0;
        Check(std::abs(low_half - n / 2) <= 5.0 * std::sqrt(n * 0.25), "below 3e9: no bias");
        // Discrete shares.
        std::array<int, 4> counts{};
        const std::vector<double> weights{0.1, 0.0, 0.6, 0.3};
        for (int i = 0; i < n; ++i) ++counts.at(pr::Discrete(r, weights));
        Check(counts[1] == 0, "a zero weight is never drawn");
        for (size_t i = 0; i < 4; ++i)
            Check(std::abs(counts[i] - n * weights[i]) <= 5.0 * std::sqrt(n * weights[i] * (1 - weights[i])) + 1e-9,
                  "discrete share " + std::to_string(i));
    }

    // ---- Shuffle: all 24 orders of 4 items about equally often (chi-square,
    // 23 degrees of freedom; 60 is far beyond the 0.9999 quantile, about 54).
    {
        std::mt19937_64 r(3);
        std::map<std::array<int, 4>, int> seen;
        const int n = 240000;
        for (int i = 0; i < n; ++i) {
            std::array<int, 4> a{0, 1, 2, 3};
            pr::Shuffle(a.begin(), a.end(), r);
            ++seen[a];
        }
        Check(seen.size() == 24, "every order of 4 appears");
        double chi = 0.0;
        for (const auto& [order, count] : seen) chi += (count - n / 24.0) * (count - n / 24.0) / (n / 24.0);
        Check(chi < 60.0, "shuffle orders are uniform: chi-square " + std::to_string(chi));
    }

    std::cout << "portable random: " << checks << " checks passed\n";
    return 0;
}
