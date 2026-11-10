// Seeded ArrayFire draws are the same on every device: Philox uniforms bit for
// bit, normals (DeviceNormal) to float rounding. The pins were printed on the
// dev PC's CPU backend and pass unchanged on CUDA, OpenCL and Linux CPU; run with
// --print to print them, and CYXWIZ_TEST_ARRAYFIRE_BACKEND=cpu|cuda|opencl to
// pick the device.
#include "test_device_selection.h"

#include "algorithms/arrayfire_backend_utils.h"
#include "algorithms/device_random.h"

#include <arrayfire.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int checks = 0;
bool print_pins = false;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

template <class T>
std::vector<T> Host(const af::array& values) {
    std::vector<T> host(static_cast<size_t>(values.elements()));
    values.host(host.data());
    return host;
}

// FNV-1a over the values' bytes: equal only when every value is bit-identical.
template <class T>
uint64_t Bits(const af::array& values) {
    const auto host = Host<T>(values);
    uint64_t h = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(host.data());
    for (size_t i = 0; i < host.size() * sizeof(T); ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
    return h;
}

void PinBits(const char* name, uint64_t got, uint64_t want) {
    if (print_pins) {
        std::printf("%s: 0x%016llxull\n", name, static_cast<unsigned long long>(got));
        return;
    }
    Check(got == want, std::string(name) + ": the pinned bits");
}

void PinValues(const char* name, const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    if (print_pins) {
        std::printf("%s:", name);
        for (const double v : got) std::printf(" %.17g,", v);
        std::printf("\n");
        return;
    }
    Check(got.size() == want.size(), std::string(name) + ": count");
    for (size_t i = 0; i < got.size(); ++i)
        Check(std::abs(got[i] - want[i]) <= tolerance * (1.0 + std::abs(want[i])),
              std::string(name) + " [" + std::to_string(i) + "]: " + std::to_string(got[i]));
}

}  // namespace

int main(int argc, char** argv) {
    print_pins = argc > 1 && std::strcmp(argv[1], "--print") == 0;
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");

    // Philox uniforms: the same bits on every device.
    {
        af::randomEngine engine = cyxwiz::DeviceRandomEngine(42);
        Check(engine.getType() == AF_RANDOM_ENGINE_PHILOX_4X32_10, "DeviceRandomEngine is Philox");
        PinBits("randu f32 65536", Bits<float>(af::randu(af::dim4(1 << 16), f32, engine)), 0xdbf38f9d7662b274ull);
        PinBits("randu f32 [37, 11] after", Bits<float>(af::randu(af::dim4(37, 11), f32, engine)),
                0x84092bc3b009da76ull);
    }

    // Normals from Philox uniforms: equal to float rounding on every device.
    {
        af::randomEngine engine = cyxwiz::DeviceRandomEngine(42);
        const auto first = Host<float>(cyxwiz::DeviceNormal(af::dim4(6), f32, engine));
        PinValues("normal f32", std::vector<double>(first.begin(), first.end()),
                  {0.15181569755077362, -0.098955929279327393, -0.59065139293670654, -0.081402339041233063,
                   -1.4785032272338867, -0.70978033542633057},
                  2e-6);
        const af::array many = cyxwiz::DeviceNormal(af::dim4(1 << 20), f32, engine);
        const double mean = af::mean<double>(many);
        const double var = af::var<double>(many, AF_VARIANCE_POPULATION);
        Check(std::abs(mean) < 5.0 / std::sqrt(double(1 << 20)), "normal mean " + std::to_string(mean));
        Check(std::abs(var - 1.0) < 0.01, "normal variance " + std::to_string(var));
        Check(af::allTrue<bool>(!af::isNaN(many)) && af::allTrue<bool>(!af::isInf(many)), "no NaN or infinity");
        const af::array halves = cyxwiz::DeviceNormal(af::dim4(4), f16, engine);
        Check(halves.type() == f16, "the asked type comes back");
    }
    if (cyxwiz::test::DeviceComputesInDouble()) {
        af::randomEngine engine = cyxwiz::DeviceRandomEngine(7);
        const auto first = Host<double>(cyxwiz::DeviceNormal(af::dim4(4), f64, engine));
        PinValues("normal f64", first,
                  {-0.0756799689405038, -0.70860916991006584, 1.0387690355889747, -0.17747920923658664}, 1e-12);
    }

    // The process's default engine is Philox after seeding, whatever it was.
    {
        af::setDefaultRandomEngineType(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
        cyxwiz::SeedCurrentArrayFireRandomEngine(42);
        Check(af::getDefaultRandomEngine().getType() == AF_RANDOM_ENGINE_PHILOX_4X32_10, "seeding makes the default Philox");
        const uint64_t once = Bits<float>(af::randu(af::dim4(4096), f32));
        cyxwiz::SeedCurrentArrayFireRandomEngine(42);
        Check(Bits<float>(af::randu(af::dim4(4096), f32)) == once, "a seed repeats on the default engine");
        cyxwiz::SeedCurrentArrayFireRandomEngine(42);
        af::randomEngine own = cyxwiz::DeviceRandomEngine(42);
        Check(Bits<float>(af::randu(af::dim4(4096), f32)) == Bits<float>(af::randu(af::dim4(4096), f32, own)),
              "the default engine draws as DeviceRandomEngine with the same seed");
    }

    if (!print_pins) std::cout << "device random: " << checks << " checks passed\n";
    return 0;
}
