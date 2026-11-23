// Prints ArrayFire's first draws for a fixed seed, per random engine, so runs
// on different backends and operating systems can be compared line by line.
// A seeded ArrayFire draw is reproducible across machines only if these lines
// match.
#include <arrayfire.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void Print(const char* what, const af::array& values) {
    std::vector<float> host(static_cast<size_t>(values.elements()));
    values.host(host.data());
    std::printf("%-22s", what);
    for (const float v : host) std::printf(" %.7f", v);
    std::printf("\n");
}

// FNV-1a over the float bits: equal only when every value is bit-identical.
unsigned long long Bits(const af::array& values) {
    std::vector<float> host(static_cast<size_t>(values.elements()));
    values.host(host.data());
    unsigned long long h = 1469598103934665603ULL;
    for (const float v : host) {
        unsigned int u;
        std::memcpy(&u, &v, sizeof(u));
        for (int i = 0; i < 4; ++i) h = (h ^ ((u >> (8 * i)) & 0xFF)) * 1099511628211ULL;
    }
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    const char* backend = argc > 1 ? argv[1] : "cpu";
    af::Backend b = AF_BACKEND_CPU;
    if (!std::strcmp(backend, "cuda")) b = AF_BACKEND_CUDA;
    if (!std::strcmp(backend, "opencl")) b = AF_BACKEND_OPENCL;
    if (!std::strcmp(backend, "oneapi")) b = AF_BACKEND_ONEAPI;
    try {
        af::setBackend(b);
        std::printf("backend %s\n", backend);
        const struct {
            const char* name;
            af::randomEngineType type;
        } engines[] = {{"philox", AF_RANDOM_ENGINE_PHILOX_4X32_10},
                       {"threefry", AF_RANDOM_ENGINE_THREEFRY_2X32_16},
                       {"mersenne", AF_RANDOM_ENGINE_MERSENNE_GP11213}};
        for (const auto& e : engines) {
            af::randomEngine engine(e.type, 42);
            Print((std::string(e.name) + " randu").c_str(), af::randu(af::dim4(6), f32, engine));
            Print((std::string(e.name) + " randn").c_str(), af::randn(af::dim4(6), f32, engine));
            // A larger draw, then its first values: some backends split work by size.
            const af::array big = af::randu(af::dim4(1 << 16), f32, engine);
            Print((std::string(e.name) + " randu 65536").c_str(), big(af::seq(4)));
            std::printf("%-22s %.9f\n", (std::string(e.name) + " sum 65536").c_str(), af::sum<double>(big));
            std::printf("%-22s %016llx\n", (std::string(e.name) + " bits 65536").c_str(), Bits(big));
            const af::array normal = af::randn(af::dim4(1 << 16), f32, engine);
            std::printf("%-22s %016llx\n", (std::string(e.name) + " bits randn").c_str(), Bits(normal));
        }
    } catch (const af::exception& error) {
        std::printf("%s: %s\n", backend, error.what());
        return 1;
    }
    return 0;
}
