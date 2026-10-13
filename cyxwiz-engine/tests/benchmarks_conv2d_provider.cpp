// Conv2D forward + backward timing at the cats/dogs CNN geometries
// (TOFIX140 A1b). Run once as is (provider) and once with
// CYXWIZ_DISABLE_NEURAL_PROVIDERS=1 (the ArrayFire / native path) and compare.
// Not a ctest entry: a measurement tool.
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include "computation_truth/test_device_selection.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double TimeMs(cyxwiz::Conv2DModule& conv, const cyxwiz::Tensor& x, const cyxwiz::Tensor& dy, int iterations) {
    // Reading a result back waits for the device queue.
    const auto sync = [](const cyxwiz::Tensor& t) { volatile float v = t.ReadData<float>()[0]; (void)v; };
    // Warm-up (kernel compile, workspace)
    conv.Forward(x);
    sync(conv.Backward(dy));
    const auto start = std::chrono::steady_clock::now();
    cyxwiz::Tensor last;
    for (int i = 0; i < iterations; ++i) {
        conv.Forward(x);
        last = conv.Backward(dy);
    }
    sync(last);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

cyxwiz::Tensor Random(const std::vector<size_t>& shape, unsigned seed) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& f : v) f = dist(rng);
    return cyxwiz::Tensor(shape, v.data(), cyxwiz::DataType::Float32);
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return 1;
    struct Case { const char* name; size_t h, w, cin, cout, batch; int iterations; };
    const Case cases[] = {
        {"conv 3->16 @64x64, batch 32", 64, 64, 3, 16, 32, 10},
        {"conv 16->32 @32x32, batch 32", 32, 32, 16, 32, 32, 10},
        {"conv 32->64 @16x16, batch 64", 16, 16, 32, 64, 64, 10},
    };
    for (const auto& c : cases) {
        cyxwiz::Conv2DModule conv(static_cast<int>(c.cin), static_cast<int>(c.cout), 3, 1, 1, true);
        const auto x = Random({c.h, c.w, c.cin, c.batch}, 1);
        const auto dy = Random({c.h, c.w, c.cout, c.batch}, 2);
        const double ms = TimeMs(conv, x, dy, c.iterations);
        std::printf("%-32s %9.2f ms per forward+backward\n", c.name, ms);
    }
    return 0;
}
