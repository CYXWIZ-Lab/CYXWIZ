// Minimal reproducer for an ArrayFire oneAPI backend bug (found 2026-10-14 on an
// Intel Iris Xe, ArrayFire 3.10.0). Depends on ArrayFire only.
//
// af::max / af::min with indices along dim 1 skip part of each row when the
// array has few rows: on the [3, 5] matrix below, row 1 returns column 0
// (0.5535) and row 2 returns column 1 (0.3137), although columns 3 (0.7665) and
// 4 (0.9150) hold the maxima. The plain reductions (no indices) are right.
//
// Each result is compared with the same reduction along dim 0 of the transpose,
// which is right on every backend. Run:
//
//   oneapi_indexed_ops_probe [cpu|cuda|opencl|oneapi]
//
// Prints one line per check and exits 1 if any check is wrong.

#define NOMINMAX
#include <arrayfire.h>

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void Report(const char* check, bool ok) {
    std::printf("%-52s %s\n", check, ok ? "ok" : "WRONG");
    if (!ok) ++failures;
}

bool Same(const af::array& a, const af::array& b) {
    return !af::anyTrue<bool>(af::abs(a.as(f32) - b.as(f32)) > 1e-6f);
}

// A fixed [3, 5] matrix whose row maxima are known: columns 1, 3 and 4.
void FixedMatrix() {
    const float values[15] = {0.0555f, 0.5535f, 0.2385f,   // column 0
                              0.8969f, 0.3002f, 0.3137f,   // column 1
                              0.0944f, 0.0550f, 0.0022f,   // column 2
                              0.2531f, 0.7665f, 0.0864f,   // column 3
                              0.3456f, 0.1136f, 0.9150f};  // column 4 (column-major)
    const af::array x(3, 5, values);
    af::array peak, index;
    af::max(peak, index, x, 1);
    unsigned host[3];
    index.as(u32).host(host);
    std::printf("[3,5] argmax along dim 1: %u %u %u (expected 1 3 4)\n", host[0], host[1], host[2]);
    Report("af::max with indices, dim 1, fixed [3,5]", host[0] == 1 && host[1] == 3 && host[2] == 4);
}

// Random shapes: dim 1 against dim 0 of the transpose.
void RandomShapes() {
    const int shapes[][2] = {{3, 5}, {3, 4}, {8, 5}, {37, 100}, {5, 1000}, {100, 37}, {2000, 200}};
    for (const auto& shape : shapes) {
        af::setSeed(7);
        const af::array x = af::randn(shape[0], shape[1]);
        const af::array xt = x.T();
        af::array v1, i1, v0, i0;
        char label[96];

        af::max(v1, i1, x, 1);
        af::max(v0, i0, xt, 0);
        std::snprintf(label, sizeof(label), "af::max with indices, dim 1, randn [%d,%d]", shape[0], shape[1]);
        Report(label, Same(v1, v0.T()) && Same(i1, i0.T()));

        af::min(v1, i1, x, 1);
        af::min(v0, i0, xt, 0);
        std::snprintf(label, sizeof(label), "af::min with indices, dim 1, randn [%d,%d]", shape[0], shape[1]);
        Report(label, Same(v1, v0.T()) && Same(i1, i0.T()));

        // Control: the plain reduction along dim 1.
        std::snprintf(label, sizeof(label), "af::max without indices, dim 1, randn [%d,%d]", shape[0], shape[1]);
        Report(label, Same(af::max(x, 1), af::max(xt, 0).T()));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            const char* name = argv[1];
            if (!std::strcmp(name, "cpu")) af::setBackend(AF_BACKEND_CPU);
            else if (!std::strcmp(name, "cuda")) af::setBackend(AF_BACKEND_CUDA);
            else if (!std::strcmp(name, "opencl")) af::setBackend(AF_BACKEND_OPENCL);
            else if (!std::strcmp(name, "oneapi")) af::setBackend(AF_BACKEND_ONEAPI);
            else {
                std::fprintf(stderr, "usage: %s [cpu|cuda|opencl|oneapi]\n", argv[0]);
                return 2;
            }
        }
        af::info();
        FixedMatrix();
        RandomShapes();
    } catch (const af::exception& e) {
        std::fprintf(stderr, "ArrayFire error: %s\n", e.what());
        return 2;
    }
    std::printf("%s\n", failures == 0 ? "all checks ok" : "some checks WRONG");
    return failures == 0 ? 0 : 1;
}
