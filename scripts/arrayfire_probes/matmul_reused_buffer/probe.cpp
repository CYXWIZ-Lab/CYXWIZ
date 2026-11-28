// Does af::matmul write NaN into its result when the buffer the memory pool
// hands it last held NaN? A GEMM that computes beta * C with beta = 0 instead of
// not reading C turns 0 * NaN into NaN. Seen as: on OpenCL, a model whose
// weights went NaN, freed, made the next model's first Dense output NaN.
//
// For each shape (the Dense layer's x [N, in] times W^T [in, out], and a plain
// product): fill a buffer of the result's size with NaN, release it, compute
// the product of finite inputs, and count non-finite values in the result.
#include <arrayfire.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Shape {
    dim_t rows, inner, cols;
};

int CountNonFinite(const af::array& a) {
    std::vector<float> host(static_cast<size_t>(a.elements()));
    a.host(host.data());
    int bad = 0;
    for (float v : host) bad += std::isfinite(v) ? 0 : 1;
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "opencl";
    const af::Backend backend = name == "cpu"    ? AF_BACKEND_CPU
                                : name == "cuda" ? AF_BACKEND_CUDA
                                : name == "oneapi" ? AF_BACKEND_ONEAPI
                                                   : AF_BACKEND_OPENCL;
    af::setBackend(backend);
    af::info();
    const std::vector<Shape> shapes = {{4, 8, 1}, {16, 8, 1}, {1, 8, 4}, {1, 64, 10}, {1, 8, 1}, {4, 1, 8}, {4, 18, 3}, {8, 64, 10}, {32, 128, 64}, {3, 5, 7}};
    int failures = 0;
    for (const Shape& s : shapes) {
        for (int transposed = 0; transposed < 2; ++transposed) {
            // The inputs first, so the freed NaN buffer is the next one handed out
            const af::array x = af::randu(s.rows, s.inner, f32);
            const af::array w = transposed ? af::randu(s.cols, s.inner, f32) : af::randu(s.inner, s.cols, f32);
            x.eval();
            w.eval();
            {
                // A pool buffer of the result's size, last holding NaN
                af::array poison = af::constant(std::nanf(""), s.rows, s.cols, f32);
                poison.eval();
                af::sync();
            }
            af::array product = transposed ? af::matmul(x, w, AF_MAT_NONE, AF_MAT_TRANS)  // Dense: x * W^T
                                           : af::matmul(x, w);
            product.eval();
            const int bad = CountNonFinite(product);
            std::printf("%s %lldx%lld * %lldx%lld%s: %d of %lld non-finite\n", bad ? "FAIL" : "ok  ",
                        static_cast<long long>(s.rows), static_cast<long long>(s.inner),
                        static_cast<long long>(s.inner), static_cast<long long>(s.cols),
                        transposed ? " (W^T)" : "", bad, static_cast<long long>(product.elements()));
            failures += bad ? 1 : 0;
        }
    }
    std::printf("%s: %d of %zu products wrote non-finite values\n", name.c_str(), failures, shapes.size() * 2);
    return failures ? 1 : 0;
}
