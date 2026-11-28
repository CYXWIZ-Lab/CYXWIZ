#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_matmul.h"

#include <arrayfire.h>

#include <cmath>
#include <string>
#include <vector>

// DeviceMatMul (TOFIX242 2.2, ORA-14): ArrayFire's OpenCL backend reads the
// result buffer of a one-column product (its BLAS's matrix-vector routine
// computes beta * y with beta 0), so a buffer the memory pool hands out with
// NaN in it made the product NaN. Every product of every backend, after a NaN
// buffer of the result's size was released, must be finite and equal the
// product summed on the host.

namespace {

class ScopedBackendRestore {
public:
    ScopedBackendRestore() : backend_(af::getActiveBackend()), device_(af::getDevice()) {}
    ~ScopedBackendRestore() {
        try {
            af::setBackend(backend_);
            af::setDevice(device_);
        } catch (...) {
        }
    }

private:
    af::Backend backend_;
    int device_;
};

struct Case {
    dim_t rows, inner, cols;
    af::matProp lhs_op, rhs_op;
};

std::vector<float> Host(const af::array& a) {
    std::vector<float> out(static_cast<size_t>(a.elements()));
    a.host(out.data());
    return out;
}

}  // namespace

TEST_CASE("DeviceMatMul does not read a reused result buffer on any backend", "[arrayfire][matmul]") {
    const ScopedBackendRestore restore;
    const int available = af::getAvailableBackends();
    const std::vector<Case> cases = {
        {4, 8, 1, AF_MAT_NONE, AF_MAT_NONE},  {4, 8, 1, AF_MAT_NONE, AF_MAT_TRANS},   // a Dense with 1 unit
        {16, 8, 1, AF_MAT_TRANS, AF_MAT_NONE}, {1, 8, 1, AF_MAT_NONE, AF_MAT_NONE},
        {3, 5, 7, AF_MAT_NONE, AF_MAT_NONE},  {1, 6, 4, AF_MAT_NONE, AF_MAT_TRANS},
        {3, 5, 1, AF_MAT_TRANS, AF_MAT_TRANS},  // both sides transposed, one column
        {4, 1, 6, AF_MAT_NONE, AF_MAT_TRANS}};  // a one-column right side, transposed: 6 columns, not widened
    for (const af::Backend backend : {AF_BACKEND_CPU, AF_BACKEND_CUDA, AF_BACKEND_OPENCL}) {
        if ((available & static_cast<int>(backend)) == 0) continue;
        af::setBackend(backend);
        for (int device = 0; device < af::getDeviceCount(); ++device) {
            af::setDevice(device);
            for (const Case& c : cases) {
                DYNAMIC_SECTION("backend " << static_cast<int>(backend) << " device " << device << " " << c.rows
                                           << "x" << c.inner << " * " << c.inner << "x" << c.cols << " ops "
                                           << static_cast<int>(c.lhs_op) << "," << static_cast<int>(c.rhs_op)) {
                    const bool lhs_t = c.lhs_op != AF_MAT_NONE;
                    const bool rhs_t = c.rhs_op != AF_MAT_NONE;
                    const af::array lhs = lhs_t ? af::randu(c.inner, c.rows, f32) : af::randu(c.rows, c.inner, f32);
                    const af::array rhs = rhs_t ? af::randu(c.cols, c.inner, f32) : af::randu(c.inner, c.cols, f32);
                    lhs.eval();
                    rhs.eval();
                    {
                        af::array poison = af::constant(std::nanf(""), c.rows, c.cols, f32);
                        poison.eval();
                        af::sync();
                    }
                    const af::array product = cyxwiz::DeviceMatMul(lhs, rhs, c.lhs_op, c.rhs_op);
                    REQUIRE(product.dims(0) == c.rows);
                    REQUIRE(product.dims(1) == c.cols);
                    const auto p = Host(product);
                    const auto l = Host(lhs);
                    const auto r = Host(rhs);
                    for (dim_t i = 0; i < c.rows; ++i) {
                        for (dim_t j = 0; j < c.cols; ++j) {
                            double expected = 0.0;
                            for (dim_t k = 0; k < c.inner; ++k) {
                                const float a = lhs_t ? l[static_cast<size_t>(k + i * c.inner)]
                                                      : l[static_cast<size_t>(i + k * c.rows)];
                                const float b = rhs_t ? r[static_cast<size_t>(j + k * c.cols)]
                                                      : r[static_cast<size_t>(k + j * c.inner)];
                                expected += static_cast<double>(a) * b;
                            }
                            const float got = p[static_cast<size_t>(i + j * c.rows)];
                            INFO("[" << i << ", " << j << "] = " << got << ", expected " << expected);
                            REQUIRE(std::isfinite(got));
                            REQUIRE(std::abs(got - expected) <= 1e-4 * (1.0 + std::abs(expected)));
                        }
                    }
                }
            }
        }
    }
}
