// Tensor linear algebra (LinearAlgebra::*(const Tensor&)) on the ArrayFire
// device (TOFIX140). Non-symmetric and non-square inputs, hand values: the
// device path used to read the raw storage (a square row-major A reads as
// A^T), so Solve solved A^T x = b and Transpose scrambled non-square input.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/linear_algebra.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <vector>

namespace {

cyxwiz::Tensor Matrix(const std::vector<size_t>& shape, const std::vector<float>& values) {
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void RequireValues(const cyxwiz::TensorResult& result, const std::vector<size_t>& shape,
                   const std::vector<float>& expected) {
    REQUIRE(result.success);
    REQUIRE(result.tensor.Shape() == shape);
    const float* data = result.tensor.ReadData<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(data[i] == Catch::Approx(expected[i]).margin(1e-5));
    }
}

}  // namespace

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor Solve handles a non-symmetric system", "[linear_algebra][tensor]") {
    const auto a = Matrix({2, 2}, {4, 2, 1, 3});  // 4x + 2y = 10, x + 3y = 5
    // A^T x = b would give (2.5, 0).
    RequireValues(cyxwiz::LinearAlgebra::Solve(a, Matrix({2}, {10, 5})), {2}, {2, 1});
    // Two right-hand sides: the second is 4x + 2y = 0, x + 3y = 10.
    RequireValues(cyxwiz::LinearAlgebra::Solve(a, Matrix({2, 2}, {10, 0, 5, 10})), {2, 2}, {2, -2, 1, 4});
}

TEST_CASE("Tensor Transpose, Inverse and Multiply keep row-major meaning", "[linear_algebra][tensor]") {
    const auto wide = Matrix({2, 3}, {1, 2, 3, 4, 5, 6});
    RequireValues(cyxwiz::LinearAlgebra::Transpose(wide), {3, 2}, {1, 4, 2, 5, 3, 6});
    RequireValues(cyxwiz::LinearAlgebra::Inverse(Matrix({2, 2}, {4, 2, 1, 3})), {2, 2},
                  {0.3f, -0.2f, -0.1f, 0.4f});
    RequireValues(cyxwiz::LinearAlgebra::Multiply(wide, Matrix({3, 2}, {1, 0, 0, 1, 1, 1})), {2, 2},
                  {4, 5, 10, 11});
}

TEST_CASE("Tensor LeastSquares and FrobeniusNorm", "[linear_algebra][tensor]") {
    // Overdetermined but consistent: x = (1, 2).
    RequireValues(cyxwiz::LinearAlgebra::LeastSquares(Matrix({3, 2}, {1, 0, 0, 1, 1, 1}), Matrix({3}, {1, 2, 3})),
                  {2}, {1, 2});
    const auto norm = cyxwiz::LinearAlgebra::FrobeniusNorm(Matrix({2, 2}, {1, 2, 3, 4}));
    REQUIRE(norm.success);
    CHECK(norm.value == Catch::Approx(std::sqrt(30.0)));
}
#endif
