#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/tensor.h>
#include <cyxwiz/memory_manager.h>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

TEST_CASE("Tensor creation", "[tensor]") {
    cyxwiz::Tensor t({2, 3}, cyxwiz::DataType::Float32);

    REQUIRE(t.NumDimensions() == 2);
    REQUIRE(t.NumElements() == 6);
    REQUIRE(t.GetDataType() == cyxwiz::DataType::Float32);
}

TEST_CASE("Tensor zeros", "[tensor]") {
    auto t = cyxwiz::Tensor::Zeros({4, 4});
    REQUIRE(t.NumElements() == 16);
}

TEST_CASE("Tensor random float factory preserves shape and range", "[tensor]") {
    auto t = cyxwiz::Tensor::Random({4, 4}, cyxwiz::DataType::Float32);

    REQUIRE(t.Shape() == std::vector<size_t>{4, 4});
    REQUIRE(t.GetDataType() == cyxwiz::DataType::Float32);

    const float* data = t.Data<float>();
    for (size_t i = 0; i < t.NumElements(); ++i) {
        REQUIRE(data[i] >= 0.0f);
        REQUIRE(data[i] <= 1.0f);
    }
}

TEST_CASE("Tensor RangeN fills sequential values", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3}, cyxwiz::DataType::Int32);

    REQUIRE(t.Shape() == std::vector<size_t>{2, 3});
    const int32_t* data = t.Data<int32_t>();
    for (int32_t i = 0; i < 6; i++) {
        REQUIRE(data[i] == i);
    }
}

TEST_CASE("Tensor reshape preserves row-major data", "[tensor]") {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    cyxwiz::Tensor t({2, 3}, data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor reshaped = t.Reshape({3, 2});

    REQUIRE(reshaped.Shape() == std::vector<size_t>{3, 2});
    const float* out = reshaped.Data<float>();
    for (size_t i = 0; i < 6; i++) {
        REQUIRE(out[i] == data[i]);
    }
}

TEST_CASE("Tensor transpose swaps 2D row-major axes", "[tensor]") {
    int32_t data[] = {
        1, 2, 3,
        4, 5, 6
    };
    cyxwiz::Tensor t({2, 3}, data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor transposed = t.Transpose();

    REQUIRE(transposed.Shape() == std::vector<size_t>{3, 2});
    const int32_t* out = transposed.Data<int32_t>();
    REQUIRE(out[0] == 1);
    REQUIRE(out[1] == 4);
    REQUIRE(out[2] == 2);
    REQUIRE(out[3] == 5);
    REQUIRE(out[4] == 3);
    REQUIRE(out[5] == 6);
}

TEST_CASE("Tensor view aliases reshape semantics", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3}, cyxwiz::DataType::Float32);

    cyxwiz::Tensor viewed = t.View({6});

    REQUIRE(viewed.Shape() == std::vector<size_t>{6});
    const float* out = viewed.Data<float>();
    for (size_t i = 0; i < 6; i++) {
        REQUIRE(out[i] == static_cast<float>(i));
    }
}

TEST_CASE("Tensor squeeze removes singleton dimensions", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({1, 2, 1, 3}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor squeezed = t.Squeeze();
    cyxwiz::Tensor dim_squeezed = t.Squeeze(2);

    REQUIRE(squeezed.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(dim_squeezed.Shape() == std::vector<size_t>{1, 2, 3});
    REQUIRE_THROWS_AS(t.Squeeze(1), std::runtime_error);
}

TEST_CASE("Tensor unsqueeze inserts singleton dimension", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor front = t.Unsqueeze(0);
    cyxwiz::Tensor middle = t.Unsqueeze(1);
    cyxwiz::Tensor back = t.Unsqueeze(-1);

    REQUIRE(front.Shape() == std::vector<size_t>{1, 2, 3});
    REQUIRE(middle.Shape() == std::vector<size_t>{2, 1, 3});
    REQUIRE(back.Shape() == std::vector<size_t>{2, 3, 1});
}

TEST_CASE("Tensor flatten collapses selected dimensions", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3, 4}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor all = t.Flatten();
    cyxwiz::Tensor tail = t.Flatten(1);
    cyxwiz::Tensor middle = t.Flatten(0, 1);

    REQUIRE(all.Shape() == std::vector<size_t>{24});
    REQUIRE(tail.Shape() == std::vector<size_t>{2, 12});
    REQUIRE(middle.Shape() == std::vector<size_t>{6, 4});
}

TEST_CASE("Tensor transpose swaps arbitrary dimensions", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3, 4}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor transposed = t.Transpose(0, 2);

    REQUIRE(transposed.Shape() == std::vector<size_t>{4, 3, 2});
    const int32_t* out = transposed.Data<int32_t>();
    REQUIRE(out[0] == 0);
    REQUIRE(out[1] == 12);
    REQUIRE(out[2] == 4);
    REQUIRE(out[3] == 16);
    REQUIRE(out[23] == 23);
}

TEST_CASE("Tensor permute reorders arbitrary dimensions", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3, 4}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor permuted = t.Permute({2, 0, 1});

    REQUIRE(permuted.Shape() == std::vector<size_t>{4, 2, 3});
    const int32_t* out = permuted.Data<int32_t>();
    REQUIRE(out[0] == 0);
    REQUIRE(out[1] == 4);
    REQUIRE(out[2] == 8);
    REQUIRE(out[3] == 12);
    REQUIRE(out[4] == 16);
    REQUIRE(out[5] == 20);
    REQUIRE(out[23] == 23);
}

TEST_CASE("Tensor permute validates dimensions and preserves dtype", "[tensor]") {
    double data[] = {1.0, 2.0, 3.0, 4.0};
    cyxwiz::Tensor t({1, 2, 2}, data, cyxwiz::DataType::Float64);

    cyxwiz::Tensor permuted = t.Permute({-1, 1, 0});

    REQUIRE(permuted.Shape() == std::vector<size_t>{2, 2, 1});
    REQUIRE(permuted.GetDataType() == cyxwiz::DataType::Float64);
    REQUIRE(permuted.Data<double>()[0] == 1.0);
    REQUIRE(permuted.Data<double>()[1] == 3.0);
    REQUIRE(permuted.Data<double>()[2] == 2.0);
    REQUIRE(permuted.Data<double>()[3] == 4.0);
    REQUIRE_THROWS_AS(t.Permute({0, 1}), std::runtime_error);
    REQUIRE_THROWS_AS(t.Permute({0, 0, 1}), std::runtime_error);
    REQUIRE_THROWS_AS(t.Permute({0, 1, 3}), std::runtime_error);
}

TEST_CASE("Tensor scalar reductions work for float tensors", "[tensor]") {
    float data[] = {1.0f, -2.0f, 3.0f, 4.0f};
    cyxwiz::Tensor t({2, 2}, data, cyxwiz::DataType::Float32);

    REQUIRE(t.Sum().Data<float>()[0] == 6.0f);
    REQUIRE(t.Mean().Data<float>()[0] == 1.5f);
    REQUIRE(t.Max().Data<float>()[0] == 4.0f);
    REQUIRE(t.Min().Data<float>()[0] == -2.0f);
    REQUIRE(t.Prod().Data<float>()[0] == -24.0f);
    REQUIRE(t.Var().Data<float>()[0] == Catch::Approx(5.25f));
    REQUIRE(t.Std().Data<float>()[0] == Catch::Approx(std::sqrt(5.25f)));
}

TEST_CASE("Tensor scalar reductions work for integer tensors", "[tensor]") {
    int32_t data[] = {1, 2, 3, 4};
    cyxwiz::Tensor t({4}, data, cyxwiz::DataType::Int32);

    REQUIRE(t.Sum().Data<int32_t>()[0] == 10);
    REQUIRE(t.Mean().GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(t.Mean().Data<float>()[0] == 2.5f);
    REQUIRE(t.Max().Data<int32_t>()[0] == 4);
    REQUIRE(t.Min().Data<int32_t>()[0] == 1);
    REQUIRE(t.Prod().Data<int32_t>()[0] == 24);
    REQUIRE(t.Var().GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(t.Var().Data<float>()[0] == Catch::Approx(1.25f));
    REQUIRE(t.Std().Data<float>()[0] == Catch::Approx(std::sqrt(1.25f)));
}

TEST_CASE("Tensor scalar variance preserves Float64 output", "[tensor]") {
    double data[] = {1.0, 2.0, 3.0};
    cyxwiz::Tensor t({3}, data, cyxwiz::DataType::Float64);

    cyxwiz::Tensor variance = t.Var();
    cyxwiz::Tensor stddev = t.Std();

    REQUIRE(variance.GetDataType() == cyxwiz::DataType::Float64);
    REQUIRE(stddev.GetDataType() == cyxwiz::DataType::Float64);
    REQUIRE(variance.Data<double>()[0] == Catch::Approx(2.0 / 3.0));
    REQUIRE(stddev.Data<double>()[0] == Catch::Approx(std::sqrt(2.0 / 3.0)));
}

TEST_CASE("Tensor scalar variance rejects empty tensors", "[tensor]") {
    cyxwiz::Tensor t({0}, cyxwiz::DataType::Float32);

    REQUIRE_THROWS_AS(t.Var(), std::runtime_error);
    REQUIRE_THROWS_AS(t.Std(), std::runtime_error);
}

TEST_CASE("Tensor dimension reductions preserve integer dtype and shape", "[tensor]") {
    int32_t data[] = {
        1, 2, 3,
        4, 5, 6
    };
    cyxwiz::Tensor t({2, 3}, data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor col_sum = t.Sum(0);
    cyxwiz::Tensor row_sum = t.Sum(1, true);
    cyxwiz::Tensor col_prod = t.Prod(0);
    cyxwiz::Tensor col_max = t.Max(0);
    cyxwiz::Tensor row_min = t.Min(1);

    REQUIRE(col_sum.GetDataType() == cyxwiz::DataType::Int32);
    REQUIRE(col_sum.Shape() == std::vector<size_t>{3});
    REQUIRE(col_sum.Data<int32_t>()[0] == 5);
    REQUIRE(col_sum.Data<int32_t>()[1] == 7);
    REQUIRE(col_sum.Data<int32_t>()[2] == 9);

    REQUIRE(row_sum.Shape() == std::vector<size_t>{2, 1});
    REQUIRE(row_sum.Data<int32_t>()[0] == 6);
    REQUIRE(row_sum.Data<int32_t>()[1] == 15);

    REQUIRE(col_prod.Data<int32_t>()[0] == 4);
    REQUIRE(col_prod.Data<int32_t>()[1] == 10);
    REQUIRE(col_prod.Data<int32_t>()[2] == 18);
    REQUIRE(col_max.Data<int32_t>()[0] == 4);
    REQUIRE(col_max.Data<int32_t>()[1] == 5);
    REQUIRE(col_max.Data<int32_t>()[2] == 6);
    REQUIRE(row_min.Data<int32_t>()[0] == 1);
    REQUIRE(row_min.Data<int32_t>()[1] == 4);
}

TEST_CASE("Tensor dimension max and min seed from data values", "[tensor]") {
    int32_t data[] = {
        -5, -2,
        -3, -4
    };
    cyxwiz::Tensor t({2, 2}, data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor row_max = t.Max(1);
    cyxwiz::Tensor col_min = t.Min(0);

    REQUIRE(row_max.Shape() == std::vector<size_t>{2});
    REQUIRE(row_max.Data<int32_t>()[0] == -2);
    REQUIRE(row_max.Data<int32_t>()[1] == -3);
    REQUIRE(col_min.Data<int32_t>()[0] == -5);
    REQUIRE(col_min.Data<int32_t>()[1] == -4);
}

TEST_CASE("Tensor dimension statistical reductions return real tensors", "[tensor]") {
    double data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    cyxwiz::Tensor t({2, 3}, data, cyxwiz::DataType::Float64);

    cyxwiz::Tensor col_mean = t.Mean(0);
    cyxwiz::Tensor row_mean = t.Mean(1, true);
    cyxwiz::Tensor row_var = t.Var(1);
    cyxwiz::Tensor row_std = t.Std(1);

    REQUIRE(col_mean.GetDataType() == cyxwiz::DataType::Float64);
    REQUIRE(col_mean.Shape() == std::vector<size_t>{3});
    REQUIRE(col_mean.Data<double>()[0] == Catch::Approx(2.5));
    REQUIRE(col_mean.Data<double>()[1] == Catch::Approx(3.5));
    REQUIRE(col_mean.Data<double>()[2] == Catch::Approx(4.5));

    REQUIRE(row_mean.Shape() == std::vector<size_t>{2, 1});
    REQUIRE(row_mean.Data<double>()[0] == Catch::Approx(2.0));
    REQUIRE(row_mean.Data<double>()[1] == Catch::Approx(5.0));
    REQUIRE(row_var.Data<double>()[0] == Catch::Approx(2.0 / 3.0));
    REQUIRE(row_var.Data<double>()[1] == Catch::Approx(2.0 / 3.0));
    REQUIRE(row_std.Data<double>()[0] == Catch::Approx(std::sqrt(2.0 / 3.0)));
    REQUIRE(row_std.Data<double>()[1] == Catch::Approx(std::sqrt(2.0 / 3.0)));
}

TEST_CASE("Tensor dimension reductions validate dimensions and empty inputs", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({2, 3}, cyxwiz::DataType::Float32);
    cyxwiz::Tensor empty({0, 3}, cyxwiz::DataType::Float32);

    cyxwiz::Tensor sum = empty.Sum(0);
    cyxwiz::Tensor prod = empty.Prod(0);

    REQUIRE(sum.Shape() == std::vector<size_t>{3});
    REQUIRE(sum.Data<float>()[0] == 0.0f);
    REQUIRE(sum.Data<float>()[1] == 0.0f);
    REQUIRE(sum.Data<float>()[2] == 0.0f);
    REQUIRE(prod.Data<float>()[0] == 1.0f);
    REQUIRE(prod.Data<float>()[1] == 1.0f);
    REQUIRE(prod.Data<float>()[2] == 1.0f);

    REQUIRE_THROWS_AS(t.Mean(2), std::runtime_error);
    REQUIRE_THROWS_AS(empty.Max(0), std::runtime_error);
    REQUIRE_THROWS_AS(empty.Min(0), std::runtime_error);
    REQUIRE_THROWS_AS(empty.Mean(0), std::runtime_error);
    REQUIRE_THROWS_AS(empty.Var(0), std::runtime_error);
    REQUIRE_THROWS_AS(empty.Std(0), std::runtime_error);
}

TEST_CASE("Tensor dot computes 1D inner products", "[tensor]") {
    int32_t left_data[] = {1, 2, 3};
    int32_t right_data[] = {4, 5, 6};
    cyxwiz::Tensor left({3}, left_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor right({3}, right_data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor result = left.Dot(right);

    REQUIRE(result.Shape() == std::vector<size_t>{1});
    REQUIRE(result.GetDataType() == cyxwiz::DataType::Int32);
    REQUIRE(result.Data<int32_t>()[0] == 32);
}

TEST_CASE("Tensor dot computes row-wise 2D inner products", "[tensor]") {
    float left_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float right_data[] = {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f
    };
    cyxwiz::Tensor left({2, 3}, left_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor right({2, 3}, right_data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor result = left.Dot(right);

    REQUIRE(result.Shape() == std::vector<size_t>{2, 1});
    REQUIRE(result.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(result.Data<float>()[0] == Catch::Approx(6.0f));
    REQUIRE(result.Data<float>()[1] == Catch::Approx(30.0f));
}

TEST_CASE("Tensor dot validates rank size and dtype", "[tensor]") {
    float left_data[] = {1.0f, 2.0f};
    float right_data[] = {3.0f, 4.0f, 5.0f};
    int32_t int_data[] = {1, 2};
    cyxwiz::Tensor left({2}, left_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor right({3}, right_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor matrix({1, 2}, left_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor int_tensor({2}, int_data, cyxwiz::DataType::Int32);

    REQUIRE_THROWS_AS(left.Dot(right), std::runtime_error);
    REQUIRE_THROWS_AS(left.Dot(matrix), std::runtime_error);
    REQUIRE_THROWS_AS(left.Dot(int_tensor), std::runtime_error);
}

TEST_CASE("Tensor batch matmul computes row-major 3D batches", "[tensor]") {
    float left_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    float right_data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    cyxwiz::Tensor left({2, 2, 3}, left_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor right({2, 3, 2}, right_data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor result = left.BatchMatMul(right);

    REQUIRE(result.Shape() == std::vector<size_t>{2, 2, 2});
    REQUIRE(result.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(result.Data<float>()[0] == Catch::Approx(22.0f));
    REQUIRE(result.Data<float>()[1] == Catch::Approx(28.0f));
    REQUIRE(result.Data<float>()[2] == Catch::Approx(49.0f));
    REQUIRE(result.Data<float>()[3] == Catch::Approx(64.0f));
    REQUIRE(result.Data<float>()[4] == Catch::Approx(220.0f));
    REQUIRE(result.Data<float>()[5] == Catch::Approx(244.0f));
    REQUIRE(result.Data<float>()[6] == Catch::Approx(301.0f));
    REQUIRE(result.Data<float>()[7] == Catch::Approx(334.0f));
}

TEST_CASE("Tensor batch matmul preserves dtype and handles empty batches", "[tensor]") {
    int64_t left_data[] = {1, 2, 3, 4};
    int64_t right_data[] = {5, 6, 7, 8};
    cyxwiz::Tensor left({1, 2, 2}, left_data, cyxwiz::DataType::Int64);
    cyxwiz::Tensor right({1, 2, 2}, right_data, cyxwiz::DataType::Int64);
    cyxwiz::Tensor empty_left({0, 2, 3}, cyxwiz::DataType::Float32);
    cyxwiz::Tensor empty_right({0, 3, 4}, cyxwiz::DataType::Float32);

    cyxwiz::Tensor result = left.BatchMatMul(right);
    cyxwiz::Tensor empty = empty_left.BatchMatMul(empty_right);

    REQUIRE(result.GetDataType() == cyxwiz::DataType::Int64);
    REQUIRE(result.Shape() == std::vector<size_t>{1, 2, 2});
    REQUIRE(result.Data<int64_t>()[0] == 19);
    REQUIRE(result.Data<int64_t>()[1] == 22);
    REQUIRE(result.Data<int64_t>()[2] == 43);
    REQUIRE(result.Data<int64_t>()[3] == 50);
    REQUIRE(empty.Shape() == std::vector<size_t>{0, 2, 4});
    REQUIRE(empty.NumElements() == 0);
}

TEST_CASE("Tensor batch matmul validates rank shape and dtype", "[tensor]") {
    float left_data[] = {1.0f, 2.0f};
    float shape_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    double right_data[] = {1.0, 2.0};
    cyxwiz::Tensor left({1, 1, 2}, left_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor bad_rank({2}, left_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor bad_batch({2, 2, 1}, shape_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor bad_inner({1, 3, 1}, shape_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor bad_dtype({1, 2, 1}, right_data, cyxwiz::DataType::Float64);

    REQUIRE_THROWS_AS(left.BatchMatMul(bad_rank), std::runtime_error);
    REQUIRE_THROWS_AS(left.BatchMatMul(bad_batch), std::runtime_error);
    REQUIRE_THROWS_AS(left.BatchMatMul(bad_inner), std::runtime_error);
    REQUIRE_THROWS_AS(left.BatchMatMul(bad_dtype), std::runtime_error);
}

TEST_CASE("Tensor scalar arithmetic preserves tensor dtype", "[tensor]") {
    float float_data[] = {1.0f, -2.0f, 3.0f};
    cyxwiz::Tensor f({3}, float_data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor added = f + 2.0f;
    cyxwiz::Tensor multiplied = f * -3.0f;
    cyxwiz::Tensor divided = f / 2.0f;

    REQUIRE(added.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(added.Data<float>()[0] == 3.0f);
    REQUIRE(added.Data<float>()[1] == 0.0f);
    REQUIRE(multiplied.Data<float>()[2] == -9.0f);
    REQUIRE(divided.Data<float>()[1] == -1.0f);

    int32_t int_data[] = {5, 6};
    cyxwiz::Tensor i({2}, int_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor int_divided = i / 2.0f;

    REQUIRE(int_divided.GetDataType() == cyxwiz::DataType::Int32);
    REQUIRE(int_divided.Data<int32_t>()[0] == 2);
    REQUIRE(int_divided.Data<int32_t>()[1] == 3);
    REQUIRE_THROWS_AS(i / 0.0f, std::runtime_error);
}

TEST_CASE("Tensor unary elementwise math returns expected values", "[tensor]") {
    float data[] = {1.0f, 4.0f, 9.0f};
    cyxwiz::Tensor t({3}, data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor sqrt = t.Sqrt();
    cyxwiz::Tensor pow = t.Pow(2.0f);
    cyxwiz::Tensor log = t.Log();

    REQUIRE(sqrt.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(sqrt.Data<float>()[0] == 1.0f);
    REQUIRE(sqrt.Data<float>()[1] == 2.0f);
    REQUIRE(sqrt.Data<float>()[2] == 3.0f);
    REQUIRE(pow.Data<float>()[2] == 81.0f);
    REQUIRE(log.Data<float>()[1] == Catch::Approx(std::log(4.0f)));
}

TEST_CASE("Tensor elementwise abs sign clip and negate work", "[tensor]") {
    int32_t data[] = {-3, 0, 7};
    cyxwiz::Tensor t({3}, data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor abs = t.Abs();
    cyxwiz::Tensor sign = t.Sign();
    cyxwiz::Tensor clipped = t.Clip(-1.0f, 5.0f);
    cyxwiz::Tensor negated = -t;

    REQUIRE(abs.GetDataType() == cyxwiz::DataType::Int32);
    REQUIRE(abs.Data<int32_t>()[0] == 3);
    REQUIRE(abs.Data<int32_t>()[2] == 7);
    REQUIRE(sign.Data<int32_t>()[0] == -1);
    REQUIRE(sign.Data<int32_t>()[1] == 0);
    REQUIRE(sign.Data<int32_t>()[2] == 1);
    REQUIRE(clipped.Data<int32_t>()[0] == -1);
    REQUIRE(clipped.Data<int32_t>()[2] == 5);
    REQUIRE(negated.Data<int32_t>()[0] == 3);
    REQUIRE(negated.Data<int32_t>()[2] == -7);
    REQUIRE_THROWS_AS(t.Clip(2.0f, 1.0f), std::runtime_error);
}

TEST_CASE("Tensor comparison operators return UInt8 masks", "[tensor]") {
    int32_t left_data[] = {1, 2, 3, 4};
    int32_t right_data[] = {1, 3, 2, 4};
    cyxwiz::Tensor left({2, 2}, left_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor right({2, 2}, right_data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor gt = left > right;
    cyxwiz::Tensor ge = left >= right;
    cyxwiz::Tensor lt = left < right;
    cyxwiz::Tensor eq = left == right;
    cyxwiz::Tensor ne = left != right;

    REQUIRE(gt.GetDataType() == cyxwiz::DataType::UInt8);
    REQUIRE(gt.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(gt.Data<uint8_t>()[0] == 0);
    REQUIRE(gt.Data<uint8_t>()[2] == 1);
    REQUIRE(ge.Data<uint8_t>()[0] == 1);
    REQUIRE(ge.Data<uint8_t>()[1] == 0);
    REQUIRE(lt.Data<uint8_t>()[1] == 1);
    REQUIRE(eq.Data<uint8_t>()[0] == 1);
    REQUIRE(eq.Data<uint8_t>()[2] == 0);
    REQUIRE(ne.Data<uint8_t>()[1] == 1);
    REQUIRE(ne.Data<uint8_t>()[3] == 0);
}

TEST_CASE("Tensor scalar comparisons preserve shape", "[tensor]") {
    float data[] = {1.0f, 2.5f, 3.0f};
    cyxwiz::Tensor t({3}, data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor gt = t > 2.0f;
    cyxwiz::Tensor le = t <= 2.5f;
    cyxwiz::Tensor eq = t == 2.5f;

    REQUIRE(gt.Shape() == std::vector<size_t>{3});
    REQUIRE(gt.GetDataType() == cyxwiz::DataType::UInt8);
    REQUIRE(gt.Data<uint8_t>()[0] == 0);
    REQUIRE(gt.Data<uint8_t>()[1] == 1);
    REQUIRE(gt.Data<uint8_t>()[2] == 1);
    REQUIRE(le.Data<uint8_t>()[0] == 1);
    REQUIRE(le.Data<uint8_t>()[1] == 1);
    REQUIRE(le.Data<uint8_t>()[2] == 0);
    REQUIRE(eq.Data<uint8_t>()[1] == 1);
}

TEST_CASE("Tensor comparisons broadcast and support mixed dtypes", "[tensor]") {
    int32_t left_data[] = {
        1, 2, 3,
        4, 5, 6
    };
    double right_data[] = {2.0, 5.0, 4.0};
    cyxwiz::Tensor left({2, 3}, left_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor right({3}, right_data, cyxwiz::DataType::Float64);

    cyxwiz::Tensor gt = left > right;
    cyxwiz::Tensor le = left <= right;

    REQUIRE(gt.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(gt.GetDataType() == cyxwiz::DataType::UInt8);
    REQUIRE(gt.Data<uint8_t>()[0] == 0);
    REQUIRE(gt.Data<uint8_t>()[2] == 0);
    REQUIRE(gt.Data<uint8_t>()[3] == 1);
    REQUIRE(gt.Data<uint8_t>()[5] == 1);
    REQUIRE(le.Data<uint8_t>()[0] == 1);
    REQUIRE(le.Data<uint8_t>()[4] == 1);
    REQUIRE(le.Data<uint8_t>()[5] == 0);
    REQUIRE_THROWS_AS(left < cyxwiz::Tensor::Ones({2}, cyxwiz::DataType::Float32), std::runtime_error);
}

TEST_CASE("Tensor logical operators return UInt8 masks", "[tensor]") {
    uint8_t left_data[] = {1, 0, 1, 0};
    uint8_t right_data[] = {1, 1, 0, 0};
    cyxwiz::Tensor left({2, 2}, left_data, cyxwiz::DataType::UInt8);
    cyxwiz::Tensor right({2, 2}, right_data, cyxwiz::DataType::UInt8);

    cyxwiz::Tensor both = left && right;
    cyxwiz::Tensor either = left || right;
    cyxwiz::Tensor inverted = !left;

    REQUIRE(both.GetDataType() == cyxwiz::DataType::UInt8);
    REQUIRE(both.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(both.Data<uint8_t>()[0] == 1);
    REQUIRE(both.Data<uint8_t>()[1] == 0);
    REQUIRE(both.Data<uint8_t>()[2] == 0);
    REQUIRE(both.Data<uint8_t>()[3] == 0);
    REQUIRE(either.Data<uint8_t>()[0] == 1);
    REQUIRE(either.Data<uint8_t>()[1] == 1);
    REQUIRE(either.Data<uint8_t>()[2] == 1);
    REQUIRE(either.Data<uint8_t>()[3] == 0);
    REQUIRE(inverted.Data<uint8_t>()[0] == 0);
    REQUIRE(inverted.Data<uint8_t>()[1] == 1);
}

TEST_CASE("Tensor logical operators broadcast and support mixed dtypes", "[tensor]") {
    int32_t left_data[] = {
        0, 2, -3,
        4, 0, 6
    };
    double right_data[] = {1.0, 0.0, 2.0};
    cyxwiz::Tensor left({2, 3}, left_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor right({3}, right_data, cyxwiz::DataType::Float64);

    cyxwiz::Tensor both = left && right;
    cyxwiz::Tensor either = left || right;
    cyxwiz::Tensor inverted = !right;

    REQUIRE(both.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(both.GetDataType() == cyxwiz::DataType::UInt8);
    REQUIRE(both.Data<uint8_t>()[0] == 0);
    REQUIRE(both.Data<uint8_t>()[1] == 0);
    REQUIRE(both.Data<uint8_t>()[2] == 1);
    REQUIRE(both.Data<uint8_t>()[3] == 1);
    REQUIRE(both.Data<uint8_t>()[4] == 0);
    REQUIRE(both.Data<uint8_t>()[5] == 1);
    REQUIRE(either.Data<uint8_t>()[0] == 1);
    REQUIRE(either.Data<uint8_t>()[1] == 1);
    REQUIRE(either.Data<uint8_t>()[4] == 0);
    REQUIRE(inverted.Shape() == std::vector<size_t>{3});
    REQUIRE(inverted.Data<uint8_t>()[0] == 0);
    REQUIRE(inverted.Data<uint8_t>()[1] == 1);
    REQUIRE(inverted.Data<uint8_t>()[2] == 0);
    REQUIRE_THROWS_AS(left && cyxwiz::Tensor::Ones({2}, cyxwiz::DataType::Float32), std::runtime_error);
}

TEST_CASE("Tensor tensor exponent pow supports dtype promotion", "[tensor]") {
    int32_t base_data[] = {2, 3, 4};
    float exp_data[] = {3.0f, 2.0f, 0.5f};
    cyxwiz::Tensor base({3}, base_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor exponent({3}, exp_data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor result = base.Pow(exponent);

    REQUIRE(result.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(result.Data<float>()[0] == 8.0f);
    REQUIRE(result.Data<float>()[1] == 9.0f);
    REQUIRE(result.Data<float>()[2] == 2.0f);
    REQUIRE_THROWS_AS(base.Pow(cyxwiz::Tensor::Ones({2})), std::runtime_error);
}

TEST_CASE("Tensor broadcast shape helpers follow NumPy-style rules", "[tensor]") {
    REQUIRE(cyxwiz::Tensor::IsBroadcastable({2, 3, 4}, {3, 1}));
    REQUIRE(cyxwiz::Tensor::IsBroadcastable({5, 1}, {1, 7}));
    REQUIRE_FALSE(cyxwiz::Tensor::IsBroadcastable({2, 3}, {4, 3}));

    REQUIRE(cyxwiz::Tensor::BroadcastShape({2, 3, 4}, {3, 1}) ==
            std::vector<size_t>{2, 3, 4});
    REQUIRE(cyxwiz::Tensor::BroadcastShape({5, 1}, {1, 7}) ==
            std::vector<size_t>{5, 7});
    REQUIRE_THROWS_AS(cyxwiz::Tensor::BroadcastShape({2, 3}, {4, 3}), std::runtime_error);
}

TEST_CASE("Tensor expand materializes broadcasted float data", "[tensor]") {
    float data[] = {10.0f, 20.0f, 30.0f};
    cyxwiz::Tensor t({3}, data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor expanded = t.Expand({2, 3});

    REQUIRE(expanded.Shape() == std::vector<size_t>{2, 3});
    const float* out = expanded.Data<float>();
    REQUIRE(out[0] == 10.0f);
    REQUIRE(out[1] == 20.0f);
    REQUIRE(out[2] == 30.0f);
    REQUIRE(out[3] == 10.0f);
    REQUIRE(out[4] == 20.0f);
    REQUIRE(out[5] == 30.0f);
}

TEST_CASE("Tensor broadcast to repeats singleton dimensions for integer tensors", "[tensor]") {
    int32_t data[] = {1, 2};
    cyxwiz::Tensor t({2, 1}, data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor expanded = t.BroadcastTo({2, 3});

    REQUIRE(expanded.Shape() == std::vector<size_t>{2, 3});
    const int32_t* out = expanded.Data<int32_t>();
    REQUIRE(out[0] == 1);
    REQUIRE(out[1] == 1);
    REQUIRE(out[2] == 1);
    REQUIRE(out[3] == 2);
    REQUIRE(out[4] == 2);
    REQUIRE(out[5] == 2);
    REQUIRE_THROWS_AS(t.Expand({3, 2}), std::runtime_error);
    REQUIRE_THROWS_AS(t.BroadcastTo({1, 3}), std::runtime_error);
}

TEST_CASE("Tensor At and Set provide bounds-checked element access", "[tensor]") {
    float data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    cyxwiz::Tensor t({2, 3}, data, cyxwiz::DataType::Float32);

    REQUIRE(t.At(0) == 1.0f);
    REQUIRE(t.At(1, 2) == 6.0f);

    t.Set(1, 1, 42.5f);
    REQUIRE(t.At(4) == 42.5f);
    REQUIRE(t.Data<float>()[4] == 42.5f);

    REQUIRE_THROWS_AS(t.At(2, 0), std::out_of_range);
    REQUIRE_THROWS_AS(t.Set(6, 1.0f), std::out_of_range);
    REQUIRE_THROWS_AS(t.At(0, 0, 0), std::runtime_error);
}

TEST_CASE("Tensor At and Set cast through float for integer tensors", "[tensor]") {
    int32_t data[] = {1, 2, 3, 4};
    cyxwiz::Tensor t({2, 2}, data, cyxwiz::DataType::Int32);

    REQUIRE(t.At(1, 0) == 3.0f);
    t.Set(0, 1, 8.8f);

    REQUIRE(t.At(0, 1) == 8.0f);
    REQUIRE(t.Data<int32_t>()[1] == 8);
}

TEST_CASE("Tensor typed accessors preserve native integer values", "[tensor]") {
    int32_t data[] = {1, 2, 3, 4};
    cyxwiz::Tensor t({2, 2}, data, cyxwiz::DataType::Int32);

    REQUIRE(t.AtAs<int32_t>(1, 0) == 3);
    t.SetAs<int32_t>(0, 1, 9);

    REQUIRE(t.AtAs<int32_t>(0, 1) == 9);
    REQUIRE(t.Data<int32_t>()[1] == 9);
    REQUIRE_THROWS_AS(t.AtAs<float>(0), std::runtime_error);
    REQUIRE_THROWS_AS(t.SetAs<int32_t>(2, 0, 1), std::out_of_range);
    REQUIRE_THROWS_AS(t.AtAs<int32_t>(0, 0, 0), std::runtime_error);
}

TEST_CASE("Tensor typed accessors support floating point tensors", "[tensor]") {
    double data[] = {1.25, 2.5, 3.75, 4.125};
    cyxwiz::Tensor t({2, 2}, data, cyxwiz::DataType::Float64);

    REQUIRE(t.AtAs<double>(1, 1) == 4.125);
    t.SetAs<double>(0, 0, 9.5);

    REQUIRE(t.AtAs<double>(0) == 9.5);
    REQUIRE_THROWS_AS(t.SetAs<float>(0, 1.0f), std::runtime_error);
}

TEST_CASE("Tensor slice extracts stepped ranges across dimensions", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({3, 4}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor rows = t.Slice(0, 1);
    cyxwiz::Tensor cols = t.Slice(1, 0, -1, 2);
    cyxwiz::Tensor tail = t.Slice(-1, -3, -1);

    REQUIRE(rows.Shape() == std::vector<size_t>{2, 4});
    REQUIRE(rows.Data<int32_t>()[0] == 4);
    REQUIRE(rows.Data<int32_t>()[7] == 11);

    REQUIRE(cols.Shape() == std::vector<size_t>{3, 2});
    REQUIRE(cols.Data<int32_t>()[0] == 0);
    REQUIRE(cols.Data<int32_t>()[1] == 2);
    REQUIRE(cols.Data<int32_t>()[4] == 8);
    REQUIRE(cols.Data<int32_t>()[5] == 10);

    REQUIRE(tail.Shape() == std::vector<size_t>{3, 3});
    REQUIRE(tail.Data<int32_t>()[0] == 1);
    REQUIRE(tail.Data<int32_t>()[1] == 2);
    REQUIRE(tail.Data<int32_t>()[2] == 3);
    REQUIRE(tail.Data<int32_t>()[8] == 11);

    REQUIRE_THROWS_AS(t.Slice(1, 3, 1), std::runtime_error);
    REQUIRE_THROWS_AS(t.Slice(1, 0, -1, 0), std::runtime_error);
}

TEST_CASE("Tensor index select gathers requested indices", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({3, 4}, cyxwiz::DataType::Int32);

    cyxwiz::Tensor rows = t.IndexSelect(0, {2, 0});
    cyxwiz::Tensor cols = t.IndexSelect(-1, {3, 1, -4});

    REQUIRE(rows.Shape() == std::vector<size_t>{2, 4});
    REQUIRE(rows.Data<int32_t>()[0] == 8);
    REQUIRE(rows.Data<int32_t>()[3] == 11);
    REQUIRE(rows.Data<int32_t>()[4] == 0);
    REQUIRE(rows.Data<int32_t>()[7] == 3);

    REQUIRE(cols.Shape() == std::vector<size_t>{3, 3});
    REQUIRE(cols.Data<int32_t>()[0] == 3);
    REQUIRE(cols.Data<int32_t>()[1] == 1);
    REQUIRE(cols.Data<int32_t>()[2] == 0);
    REQUIRE(cols.Data<int32_t>()[6] == 11);
    REQUIRE(cols.Data<int32_t>()[7] == 9);
    REQUIRE(cols.Data<int32_t>()[8] == 8);

    REQUIRE_THROWS_AS(t.IndexSelect(1, {4}), std::out_of_range);
    REQUIRE_THROWS_AS(t.IndexSelect(3, {0}), std::runtime_error);
}

TEST_CASE("Tensor cat concatenates along selected dimensions", "[tensor]") {
    int32_t a_data[] = {
        1, 2,
        3, 4
    };
    int32_t b_data[] = {
        5, 6,
        7, 8
    };
    cyxwiz::Tensor a({2, 2}, a_data, cyxwiz::DataType::Int32);
    cyxwiz::Tensor b({2, 2}, b_data, cyxwiz::DataType::Int32);

    cyxwiz::Tensor rows = cyxwiz::Tensor::Cat({a, b}, 0);
    cyxwiz::Tensor cols = cyxwiz::Tensor::Cat({a, b}, 1);

    REQUIRE(rows.Shape() == std::vector<size_t>{4, 2});
    REQUIRE(rows.Data<int32_t>()[0] == 1);
    REQUIRE(rows.Data<int32_t>()[3] == 4);
    REQUIRE(rows.Data<int32_t>()[4] == 5);
    REQUIRE(rows.Data<int32_t>()[7] == 8);

    REQUIRE(cols.Shape() == std::vector<size_t>{2, 4});
    REQUIRE(cols.Data<int32_t>()[0] == 1);
    REQUIRE(cols.Data<int32_t>()[1] == 2);
    REQUIRE(cols.Data<int32_t>()[2] == 5);
    REQUIRE(cols.Data<int32_t>()[3] == 6);
    REQUIRE(cols.Data<int32_t>()[4] == 3);
    REQUIRE(cols.Data<int32_t>()[7] == 8);
}

TEST_CASE("Tensor cat rejects incompatible tensors", "[tensor]") {
    cyxwiz::Tensor a = cyxwiz::Tensor::Ones({2, 2}, cyxwiz::DataType::Float32);
    cyxwiz::Tensor bad_shape = cyxwiz::Tensor::Ones({3, 3}, cyxwiz::DataType::Float32);
    cyxwiz::Tensor bad_dtype = cyxwiz::Tensor::Ones({2, 2}, cyxwiz::DataType::Int32);

    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({a, bad_shape}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({a, bad_dtype}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({a}, 2), std::runtime_error);
}

TEST_CASE("Tensor stack inserts a new dimension", "[tensor]") {
    float a_data[] = {1.0f, 2.0f};
    float b_data[] = {3.0f, 4.0f};
    cyxwiz::Tensor a({2}, a_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor b({2}, b_data, cyxwiz::DataType::Float32);

    cyxwiz::Tensor front = cyxwiz::Tensor::Stack({a, b}, 0);
    cyxwiz::Tensor back = cyxwiz::Tensor::Stack({a, b}, -1);

    REQUIRE(front.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(front.Data<float>()[0] == 1.0f);
    REQUIRE(front.Data<float>()[1] == 2.0f);
    REQUIRE(front.Data<float>()[2] == 3.0f);
    REQUIRE(front.Data<float>()[3] == 4.0f);

    REQUIRE(back.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(back.Data<float>()[0] == 1.0f);
    REQUIRE(back.Data<float>()[1] == 3.0f);
    REQUIRE(back.Data<float>()[2] == 2.0f);
    REQUIRE(back.Data<float>()[3] == 4.0f);
}

TEST_CASE("Tensor split and chunk produce row-major slices", "[tensor]") {
    auto t = cyxwiz::Tensor::RangeN({3, 4}, cyxwiz::DataType::Int32);

    std::vector<cyxwiz::Tensor> split_size = t.Split(2, 1);
    std::vector<cyxwiz::Tensor> split_sizes = t.Split({1, 3}, 1);
    std::vector<cyxwiz::Tensor> chunks = t.Chunk(2, 0);

    REQUIRE(split_size.size() == 2);
    REQUIRE(split_size[0].Shape() == std::vector<size_t>{3, 2});
    REQUIRE(split_size[0].Data<int32_t>()[0] == 0);
    REQUIRE(split_size[0].Data<int32_t>()[1] == 1);
    REQUIRE(split_size[1].Data<int32_t>()[0] == 2);
    REQUIRE(split_size[1].Data<int32_t>()[5] == 11);

    REQUIRE(split_sizes.size() == 2);
    REQUIRE(split_sizes[0].Shape() == std::vector<size_t>{3, 1});
    REQUIRE(split_sizes[1].Shape() == std::vector<size_t>{3, 3});
    REQUIRE(split_sizes[0].Data<int32_t>()[2] == 8);
    REQUIRE(split_sizes[1].Data<int32_t>()[0] == 1);
    REQUIRE(split_sizes[1].Data<int32_t>()[8] == 11);

    REQUIRE(chunks.size() == 2);
    REQUIRE(chunks[0].Shape() == std::vector<size_t>{2, 4});
    REQUIRE(chunks[1].Shape() == std::vector<size_t>{1, 4});
    REQUIRE(chunks[0].Data<int32_t>()[7] == 7);
    REQUIRE(chunks[1].Data<int32_t>()[0] == 8);

    REQUIRE_THROWS_AS(t.Split(0, 0), std::runtime_error);
    REQUIRE_THROWS_AS(t.Split({1, 1}, 1), std::runtime_error);
    REQUIRE_THROWS_AS(t.Chunk(0, 0), std::runtime_error);
}

TEST_CASE("Tensor split and chunk handle empty dimensions", "[tensor]") {
    cyxwiz::Tensor t({0, 3}, cyxwiz::DataType::Float32);

    REQUIRE(t.Split(2, 0).empty());
    REQUIRE(t.Chunk(2, 0).empty());
}

TEST_CASE("Tensor size calculations detect overflow", "[tensor]") {
    REQUIRE_THROWS_AS(
        cyxwiz::Tensor({(std::numeric_limits<size_t>::max)(), 2}, cyxwiz::DataType::UInt8),
        std::overflow_error);
    REQUIRE_THROWS_AS(
        cyxwiz::Tensor({((std::numeric_limits<size_t>::max)() / 4) + 1}, cyxwiz::DataType::Float32),
        std::overflow_error);
}

TEST_CASE("Tensor arithmetic syncs lazily to host data", "[tensor]") {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};

    cyxwiz::Tensor a({2, 2}, a_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor b({2, 2}, b_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor c = a + b;

    REQUIRE(c.Shape() == std::vector<size_t>{2, 2});
    const float* c_data = c.Data<float>();
    REQUIRE(c_data[0] == 11.0f);
    REQUIRE(c_data[1] == 22.0f);
    REQUIRE(c_data[2] == 33.0f);
    REQUIRE(c_data[3] == 44.0f);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor device arithmetic keeps host data unmaterialized", "[tensor][arrayfire]") {
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    cyxwiz::Tensor a(af::constant(2.0f, 4, f32));
    cyxwiz::Tensor b(af::constant(5.0f, 4, f32));
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    cyxwiz::Tensor sum = a + b;
    REQUIRE(sum.Shape() == std::vector<size_t>{4});
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const cyxwiz::Tensor& readable_sum = sum;
    const float* host = readable_sum.Data<float>();
    REQUIRE(host[0] == 7.0f);
    REQUIRE(host[3] == 7.0f);
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + sum.NumBytes());
}
#endif

TEST_CASE("Tensor host allocations are tracked", "[tensor]") {
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();
    {
        cyxwiz::Tensor t({8}, cyxwiz::DataType::Float32);
        REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + t.NumBytes());
    }
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);
}

TEST_CASE("Tensor copy and move ownership keeps memory accounting balanced", "[tensor]") {
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();
    {
        float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        cyxwiz::Tensor original({4}, values, cyxwiz::DataType::Float32);
        const size_t one_tensor_bytes = original.NumBytes();

        cyxwiz::Tensor copied(original);
        REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + (one_tensor_bytes * 2));
        REQUIRE(copied.Data<float>()[2] == 3.0f);

        cyxwiz::Tensor assigned({2}, cyxwiz::DataType::Float32);
        assigned = copied;
        REQUIRE(assigned.Shape() == std::vector<size_t>{4});
        REQUIRE(assigned.Data<float>()[3] == 4.0f);

        cyxwiz::Tensor moved(std::move(assigned));
        REQUIRE(moved.Shape() == std::vector<size_t>{4});
        REQUIRE(moved.Data<float>()[0] == 1.0f);

        cyxwiz::Tensor move_assigned;
        move_assigned = std::move(moved);
        REQUIRE(move_assigned.Shape() == std::vector<size_t>{4});
        REQUIRE(move_assigned.Data<float>()[1] == 2.0f);
    }
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor row-major 2D ArrayFire conversion preserves layout", "[tensor][arrayfire]") {
    float data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    cyxwiz::Tensor t({2, 3}, data, cyxwiz::DataType::Float32);

    af::array arr = t.GetArrayRowMajor2D();
    REQUIRE(arr.dims(0) == 2);
    REQUIRE(arr.dims(1) == 3);

    cyxwiz::Tensor roundtrip = cyxwiz::Tensor::FromArrayRowMajor2D(arr);
    REQUIRE(roundtrip.Shape() == std::vector<size_t>{2, 3});

    const float* out = roundtrip.Data<float>();
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(out[i] == data[i]);
    }
}

TEST_CASE("Tensor row-major 3D ArrayFire conversion preserves layout", "[tensor][arrayfire]") {
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f
    };
    cyxwiz::Tensor t({2, 3, 4}, data, cyxwiz::DataType::Float32);

    af::array arr = t.GetArrayRowMajor3D();
    REQUIRE(arr.dims(0) == 2);
    REQUIRE(arr.dims(1) == 3);
    REQUIRE(arr.dims(2) == 4);

    cyxwiz::Tensor roundtrip = cyxwiz::Tensor::FromArrayRowMajor3D(arr);
    REQUIRE(roundtrip.Shape() == std::vector<size_t>{2, 3, 4});

    const float* out = roundtrip.Data<float>();
    for (size_t i = 0; i < 24; ++i) {
        REQUIRE(out[i] == data[i]);
    }
}

TEST_CASE("Tensor row-major ArrayFire setters materialize host data lazily", "[tensor][arrayfire]") {
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    af::array arr2 = af::constant(2.0f, 2, 3, f32);
    cyxwiz::Tensor t2 = cyxwiz::Tensor::FromArrayRowMajor2D(arr2);
    REQUIRE(t2.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);
    REQUIRE(t2.Data<float>()[0] == 2.0f);

    const size_t after_2d_host = cyxwiz::MemoryManager::GetAllocatedBytes();
    af::array arr3 = af::constant(4.0f, 2, 3, 4, f32);
    cyxwiz::Tensor t3 = cyxwiz::Tensor::FromArrayRowMajor3D(arr3);
    REQUIRE(t3.Shape() == std::vector<size_t>{2, 3, 4});
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == after_2d_host);
    REQUIRE(t3.Data<float>()[23] == 4.0f);
}

TEST_CASE("Tensor ArrayFire-native construction preserves non-constant host order", "[tensor][arrayfire]") {
    float data2[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    af::array arr2(2, 3, data2);
    cyxwiz::Tensor t2(arr2);

    REQUIRE(t2.Shape() == std::vector<size_t>{2, 3});
    const float* out2 = t2.Data<float>();
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(out2[i] == data2[i]);
    }

    float data3[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    af::array arr3(2, 3, 2, data3);
    cyxwiz::Tensor t3(arr3);

    REQUIRE(t3.Shape() == std::vector<size_t>{2, 3, 2});
    const float* out3 = t3.Data<float>();
    for (size_t i = 0; i < 12; ++i) {
        REQUIRE(out3[i] == data3[i]);
    }
}

TEST_CASE("Tensor SetFromArray preserves non-constant ArrayFire-native host order", "[tensor][arrayfire]") {
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    af::array arr(2, 2, 2, data);

    cyxwiz::Tensor t;
    t.SetFromArray(arr);

    REQUIRE(t.Shape() == std::vector<size_t>{2, 2, 2});
    const float* out = t.Data<float>();
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(out[i] == data[i]);
    }
}

TEST_CASE("Tensor copy and move preserve row-major device layout", "[tensor][arrayfire]") {
    float data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    cyxwiz::Tensor original({2, 2, 3}, data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor device_only = cyxwiz::Tensor::FromArrayRowMajor3D(original.GetArrayRowMajor3D());

    cyxwiz::Tensor copied(device_only);
    cyxwiz::Tensor moved(std::move(copied));
    cyxwiz::Tensor assigned;
    assigned = moved;
    cyxwiz::Tensor move_assigned;
    move_assigned = std::move(assigned);

    REQUIRE(move_assigned.Shape() == std::vector<size_t>{2, 2, 3});
    const float* out = move_assigned.Data<float>();
    for (size_t i = 0; i < 12; ++i) {
        REQUIRE(out[i] == data[i]);
    }
}

TEST_CASE("Tensor ArrayFire construction materializes host data lazily", "[tensor][arrayfire]") {
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    af::array arr = af::constant(3.0f, 4, f32);
    cyxwiz::Tensor t(arr);

    REQUIRE(t.Shape() == std::vector<size_t>{4});
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const float* host = t.Data<float>();
    REQUIRE(host[0] == 3.0f);
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + t.NumBytes());
}

TEST_CASE("Tensor host mutation invalidates cached ArrayFire data", "[tensor][arrayfire]") {
    float data[] = {1.0f, 2.0f, 3.0f};
    cyxwiz::Tensor t({3}, data, cyxwiz::DataType::Float32);

    (void)t.GetArray();
    t.Data<float>()[1] = 7.0f;

    af::array refreshed = t.GetArray();
    std::vector<float> host(3);
    refreshed.host(host.data());

    REQUIRE(host[0] == 1.0f);
    REQUIRE(host[1] == 7.0f);
    REQUIRE(host[2] == 3.0f);
}
#endif
