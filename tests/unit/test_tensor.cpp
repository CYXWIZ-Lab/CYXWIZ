#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/tensor.h>

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

// TODO: Add more tests
