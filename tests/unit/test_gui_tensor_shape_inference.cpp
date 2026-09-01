#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/gui/tensor_shape_inference_contract.h"

#include <limits>
#include <vector>

using gui::tensor_shape_inference::InferPool2DOutputShape;
using gui::tensor_shape_inference::ResolveReshapeOutputShape;

TEST_CASE("GUI pooling shape matches the runtime valid-window contract",
          "[gui][shape-inference][pooling]") {
    CHECK(InferPool2DOutputShape({6, 7, 3}, 3, 2) ==
          std::vector<size_t>{2, 3, 3});
    CHECK(InferPool2DOutputShape({5, 5, 4}, 2, 1) ==
          std::vector<size_t>{4, 4, 4});

    CHECK_FALSE(InferPool2DOutputShape({2, 2, 3}, 3, 1));
    CHECK_FALSE(InferPool2DOutputShape({5, 5, 3}, 0, 1));
    CHECK_FALSE(InferPool2DOutputShape({5, 5, 3}, 2, 0));
    CHECK_FALSE(InferPool2DOutputShape({5, 5}, 2, 2));
}

TEST_CASE("GUI reshape inference preserves the input element count",
          "[gui][shape-inference][reshape]") {
    CHECK(ResolveReshapeOutputShape({2, 3, 4}, "[2, -1, 3]") ==
          std::vector<size_t>{2, 4, 3});
    CHECK(ResolveReshapeOutputShape({2, 3, 4}, "6,4") ==
          std::vector<size_t>{6, 4});

    CHECK_FALSE(ResolveReshapeOutputShape({2, 3, 4}, "[5, 5]"));
    CHECK_FALSE(ResolveReshapeOutputShape({2, 3, 4}, "[-1, -1]"));
    CHECK_FALSE(ResolveReshapeOutputShape({2, 3, 4}, "[2, 0, 12]"));
    CHECK_FALSE(ResolveReshapeOutputShape({2, 3, 4}, "[2,,12]"));
    CHECK_FALSE(ResolveReshapeOutputShape(
        {std::numeric_limits<size_t>::max(), 2}, "[-1]"));
}
