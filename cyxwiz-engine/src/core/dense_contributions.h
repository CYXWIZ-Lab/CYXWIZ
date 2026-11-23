#pragma once

// Contributions (TOFIX178 6.4, approved mockup boards Main and Dense): how a
// Dense layer's output was computed for one sample, output = sum of w x x
// over its inputs + b, from the captured input and weights of that step (the
// same numbers the forward used). The largest terms first.

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz::contributions {

struct Term {
    size_t input = 0;
    float w = 0.0f;
    float x = 0.0f;
    float wx = 0.0f;
};

struct Explanation {
    std::vector<Term> top;  // the largest |w x x| first
    double sum = 0.0;       // over every input
    double bias = 0.0;
    double value = 0.0;     // sum + bias: the output before any activation
    size_t inputs = 0;
    size_t rest = 0;        // inputs not in `top`
    double rest_sum = 0.0;  // their share of the sum
};

// One output's terms: `weights` [outputs, inputs] row-major, `x` the
// sample's input, `bias` (empty when the layer has none).
Explanation Explain(const std::vector<float>& weight_row, const std::vector<float>& x, double bias, size_t top);

// Row `row` of a [rows, columns] matrix, read from the device alone.
std::vector<float> MatrixRow(const Tensor& matrix, size_t row);

// Where input `index` of a flattened input sits in the layer it came from
// ([H, W, C] row-major): "row 4, column 6, channel 65"; "" when `source_shape`
// is not [H, W, C] with that many elements.
std::string InputPlace(size_t index, const std::vector<size_t>& source_shape);

// "+0.0681", "-0.0833": a term or a weight, 4 decimals and its sign.
std::string Signed(double value);

}  // namespace cyxwiz::contributions
