#pragma once

// Word-vector tables between the host (row-major [rows, cols]) and the device
// ([rows, cols]), shared by the word2vec and GloVe trainers (TOFIX152).

#include "arrayfire_host_materialization.h"

#include <arrayfire.h>

#include <vector>

namespace cyxwiz {

inline af::array RowsToDevice(const std::vector<float>& values, dim_t rows, dim_t cols) {
    return af::array(cols, rows, values.data()).T();
}

inline std::vector<float> RowsToHost(const af::array& matrix, const char* operation) {
    std::vector<float> values(static_cast<size_t>(matrix.elements()));
    af::array transposed = matrix.T();
    transposed.eval();
    MaterializeArrayFireToHost(transposed, values.data(), ArrayFireHostSyncCategory::OutputMaterialization, operation,
                               "row_major_words_by_dim");
    return values;
}

}  // namespace cyxwiz
