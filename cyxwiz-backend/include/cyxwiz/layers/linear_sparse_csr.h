#pragma once

#include "../api_export.h"

#include <cstddef>
#include <cstdint>

namespace cyxwiz {

/**
 * Non-owning CSR input accepted only by the first Linear projection.
 *
 * This deliberately is not a sparse Tensor API. The caller owns the host
 * buffers for the duration of ForwardSparseCsr/BackwardSparseCsr, and the
 * Linear layer owns conversion to the selected ArrayFire backend.
 */
struct CYXWIZ_API LinearSparseCsrBatchView {
    size_t rows = 0;
    size_t columns = 0;
    size_t nnz = 0;
    const int32_t* row_offsets = nullptr;
    const int32_t* column_indices = nullptr;
    const float* values = nullptr;
};

} // namespace cyxwiz
