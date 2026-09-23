#pragma once

#include "spatial_sample_shape.h"
#include <cyxwiz/tensor.h>
#include <vector>

namespace cyxwiz {

// Row features are explicitly HWC-flattened, never inferred from dimensions.
// Transform through existing semantic Tensor operations; no host staging or
// alternate tensor ownership. The inverse also serves the gradient boundary.
Tensor SpatialBatchFromRows(const Tensor &rows,
                            const std::vector<size_t> &sample_shape);
Tensor SpatialBatchToRows(const Tensor &spatial);

} // namespace cyxwiz
