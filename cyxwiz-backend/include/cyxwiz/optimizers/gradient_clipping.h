#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// Global-norm gradient clipping (torch.nn.utils.clip_grad_norm_): computes the
// L2 norm over all gradient tensors together and, when it exceeds max_norm,
// scales every gradient by max_norm / (norm + 1e-6). Returns the norm before
// clipping. One scalar is read back to the host per call.
CYXWIZ_API float ClipGradientsByGlobalNorm(std::map<std::string, Tensor>& gradients, float max_norm);

} // namespace cyxwiz
