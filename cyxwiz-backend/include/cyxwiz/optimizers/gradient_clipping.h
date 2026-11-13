// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// Global-norm gradient clipping as torch.nn.utils.clip_grad_norm_ does it
// (its default, error_if_nonfinite=False): the L2 norm over all gradient
// tensors together, coef = max_norm / (norm + 1e-6) clamped at 1, and every
// gradient scaled by coef. A non-finite norm is not refused: a NaN norm
// makes every gradient NaN (the clamp keeps NaN), an infinite one makes the
// finite gradients 0. Returns the norm before clipping; one scalar is read
// back to the host per call.
CYXWIZ_API float ClipGradientsByGlobalNorm(std::map<std::string, Tensor>& gradients, float max_norm);

// The same clip without the read-back, on the device. Returns the norm
// before clipping as a [1] tensor on the device.
CYXWIZ_API Tensor ClipGradientsByGlobalNormOnDevice(std::map<std::string, Tensor>& gradients, float max_norm);

} // namespace cyxwiz
