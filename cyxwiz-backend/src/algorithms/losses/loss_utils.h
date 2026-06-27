#pragma once

#include "cyxwiz/loss.h"
#include "cyxwiz/tensor.h"

#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace loss_detail {

void ValidateFloat32Pair(const Tensor& predictions, const Tensor& targets, const char* name);

Tensor ApplyCpuReduction(const std::vector<size_t>& input_shape,
                         const std::vector<float>& values,
                         Reduction reduction);
Tensor ApplyClassReduction(const std::vector<float>& per_sample,
                           size_t batch,
                           Reduction reduction);

Tensor CpuMSEForward(const Tensor& predictions, const Tensor& targets, Reduction reduction);
Tensor CpuMSEBackward(const Tensor& predictions, const Tensor& targets, Reduction reduction);
Tensor CpuL1Forward(const Tensor& predictions, const Tensor& targets, Reduction reduction);
Tensor CpuL1Backward(const Tensor& predictions, const Tensor& targets, Reduction reduction);
Tensor CpuSmoothL1Forward(const Tensor& predictions,
                          const Tensor& targets,
                          float delta,
                          Reduction reduction);
Tensor CpuSmoothL1Backward(const Tensor& predictions,
                           const Tensor& targets,
                           float delta,
                           Reduction reduction);
Tensor CpuBCEForward(const Tensor& predictions, const Tensor& targets, float eps, Reduction reduction);
Tensor CpuBCEBackward(const Tensor& predictions, const Tensor& targets, float eps, Reduction reduction);
float CpuSigmoidValue(float x);
Tensor CpuBCEWithLogitsForward(const Tensor& predictions,
                               const Tensor& targets,
                               Reduction reduction,
                               float pos_weight = 1.0f);
Tensor CpuBCEWithLogitsBackward(const Tensor& predictions,
                                const Tensor& targets,
                                Reduction reduction,
                                float pos_weight = 1.0f);
Tensor CpuKLDivForward(const Tensor& predictions,
                       const Tensor& targets,
                       bool log_target,
                       Reduction reduction);
Tensor CpuKLDivBackward(const Tensor& predictions,
                        const Tensor& targets,
                        bool log_target,
                        Reduction reduction);

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array TensorToAf(const Tensor& t);
Tensor AfToTensor(const af::array& arr);
void LogArrayFireLossFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& tensor,
    const char* tensor_name);
af::array ApplyReduction(const af::array& loss, Reduction reduction);
af::array StableSoftmax(const af::array& x, int axis = 0);
af::array SignLike(const af::array& x);
#endif

} // namespace loss_detail
} // namespace cyxwiz
