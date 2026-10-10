#include "loss_utils.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

// Undefine Windows macros that conflict with ArrayFire functions.
// Must be AFTER all includes (Windows headers define these).
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {
namespace loss_detail {

void ThrowLossNeedsArrayFire(const char* operation_name) {
    throw std::runtime_error(std::string(operation_name) + " runs on ArrayFire, and this build has no ArrayFire");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void ThrowLossDeviceError(const char* operation_name, const af::exception& error) {
    throw std::runtime_error(std::string(operation_name) + " failed on the ArrayFire device: " + error.what());
}
#endif

void ValidateFloat32Pair(const Tensor& predictions, const Tensor& targets, const char* name) {
    if (predictions.GetDataType() != DataType::Float32 || targets.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 tensors");
    }
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error(std::string(name) + " requires matching prediction and target shapes");
    }
}



















#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array TensorToAf(const Tensor& t) {
    return t.GetSemanticArray();
}

Tensor AfToTensor(const af::array& arr) {
    int ndims = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1) {
            ndims = i + 1;
        } else if (i == 0) {
            ndims = 1;
        }
    }

    if (ndims == 2) {
        return Tensor::FromArrayRowMajor2D(arr);
    }

    return Tensor(arr);
}

Tensor AfToTensor(const af::array& arr,
                  const std::vector<size_t>& semantic_shape) {
    return Tensor::FromSemanticArray(arr, semantic_shape);
}
#endif





#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array ApplyReduction(const af::array& loss, Reduction reduction) {
    switch (reduction) {
        case Reduction::None:
            return loss;
        case Reduction::Mean: {
            // ArrayFire's single-argument reduction operates on the first
            // non-singleton dimension. Loss::Mean is a global reduction, so
            // flatten first to produce exactly one scalar for tensors such as
            // [batch, forecast_horizon].
            af::array result = af::mean(af::flat(loss));
            result.eval();
            return result;
        }
        case Reduction::Sum: {
            af::array result = af::sum(af::flat(loss));
            result.eval();
            return result;
        }
        default: {
            af::array result = af::mean(af::flat(loss));
            result.eval();
            return result;
        }
    }
}

af::array StableSoftmax(const af::array& x, int axis) {
    af::array max_val = af::max(x, axis);
    max_val.eval();
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[axis] = x.dims(axis);
    af::array x_stable = x - af::tile(max_val, tile_dims);
    x_stable.eval();
    af::array exp_x = af::exp(x_stable);
    exp_x.eval();
    af::array sum_exp = af::sum(exp_x, axis);
    sum_exp.eval();
    af::array result = exp_x / af::tile(sum_exp, tile_dims);
    result.eval();
    return result;
}

af::array SignLike(const af::array& x) {
    const af::array ones = af::constant(1.0f, x.dims(), f32);
    const af::array minus_ones = af::constant(-1.0f, x.dims(), f32);
    const af::array zeros = af::constant(0.0f, x.dims(), f32);
    return af::select(x > 0.0f, ones, af::select(x < 0.0f, minus_ones, zeros));
}

#endif // CYXWIZ_HAS_ARRAYFIRE

} // namespace loss_detail
} // namespace cyxwiz

