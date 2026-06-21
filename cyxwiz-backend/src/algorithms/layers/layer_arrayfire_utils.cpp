#include "layer_arrayfire_utils.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE

#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

af::array TensorToAf(const Tensor& t) {
    return t.Shape().size() == 2 ? t.GetArrayRowMajor2D() : t.GetArray();
}

af::array TensorToAf3DRowMajor(const Tensor& t) {
    return t.Shape().size() == 3 ? t.GetArrayRowMajor3D() : TensorToAf(t);
}

Tensor AfToTensor(const af::array& arr) {
    // Count significant dimensions
    int ndims = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1) ndims = i + 1;
        else if (i == 0) ndims = 1;
    }

    // For 2D arrays, transpose to row-major before copying to Tensor
    if (ndims == 2) {
        return Tensor::FromArrayRowMajor2D(arr);
    }

    // For other dimensions, keep the ArrayFire result resident until host data is requested.
    return Tensor(arr);
}

Tensor AfToTensor3DRowMajor(const af::array& arr) {
    if (arr.numdims() > 3) {
        // Fall back to existing path for 4D; caller owns correctness.
        return AfToTensor(arr);
    }
    return Tensor::FromArrayRowMajor3D(arr);
}

int CheckedIntDim(size_t value, const char* name) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds ArrayFire int dimension limit");
    }
    return static_cast<int>(value);
}

af::array XavierUniform(int fan_in, int fan_out, af::dim4 dims) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    return af::randu(dims, af::dtype::f32) * 2.0f * limit - limit;
}

af::array KaimingUniform(int fan_in, af::dim4 dims) {
    float limit = std::sqrt(6.0f / fan_in);
    return af::randu(dims, af::dtype::f32) * 2.0f * limit - limit;
}

} // namespace cyxwiz

#endif
