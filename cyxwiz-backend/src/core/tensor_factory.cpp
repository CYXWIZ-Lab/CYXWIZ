#include "cyxwiz/tensor.h"

#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"
#endif

namespace cyxwiz {

namespace {

enum class FactoryKind {
    Zeros,
    Ones,
    Random,
    RandomSeeded,
    RangeN,
};

#ifdef CYXWIZ_HAS_ARRAYFIRE
// Uniform [0, 1) for floats; integers floor(u * 100), UInt8 floor(u * 256).
af::array RandomArray(dim_t count, DataType dtype, af::randomEngine* engine) {
    const auto uniform = [count, engine](af::dtype type) {
        return engine ? af::randu(af::dim4(count), type, *engine) : af::randu(af::dim4(count), type);
    };
    if (dtype == DataType::Float32 || dtype == DataType::Float64) {
        return uniform(tensor_math_utils::ArrayFireType(dtype));
    }
    const double upper = dtype == DataType::UInt8 ? 256.0 : 100.0;
    return af::floor(uniform(f32) * upper).as(tensor_math_utils::ArrayFireType(dtype));
}

// Every rank is built as one row-major flat array: the draws land in storage
// order, as the earlier per-rank layouts did, so seeded values are unchanged.
af::array CreateFactoryArray(dim_t count, DataType dtype, FactoryKind kind, uint64_t seed) {
    const af::dtype type = tensor_math_utils::ArrayFireType(dtype);
    switch (kind) {
        case FactoryKind::Zeros: return af::constant(0.0, af::dim4(count), type);
        case FactoryKind::Ones: return af::constant(1.0, af::dim4(count), type);
        case FactoryKind::Random: return RandomArray(count, dtype, nullptr);
        case FactoryKind::RandomSeeded: {
            af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, seed);
            af::array values = RandomArray(count, dtype, &engine);
            values.eval();  // draw before the local engine goes away
            return values;
        }
        case FactoryKind::RangeN: break;
    }
    return af::iota(af::dim4(count), af::dim4(1), type);
}
#endif

Tensor MaterializeFactory(const std::vector<size_t>& shape,
                          DataType dtype,
                          FactoryKind kind,
                          uint64_t seed,
                          const char* operation_name) {
    const size_t count = tensor_utils::CheckedProduct(
        shape,
        0,
        shape.size(),
        "Tensor factory: shape element count overflow");
    size_t bytes = 0;
    if (!tensor_utils::SafeMultiply(
            count, tensor_utils::ElementSize(dtype), bytes)) {
        throw std::overflow_error("Tensor factory: shape byte count overflow");
    }
    if (count == 0) return Tensor(shape, dtype);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (count > static_cast<size_t>((std::numeric_limits<dim_t>::max)())) {
        throw std::overflow_error(std::string(operation_name) + ": element count exceeds the ArrayFire range");
    }
    try {
        af::array values = CreateFactoryArray(static_cast<dim_t>(count), dtype, kind, seed);
        values.eval();
        return Tensor::FromRowMajorFlatArray(values, shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError(operation_name, e));
    }
#else
    (void)kind;
    (void)seed;
    throw std::runtime_error(std::string(operation_name) + " runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace

Tensor Tensor::Zeros(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::Zeros, 0, "Tensor::Zeros");
}

Tensor Tensor::Ones(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::Ones, 0, "Tensor::Ones");
}

Tensor Tensor::Random(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::Random, 0, "Tensor::Random");
}

Tensor Tensor::RandomSeeded(const std::vector<size_t>& shape,
                            uint64_t seed,
                            DataType dtype) {
    return MaterializeFactory(
        shape,
        dtype,
        FactoryKind::RandomSeeded,
        seed,
        "Tensor::RandomSeeded");
}

Tensor Tensor::RangeN(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::RangeN, 0, "Tensor::RangeN");
}

} // namespace cyxwiz
