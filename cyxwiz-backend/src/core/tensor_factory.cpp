#include "cyxwiz/tensor.h"

#include "tensor_backend_observation_utils.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
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

std::mt19937& CpuRandomEngine() {
    thread_local std::mt19937 engine{std::random_device{}()};
    return engine;
}

void FillRandomNative(Tensor& tensor, std::mt19937& engine) {
    const size_t count = tensor.NumElements();
    switch (tensor.GetDataType()) {
        case DataType::Float32: {
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
            float* values = tensor.MutableData<float>();
            for (size_t index = 0; index < count; ++index) {
                values[index] = distribution(engine);
            }
            return;
        }
        case DataType::Float64: {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double* values = tensor.MutableData<double>();
            for (size_t index = 0; index < count; ++index) {
                values[index] = distribution(engine);
            }
            return;
        }
        case DataType::Int32: {
            std::uniform_int_distribution<int32_t> distribution(0, 99);
            int32_t* values = tensor.MutableData<int32_t>();
            for (size_t index = 0; index < count; ++index) {
                values[index] = distribution(engine);
            }
            return;
        }
        case DataType::Int64: {
            std::uniform_int_distribution<int64_t> distribution(0, 99);
            int64_t* values = tensor.MutableData<int64_t>();
            for (size_t index = 0; index < count; ++index) {
                values[index] = distribution(engine);
            }
            return;
        }
        case DataType::UInt8: {
            std::uniform_int_distribution<int> distribution(0, 255);
            uint8_t* values = tensor.MutableData<uint8_t>();
            for (size_t index = 0; index < count; ++index) {
                values[index] = static_cast<uint8_t>(distribution(engine));
            }
            return;
        }
    }
}

template <typename T>
void FillOnes(Tensor& tensor) {
    T* values = tensor.MutableData<T>();
    for (size_t index = 0; index < tensor.NumElements(); ++index) {
        values[index] = static_cast<T>(1);
    }
}

template <typename T>
void FillRange(Tensor& tensor) {
    T* values = tensor.MutableData<T>();
    for (size_t index = 0; index < tensor.NumElements(); ++index) {
        values[index] = static_cast<T>(index);
    }
}

Tensor ApplyNativeFactory(const std::vector<size_t>& shape,
                          DataType dtype,
                          FactoryKind kind,
                          uint64_t seed) {
    Tensor result(shape, dtype);
    if (result.NumElements() == 0 || kind == FactoryKind::Zeros) {
        return result;
    }
    if (kind == FactoryKind::Ones) {
        switch (dtype) {
            case DataType::Float32: FillOnes<float>(result); break;
            case DataType::Float64: FillOnes<double>(result); break;
            case DataType::Int32: FillOnes<int32_t>(result); break;
            case DataType::Int64: FillOnes<int64_t>(result); break;
            case DataType::UInt8: FillOnes<uint8_t>(result); break;
        }
        return result;
    }
    if (kind == FactoryKind::Random) {
        FillRandomNative(result, CpuRandomEngine());
        return result;
    }
    if (kind == FactoryKind::RandomSeeded) {
        std::seed_seq seed_sequence{
            static_cast<uint32_t>(seed),
            static_cast<uint32_t>(seed >> 32),
        };
        std::mt19937 engine(seed_sequence);
        FillRandomNative(result, engine);
        return result;
    }

    switch (dtype) {
        case DataType::Float32: FillRange<float>(result); break;
        case DataType::Float64: FillRange<double>(result); break;
        case DataType::Int32: FillRange<int32_t>(result); break;
        case DataType::Int64: FillRange<int64_t>(result); break;
        case DataType::UInt8: FillRange<uint8_t>(result); break;
    }
    return result;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordFactoryArrayFireFallback(const char* operation_name,
                                    const std::vector<size_t>& shape,
                                    DataType dtype,
                                    const char* attribute,
                                    const char* error_message) {
    const std::string message =
        tensor_backend_observation::RecordArrayFireFallback(
            operation_name,
            tensor_backend_observation::DataTypeName(dtype),
            tensor_backend_observation::BuildTensorOpSignature(
                {},
                shape,
                dtype,
                std::string(attribute) + ";rank=" +
                    std::to_string(shape.size())),
            error_message);
    spdlog::warn("{}", message);
}

bool FitsArrayFireShape(const std::vector<size_t>& shape) {
    for (size_t dimension : shape) {
        if (dimension >
            static_cast<size_t>((std::numeric_limits<dim_t>::max)())) {
            return false;
        }
    }
    return true;
}

af::dim4 ReversedArrayFireDims(const std::vector<size_t>& shape) {
    af::dim4 dimensions(1, 1, 1, 1);
    for (size_t index = 0; index < shape.size(); ++index) {
        dimensions[static_cast<unsigned>(index)] =
            static_cast<dim_t>(shape[shape.size() - 1 - index]);
    }
    return dimensions;
}

af::array ToSemanticLayout(af::array row_major_storage, size_t rank) {
    if (rank == 2) return af::transpose(row_major_storage);
    if (rank == 3) return af::reorder(row_major_storage, 2, 1, 0);
    if (rank == 4) return af::reorder(row_major_storage, 3, 2, 1, 0);
    return row_major_storage;
}

af::array RandomArray(const af::dim4& storage_dims,
                      DataType dtype,
                      af::randomEngine* engine) {
    const auto random_uniform = [engine](const af::dim4& dimensions,
                                         af::dtype output_type) {
        return engine ? af::randu(dimensions, output_type, *engine)
                      : af::randu(dimensions, output_type);
    };
    if (dtype == DataType::Float32 || dtype == DataType::Float64) {
        return random_uniform(
            storage_dims, tensor_math_utils::ArrayFireType(dtype));
    }

    const double upper = dtype == DataType::UInt8 ? 256.0 : 100.0;
    return af::floor(random_uniform(storage_dims, f32) * upper)
        .as(tensor_math_utils::ArrayFireType(dtype));
}

af::array CreateFactoryArray(const std::vector<size_t>& shape,
                             DataType dtype,
                             FactoryKind kind,
                             uint64_t seed) {
    const af::dim4 storage_dims = ReversedArrayFireDims(shape);
    af::array storage;
    if (kind == FactoryKind::Zeros) {
        storage = af::constant(
            0.0, storage_dims, tensor_math_utils::ArrayFireType(dtype));
    } else if (kind == FactoryKind::Ones) {
        storage = af::constant(
            1.0, storage_dims, tensor_math_utils::ArrayFireType(dtype));
    } else if (kind == FactoryKind::Random) {
        storage = RandomArray(storage_dims, dtype, nullptr);
    } else if (kind == FactoryKind::RandomSeeded) {
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, seed);
        storage = RandomArray(storage_dims, dtype, &engine);
        storage.eval();
    } else {
        storage = af::iota(
            storage_dims,
            af::dim4(1),
            tensor_math_utils::ArrayFireType(dtype));
    }
    af::array semantic = ToSemanticLayout(std::move(storage), shape.size());
    semantic.eval();
    return semantic;
}
#endif

Tensor MaterializeFactory(const std::vector<size_t>& shape,
                          DataType dtype,
                          FactoryKind kind,
                          uint64_t seed,
                          [[maybe_unused]] const char* operation_name,
                          [[maybe_unused]] const char* operation_attribute) {
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
    if (shape.size() > 4 || !FitsArrayFireShape(shape)) {
        RecordFactoryArrayFireFallback(
            operation_name,
            shape,
            dtype,
            operation_attribute,
            shape.size() > 4
                ? "ArrayFire Tensor factories support ranks up to 4"
                : "Tensor factory dimension exceeds ArrayFire range");
        return ApplyNativeFactory(shape, dtype, kind, seed);
    }

    try {
        af::array output = CreateFactoryArray(shape, dtype, kind, seed);
        return Tensor::FromSemanticArray(output, shape);
    } catch (const af::exception& error) {
        RecordFactoryArrayFireFallback(
            operation_name,
            shape,
            dtype,
            operation_attribute,
            error.what());
    }
#endif

    return ApplyNativeFactory(shape, dtype, kind, seed);
}

} // namespace

Tensor Tensor::Zeros(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::Zeros, 0, "Tensor::Zeros", "op=zeros");
}

Tensor Tensor::Ones(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::Ones, 0, "Tensor::Ones", "op=ones");
}

Tensor Tensor::Random(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::Random, 0, "Tensor::Random", "op=random");
}

Tensor Tensor::RandomSeeded(const std::vector<size_t>& shape,
                            uint64_t seed,
                            DataType dtype) {
    return MaterializeFactory(
        shape,
        dtype,
        FactoryKind::RandomSeeded,
        seed,
        "Tensor::RandomSeeded",
        "op=random_seeded");
}

Tensor Tensor::RangeN(const std::vector<size_t>& shape, DataType dtype) {
    return MaterializeFactory(
        shape, dtype, FactoryKind::RangeN, 0, "Tensor::RangeN", "op=range_n");
}

} // namespace cyxwiz
