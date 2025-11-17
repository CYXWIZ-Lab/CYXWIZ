#include "cyxwiz/tensor.h"
#include "cyxwiz/device.h"
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
// Helper: Convert CyxWiz DataType to ArrayFire dtype
static af::dtype ToArrayFireType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return af::dtype::f32;
        case DataType::Float64: return af::dtype::f64;
        case DataType::Int32: return af::dtype::s32;
        case DataType::Int64: return af::dtype::s64;
        case DataType::UInt8: return af::dtype::u8;
        default: throw std::runtime_error("Unsupported DataType for ArrayFire");
    }
}

// Helper: Create ArrayFire array from CPU data
static af::array* CreateArrayFireArray(const std::vector<size_t>& shape, DataType dtype, const void* data) {
    // Convert shape to af::dim4
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape.size() && i < 4; i++) {
        dims[i] = shape[i];
    }

    // Create ArrayFire array from host data
    af::array* arr = new af::array(dims, ToArrayFireType(dtype));

    if (data) {
        // Copy data from CPU to GPU
        arr->write(data, arr->bytes(), afHost);
    }

    return arr;
}

// Helper: Sync ArrayFire array back to CPU memory
static void SyncArrayFireToCPU(const af::array* af_arr, void* cpu_data) {
    if (af_arr && cpu_data) {
        af_arr->host(cpu_data);
    }
}
#endif

Tensor::Tensor()
    : dtype_(DataType::Float32), device_(nullptr), data_(nullptr), owns_data_(false)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
}

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(nullptr), owns_data_(true)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0) {
        data_ = malloc(num_bytes);
        memset(data_, 0, num_bytes);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, const void* data, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(nullptr), owns_data_(true)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0 && data) {
        data_ = malloc(num_bytes);
        memcpy(data_, data, num_bytes);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_), owns_data_(true)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0 && other.data_) {
        data_ = malloc(num_bytes);
        memcpy(data_, other.data_, num_bytes);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
      data_(other.data_), owns_data_(other.owns_data_)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(other.af_array_)
#endif
{
    other.data_ = nullptr;
    other.owns_data_ = false;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    other.af_array_ = nullptr;
#endif
}

Tensor::~Tensor() {
    if (owns_data_ && data_) {
        free(data_);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (af_array_) {
        delete af_array_;
    }
#endif
}

size_t Tensor::NumElements() const {
    size_t count = 1;
    for (size_t dim : shape_) {
        count *= dim;
    }
    return count;
}

size_t Tensor::NumBytes() const {
    size_t elem_size = 0;
    switch (dtype_) {
        case DataType::Float32: elem_size = 4; break;
        case DataType::Float64: elem_size = 8; break;
        case DataType::Int32: elem_size = 4; break;
        case DataType::Int64: elem_size = 8; break;
        case DataType::UInt8: elem_size = 1; break;
    }
    return NumElements() * elem_size;
}

void* Tensor::Data() {
    return data_;
}

const void* Tensor::Data() const {
    return data_;
}

Tensor Tensor::Zeros(const std::vector<size_t>& shape, DataType dtype) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated zeros creation
    try {
        // Convert shape to af::dim4
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < shape.size() && i < 4; i++) {
            dims[i] = static_cast<unsigned int>(shape[i]);
        }

        // Create ArrayFire array filled with zeros
        af::array zeros_arr = af::constant(0.0, dims, ToArrayFireType(dtype));

        // Create tensor and copy data back to CPU
        Tensor t(shape, dtype);
        zeros_arr.host(t.data_);

        return t;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire zeros creation failed, using CPU: {}", e.what());
    }
#endif

    // CPU fallback (constructor already zeros memory via memset)
    return Tensor(shape, dtype);
}

Tensor Tensor::Ones(const std::vector<size_t>& shape, DataType dtype) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated ones creation
    try {
        // Convert shape to af::dim4
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < shape.size() && i < 4; i++) {
            dims[i] = static_cast<unsigned int>(shape[i]);
        }

        // Create ArrayFire array filled with ones
        af::array ones_arr = af::constant(1.0, dims, ToArrayFireType(dtype));

        // Create tensor and copy data back to CPU
        Tensor t(shape, dtype);
        ones_arr.host(t.data_);

        return t;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire ones creation failed, using CPU: {}", e.what());
    }
#endif

    // CPU fallback
    Tensor t(shape, dtype);

    // Fill with ones based on data type
    size_t num_elements = t.NumElements();
    switch (dtype) {
        case DataType::Float32: {
            float* data = static_cast<float*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1.0f;
            }
            break;
        }
        case DataType::Float64: {
            double* data = static_cast<double*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1.0;
            }
            break;
        }
        case DataType::Int32: {
            int32_t* data = static_cast<int32_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1;
            }
            break;
        }
        case DataType::Int64: {
            int64_t* data = static_cast<int64_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1;
            }
            break;
        }
        case DataType::UInt8: {
            uint8_t* data = static_cast<uint8_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1;
            }
            break;
        }
    }

    return t;
}

Tensor Tensor::Random(const std::vector<size_t>& shape, DataType dtype) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated random generation
    try {
        // Convert shape to af::dim4
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < shape.size() && i < 4; i++) {
            dims[i] = static_cast<unsigned int>(shape[i]);
        }

        // Create ArrayFire array with random values [0, 1)
        af::array random_arr = af::randu(dims, ToArrayFireType(dtype));

        // Create tensor and copy data back to CPU
        Tensor t(shape, dtype);
        random_arr.host(t.data_);

        return t;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire random generation failed, using CPU: {}", e.what());
    }
#endif

    // CPU fallback
    Tensor t(shape, dtype);

    // Fill with random values [0, 1) based on data type
    size_t num_elements = t.NumElements();
    switch (dtype) {
        case DataType::Float32: {
            float* data = static_cast<float*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
            break;
        }
        case DataType::Float64: {
            double* data = static_cast<double*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            }
            break;
        }
        case DataType::Int32: {
            int32_t* data = static_cast<int32_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = rand() % 100;  // Random int [0, 99]
            }
            break;
        }
        case DataType::Int64: {
            int64_t* data = static_cast<int64_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = rand() % 100;  // Random int [0, 99]
            }
            break;
        }
        case DataType::UInt8: {
            uint8_t* data = static_cast<uint8_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = rand() % 256;  // Random byte [0, 255]
            }
            break;
        }
    }

    return t;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // Check shapes match
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise addition");
    }

    // Check data types match
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise addition");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Use ArrayFire for GPU-accelerated computation
    try {
        // Create ArrayFire arrays from CPU data
        af::array* a_arr = CreateArrayFireArray(shape_, dtype_, data_);
        af::array* b_arr = CreateArrayFireArray(other.shape_, other.dtype_, other.data_);

        // Perform GPU-accelerated addition
        af::array result_arr = *a_arr + *b_arr;

        // Create result tensor
        Tensor result(shape_, dtype_);

        // Copy result back to CPU
        result_arr.host(result.data_);

        // Cleanup
        delete a_arr;
        delete b_arr;

        return result;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire operation failed, falling back to CPU: {}", e.what());
        // Fall through to CPU implementation
    }
#endif

    // CPU fallback implementation
    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
    }

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise subtraction");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise subtraction");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated subtraction
    try {
        af::array* a_arr = CreateArrayFireArray(shape_, dtype_, data_);
        af::array* b_arr = CreateArrayFireArray(other.shape_, other.dtype_, other.data_);

        af::array result_arr = *a_arr - *b_arr;

        Tensor result(shape_, dtype_);
        result_arr.host(result.data_);

        delete a_arr;
        delete b_arr;

        return result;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire subtraction failed, using CPU: {}", e.what());
    }
#endif

    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise multiplication");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise multiplication");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated multiplication
    try {
        af::array* a_arr = CreateArrayFireArray(shape_, dtype_, data_);
        af::array* b_arr = CreateArrayFireArray(other.shape_, other.dtype_, other.data_);

        af::array result_arr = *a_arr * *b_arr;

        Tensor result(shape_, dtype_);
        result_arr.host(result.data_);

        delete a_arr;
        delete b_arr;

        return result;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire multiplication failed, using CPU: {}", e.what());
    }
#endif

    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
    }

    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise division");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise division");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated division
    try {
        af::array* a_arr = CreateArrayFireArray(shape_, dtype_, data_);
        af::array* b_arr = CreateArrayFireArray(other.shape_, other.dtype_, other.data_);

        af::array result_arr = *a_arr / *b_arr;

        Tensor result(shape_, dtype_);
        result_arr.host(result.data_);

        delete a_arr;
        delete b_arr;

        return result;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire division failed, using CPU: {}", e.what());
    }
#endif

    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
    }

    return result;
}

} // namespace cyxwiz
