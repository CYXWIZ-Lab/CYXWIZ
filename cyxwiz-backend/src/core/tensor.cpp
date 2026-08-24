#include "cyxwiz/tensor.h"
#include "cyxwiz/device.h"
#include "cyxwiz/memory_manager.h"
#include "tensor_backend_observation_utils.h"
#include "tensor_utils.h"
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

class TensorHostBuffer {
public:
    TensorHostBuffer() = default;

    explicit TensorHostBuffer(size_t bytes)
        : data_(bytes > 0 ? MemoryManager::Allocate(bytes) : nullptr), size_(bytes) {
        if (bytes > 0 && !data_) {
            throw std::bad_alloc();
        }
    }

    TensorHostBuffer(const TensorHostBuffer&) = delete;
    TensorHostBuffer& operator=(const TensorHostBuffer&) = delete;

    TensorHostBuffer(TensorHostBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    TensorHostBuffer& operator=(TensorHostBuffer&& other) noexcept {
        if (this != &other) {
            Reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~TensorHostBuffer() {
        Reset();
    }

    void* Data() const { return data_; }
    size_t Size() const { return size_; }

private:
    void Reset() {
        if (data_) {
            MemoryManager::Deallocate(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    void* data_ = nullptr;
    size_t size_ = 0;
};

namespace {

void* HostData(const std::unique_ptr<TensorHostBuffer>& buffer) {
    return buffer ? buffer->Data() : nullptr;
}

std::unique_ptr<TensorHostBuffer> AllocateHostBuffer(size_t bytes) {
    return bytes > 0 ? std::make_unique<TensorHostBuffer>(bytes) : nullptr;
}

std::mt19937& CpuRandomEngine() {
    thread_local std::mt19937 engine{std::random_device{}()};
    return engine;
}

void FillRandomCpu(Tensor& tensor, std::mt19937& engine) {
    const size_t num_elements = tensor.NumElements();
    switch (tensor.GetDataType()) {
        case DataType::Float32: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float* data = static_cast<float*>(tensor.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = dist(engine);
            }
            break;
        }
        case DataType::Float64: {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double* data = static_cast<double*>(tensor.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = dist(engine);
            }
            break;
        }
        case DataType::Int32: {
            std::uniform_int_distribution<int32_t> dist(0, 99);
            int32_t* data = static_cast<int32_t*>(tensor.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = dist(engine);
            }
            break;
        }
        case DataType::Int64: {
            std::uniform_int_distribution<int64_t> dist(0, 99);
            int64_t* data = static_cast<int64_t*>(tensor.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = dist(engine);
            }
            break;
        }
        case DataType::UInt8: {
            std::uniform_int_distribution<int> dist(0, 255);
            uint8_t* data = static_cast<uint8_t*>(tensor.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<uint8_t>(dist(engine));
            }
            break;
        }
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordTensorCoreArrayFireFallback(
    const char* operation_name,
    const Tensor& input,
    const std::vector<size_t>& output_shape,
    const std::string& attributes,
    const char* error_message) {
    tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(input.GetDataType()),
        tensor_backend_observation::BuildTensorOpSignature(
            {input.Shape()},
            output_shape,
            input.GetDataType(),
            attributes),
        error_message);
}

void RecordTensorCoreCreationArrayFireFallback(
    const char* operation_name,
    const std::vector<size_t>& shape,
    DataType dtype,
    const std::string& attributes,
    const char* error_message) {
    tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(dtype),
        tensor_backend_observation::BuildTensorOpSignature(
            {},
            shape,
            dtype,
            attributes),
        error_message);
}

void RecordTensorCoreBinaryArrayFireFallback(
    const char* operation_name,
    const Tensor& left,
    const Tensor& right,
    const std::string& attributes,
    const char* error_message) {
    tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(left.GetDataType()),
        tensor_backend_observation::BuildTensorOpSignature(
            {left.Shape(), right.Shape()},
            left.Shape(),
            left.GetDataType(),
            attributes),
        error_message);
}
#endif

} // namespace

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

static DataType FromArrayFireType(af::dtype dtype) {
    switch (dtype) {
        case af::dtype::f32: return DataType::Float32;
        case af::dtype::f64: return DataType::Float64;
        case af::dtype::s32: return DataType::Int32;
        case af::dtype::s64: return DataType::Int64;
        case af::dtype::u8:  return DataType::UInt8;
        default: return DataType::Float32;
    }
}

static std::vector<size_t> ShapeFromArrayFireDims(const af::array& arr) {
    af::dim4 dims = arr.dims();
    std::vector<size_t> shape;
    for (int i = 0; i < 4; i++) {
        if (dims[i] > 1 || i == 0) {
            shape.push_back(static_cast<size_t>(dims[i]));
        } else if (dims[i] == 1 && i > 0) {
            bool has_larger = false;
            for (int j = i + 1; j < 4; j++) {
                if (dims[j] > 1) {
                    has_larger = true;
                }
            }
            if (has_larger) {
                shape.push_back(1);
            } else {
                break;
            }
        }
    }
    return shape;
}

#endif

Tensor::Tensor()
    : dtype_(DataType::Float32), device_(nullptr)
#ifdef CYXWIZ_HAS_ARRAYFIRE
      , host_current_(true), device_current_(false), device_layout_(TensorDeviceLayout::None)
#endif
{
}

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(nullptr)
#ifdef CYXWIZ_HAS_ARRAYFIRE
      , host_current_(true), device_current_(false), device_layout_(TensorDeviceLayout::None)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0) {
        host_buffer_ = AllocateHostBuffer(num_bytes);
        memset(host_buffer_->Data(), 0, num_bytes);
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, const void* data, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(nullptr)
#ifdef CYXWIZ_HAS_ARRAYFIRE
      , host_current_(true), device_current_(false), device_layout_(TensorDeviceLayout::None)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0 && data) {
        host_buffer_ = AllocateHostBuffer(num_bytes);
        memcpy(host_buffer_->Data(), data, num_bytes);
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
Tensor::Tensor(const af::array& arr)
    : device_(nullptr), af_array_(std::make_unique<af::array>(arr)),
      host_current_(false), device_current_(true), device_layout_(TensorDeviceLayout::ArrayFireNative)
{
    shape_ = ShapeFromArrayFireDims(arr);
    dtype_ = FromArrayFireType(arr.type());
}
#endif

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_)
#ifdef CYXWIZ_HAS_ARRAYFIRE
      , host_current_(other.host_current_), device_current_(other.device_current_),
        device_layout_(other.device_layout_)
#endif
{
    size_t num_bytes = NumBytes();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool copy_host = other.host_current_;
#else
    const bool copy_host = true;
#endif
    const void* other_data = HostData(other.host_buffer_);
    if (num_bytes > 0 && copy_host && other_data) {
        host_buffer_ = AllocateHostBuffer(num_bytes);
        memcpy(host_buffer_->Data(), other_data, num_bytes);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (other.device_current_ && other.af_array_) {
        af_array_ = std::make_unique<af::array>(*other.af_array_);
    }
#endif
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
      host_buffer_(std::move(other.host_buffer_))
#ifdef CYXWIZ_HAS_ARRAYFIRE
      , af_array_(std::move(other.af_array_)), host_current_(other.host_current_),
        device_current_(other.device_current_), device_layout_(other.device_layout_)
#endif
{
#ifdef CYXWIZ_HAS_ARRAYFIRE
    other.host_current_ = true;
    other.device_current_ = false;
    other.device_layout_ = TensorDeviceLayout::None;
#endif
}

Tensor::~Tensor() = default;

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // Copy from other
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        host_buffer_.reset();
#ifdef CYXWIZ_HAS_ARRAYFIRE
        af_array_.reset();
        host_current_ = other.host_current_;
        device_current_ = other.device_current_;
        device_layout_ = other.device_layout_;
#endif

        size_t num_bytes = NumBytes();
#ifdef CYXWIZ_HAS_ARRAYFIRE
        const bool copy_host = other.host_current_;
#else
        const bool copy_host = true;
#endif
        const void* other_data = HostData(other.host_buffer_);
        if (num_bytes > 0 && copy_host && other_data) {
            host_buffer_ = AllocateHostBuffer(num_bytes);
            memcpy(host_buffer_->Data(), other_data, num_bytes);
        }
#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (other.device_current_ && other.af_array_) {
            af_array_ = std::make_unique<af::array>(*other.af_array_);
        }
#endif
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Move from other
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        host_buffer_ = std::move(other.host_buffer_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
        af_array_ = std::move(other.af_array_);
        host_current_ = other.host_current_;
        device_current_ = other.device_current_;
        device_layout_ = other.device_layout_;
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE
        other.host_current_ = true;
        other.device_current_ = false;
        other.device_layout_ = TensorDeviceLayout::None;
#endif
    }
    return *this;
}

Tensor Tensor::Clone() const {
    return Tensor(*this);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array Tensor::GetArray() const {
    if (device_current_ && af_array_ && device_layout_ == TensorDeviceLayout::ArrayFireNative) {
        return *af_array_;
    }
    if (shape_.size() > 4) {
        throw std::runtime_error("Tensor::GetArray: ArrayFire supports at most 4 dimensions");
    }

    if (device_current_ && af_array_ && shape_.size() == 2 &&
        device_layout_ == TensorDeviceLayout::RowMajor2D) {
        af::array transposed = af::transpose(*af_array_);
        af::array converted = af::moddims(
            transposed,
            static_cast<dim_t>(shape_[0]),
            static_cast<dim_t>(shape_[1]));
        converted.eval();
        af_array_ = std::make_unique<af::array>(std::move(converted));
        device_layout_ = TensorDeviceLayout::ArrayFireNative;
        return *af_array_;
    }

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayoutConversion,
        "Tensor::GetArray");
    EnsureHostCurrent();

    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape_.size() && i < 4; i++) {
        dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(shape_[i]);
    }

    af_array_ = std::make_unique<af::array>(dims, ToArrayFireType(dtype_));
    void* host_data = HostData(host_buffer_);
    if (host_data) {
        af_array_->write(host_data, NumBytes(), afHost);
    }
    device_current_ = true;
    device_layout_ = TensorDeviceLayout::ArrayFireNative;

    return *af_array_;
}

af::array Tensor::GetArrayRowMajor2D() const {
    if (shape_.size() != 2) {
        return GetArray();
    }
    if (device_current_ && af_array_ && device_layout_ == TensorDeviceLayout::RowMajor2D) {
        return *af_array_;
    }

    if (device_current_ && af_array_ &&
        device_layout_ == TensorDeviceLayout::ArrayFireNative) {
        af::array reshaped = af::moddims(
            *af_array_,
            static_cast<dim_t>(shape_[1]),
            static_cast<dim_t>(shape_[0]));
        af::array converted = af::transpose(reshaped);
        converted.eval();
        af_array_ = std::make_unique<af::array>(std::move(converted));
        device_layout_ = TensorDeviceLayout::RowMajor2D;
        return *af_array_;
    }

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayoutConversion,
        "Tensor::GetArrayRowMajor2D");
    EnsureHostCurrent();

    af::dim4 swapped_dims(
        static_cast<dim_t>(shape_[1]),
        static_cast<dim_t>(shape_[0]),
        1,
        1);
    af::array arr(swapped_dims, ToArrayFireType(dtype_));
    void* host_data = HostData(host_buffer_);
    if (host_data) {
        arr.write(host_data, NumBytes(), afHost);
    }
    af_array_ = std::make_unique<af::array>(af::transpose(arr));
    device_current_ = true;
    device_layout_ = TensorDeviceLayout::RowMajor2D;
    return *af_array_;
}

af::array Tensor::GetArrayRowMajor3D() const {
    if (shape_.size() != 3) {
        return GetArray();
    }
    if (device_current_ && af_array_ && device_layout_ == TensorDeviceLayout::RowMajor3D) {
        return *af_array_;
    }

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayoutConversion,
        "Tensor::GetArrayRowMajor3D");
    EnsureHostCurrent();

    af::dim4 reversed_dims(
        static_cast<dim_t>(shape_[2]),
        static_cast<dim_t>(shape_[1]),
        static_cast<dim_t>(shape_[0]),
        1);
    af::array arr(reversed_dims, ToArrayFireType(dtype_));
    void* host_data = HostData(host_buffer_);
    if (host_data) {
        arr.write(host_data, NumBytes(), afHost);
    }
    af_array_ = std::make_unique<af::array>(af::reorder(arr, 2, 1, 0));
    device_current_ = true;
    device_layout_ = TensorDeviceLayout::RowMajor3D;
    return *af_array_;
}

af::array Tensor::GetSemanticArray() const {
    if (shape_.size() == 2) {
        return GetArrayRowMajor2D();
    }
    if (shape_.size() == 3) {
        return GetArrayRowMajor3D();
    }
    return GetArray();
}

void Tensor::SetFromArray(const af::array& arr) {
    shape_ = ShapeFromArrayFireDims(arr);
    dtype_ = FromArrayFireType(arr.type());
    host_buffer_.reset();
    af_array_ = std::make_unique<af::array>(arr);
    host_current_ = false;
    device_current_ = true;
    device_layout_ = TensorDeviceLayout::ArrayFireNative;
}

void Tensor::SetFromArrayRowMajor2D(const af::array& arr) {
    if (arr.dims(2) != 1 || arr.dims(3) != 1) {
        throw std::runtime_error("Tensor::SetFromArrayRowMajor2D: expected a 2D ArrayFire array");
    }

    shape_ = {
        static_cast<size_t>(arr.dims(0)),
        static_cast<size_t>(arr.dims(1))
    };
    dtype_ = FromArrayFireType(arr.type());
    host_buffer_.reset();
    af_array_ = std::make_unique<af::array>(arr);
    host_current_ = false;
    device_current_ = true;
    device_layout_ = TensorDeviceLayout::RowMajor2D;
}

void Tensor::SetFromArrayRowMajor3D(const af::array& arr) {
    if (arr.dims(3) != 1) {
        throw std::runtime_error("Tensor::SetFromArrayRowMajor3D: expected a 3D ArrayFire array");
    }

    shape_ = {
        static_cast<size_t>(arr.dims(0)),
        static_cast<size_t>(arr.dims(1)),
        static_cast<size_t>(arr.dims(2))
    };
    dtype_ = FromArrayFireType(arr.type());
    host_buffer_.reset();
    af_array_ = std::make_unique<af::array>(arr);
    host_current_ = false;
    device_current_ = true;
    device_layout_ = TensorDeviceLayout::RowMajor3D;
}

void Tensor::SetFromSemanticArray(
    const af::array& arr,
    std::vector<size_t> semantic_shape) {
    if (semantic_shape.size() > 4) {
        throw std::runtime_error(
            "Tensor::SetFromSemanticArray: ArrayFire supports at most 4 dimensions");
    }

    size_t expected_elements = 1;
    for (size_t dim : semantic_shape) {
        size_t next = 0;
        if (!tensor_utils::SafeMultiply(expected_elements, dim, next)) {
            throw std::overflow_error(
                "Tensor::SetFromSemanticArray: semantic shape element count overflow");
        }
        expected_elements = next;
    }
    if (expected_elements != static_cast<size_t>(arr.elements())) {
        throw std::runtime_error(
            "Tensor::SetFromSemanticArray: semantic shape does not match ArrayFire element count");
    }

    if (semantic_shape.size() == 2) {
        if (arr.dims(0) != static_cast<dim_t>(semantic_shape[0]) ||
            arr.dims(1) != static_cast<dim_t>(semantic_shape[1])) {
            throw std::runtime_error(
                "Tensor::SetFromSemanticArray: 2D semantic shape does not match ArrayFire dimensions");
        }
        SetFromArrayRowMajor2D(arr);
        return;
    }
    if (semantic_shape.size() == 3) {
        if (arr.dims(0) != static_cast<dim_t>(semantic_shape[0]) ||
            arr.dims(1) != static_cast<dim_t>(semantic_shape[1]) ||
            arr.dims(2) != static_cast<dim_t>(semantic_shape[2])) {
            throw std::runtime_error(
                "Tensor::SetFromSemanticArray: 3D semantic shape does not match ArrayFire dimensions");
        }
        SetFromArrayRowMajor3D(arr);
        return;
    }

    SetFromArray(arr);
    shape_ = std::move(semantic_shape);
}

Tensor Tensor::FromArrayRowMajor2D(const af::array& arr) {
    Tensor result;
    result.SetFromArrayRowMajor2D(arr);
    return result;
}

Tensor Tensor::FromArrayRowMajor3D(const af::array& arr) {
    Tensor result;
    result.SetFromArrayRowMajor3D(arr);
    return result;
}

Tensor Tensor::FromSemanticArray(
    const af::array& arr,
    std::vector<size_t> semantic_shape) {
    Tensor result;
    result.SetFromSemanticArray(arr, std::move(semantic_shape));
    return result;
}

void Tensor::EnsureHostCurrent() const {
    if (host_current_) {
        return;
    }
    if (!device_current_ || !af_array_) {
        throw std::runtime_error("Tensor host data is stale and no current device data is available");
    }

    const size_t bytes = NumBytes();
    if (bytes > 0 && !host_buffer_) {
        host_buffer_ = AllocateHostBuffer(bytes);
    }
    if (bytes > 0) {
        switch (device_layout_) {
            case TensorDeviceLayout::ArrayFireNative:
                af_array_->host(host_buffer_->Data());
                break;
            case TensorDeviceLayout::RowMajor2D: {
                af::array transposed = af::transpose(*af_array_);
                transposed.host(host_buffer_->Data());
                break;
            }
            case TensorDeviceLayout::RowMajor3D: {
                af::array reordered = af::reorder(*af_array_, 2, 1, 0);
                reordered.host(host_buffer_->Data());
                break;
            }
            case TensorDeviceLayout::None:
                throw std::runtime_error("Tensor host data is stale and device layout is unknown");
        }
    }
    host_current_ = true;
    if (GetArrayFireHostSyncObserver()) {
        const auto layout_name = [this]() {
            switch (device_layout_) {
                case TensorDeviceLayout::ArrayFireNative:
                    return "arrayfire_native";
                case TensorDeviceLayout::RowMajor2D:
                    return "row_major_2d";
                case TensorDeviceLayout::RowMajor3D:
                    return "row_major_3d";
                case TensorDeviceLayout::None:
                    return "none";
            }
            return "unknown";
        };
        ArrayFireHostSyncEvent event;
        event.operation_name = "Tensor::EnsureHostCurrent";
        event.reason_code = "tensor_host_materialization";
        event.tensor_shape = shape_;
        event.tensor_dtype =
            tensor_backend_observation::DataTypeName(dtype_);
        event.tensor_layout = layout_name();
        event.context =
            BuildTensorShapeContext("tensor", shape_) +
            "; dtype=" +
            event.tensor_dtype +
            "; layout=" +
            event.tensor_layout;
        event.bytes = static_cast<uint64_t>(bytes);
        NotifyArrayFireHostSync(std::move(event));
    }
}

void Tensor::MarkHostModified() const {
    EnsureHostCurrent();
    device_current_ = false;
    device_layout_ = TensorDeviceLayout::None;
}

void Tensor::ClearDeviceCache() const {
    af_array_.reset();
    device_current_ = false;
    device_layout_ = TensorDeviceLayout::None;
}
#endif

size_t Tensor::NumElements() const {
    size_t count = 1;
    for (size_t dim : shape_) {
        size_t next = 0;
        if (!tensor_utils::SafeMultiply(count, dim, next)) {
            throw std::overflow_error("Tensor::NumElements: integer overflow in dimension product");
        }
        count = next;
    }
    return count;
}

size_t Tensor::NumBytes() const {
    size_t bytes = 0;
    if (!tensor_utils::SafeMultiply(NumElements(), tensor_utils::ElementSize(dtype_), bytes)) {
        throw std::overflow_error("Tensor::NumBytes: integer overflow in byte count");
    }
    return bytes;
}

const void* Tensor::ReadData() const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    EnsureHostCurrent();
#endif
    return HostData(host_buffer_);
}

void* Tensor::MutableData() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    MarkHostModified();
#endif
    return HostData(host_buffer_);
}

void* Tensor::Data() {
    return MutableData();
}

const void* Tensor::Data() const {
    return ReadData();
}

Tensor Tensor::Zeros(const std::vector<size_t>& shape, DataType dtype) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // GPU-accelerated zeros creation
    try {
        // Convert shape to af::dim4
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < shape.size() && i < 4; i++) {
            dims[static_cast<unsigned int>(i)] = static_cast<unsigned int>(shape[i]);
        }

        // Create ArrayFire array filled with zeros
        af::array zeros_arr = af::constant(0.0, dims, ToArrayFireType(dtype));

        return Tensor(zeros_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreCreationArrayFireFallback(
            "Tensor::Zeros",
            shape,
            dtype,
            "op=zeros",
            e.what());
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
            dims[static_cast<unsigned int>(i)] = static_cast<unsigned int>(shape[i]);
        }

        // Create ArrayFire array filled with ones
        af::array ones_arr = af::constant(1.0, dims, ToArrayFireType(dtype));

        return Tensor(ones_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreCreationArrayFireFallback(
            "Tensor::Ones",
            shape,
            dtype,
            "op=ones",
            e.what());
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
            dims[static_cast<unsigned int>(i)] = static_cast<unsigned int>(shape[i]);
        }

        // Create ArrayFire array with random values [0, 1)
        af::array random_arr = af::randu(dims, ToArrayFireType(dtype));

        return Tensor(random_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreCreationArrayFireFallback(
            "Tensor::Random",
            shape,
            dtype,
            "op=random",
            e.what());
    }
#endif

    // CPU fallback
    Tensor t(shape, dtype);

    auto& engine = CpuRandomEngine();
    FillRandomCpu(t, engine);

    return t;
}

Tensor Tensor::RandomSeeded(const std::vector<size_t>& shape, uint64_t seed, DataType dtype) {
    Tensor t(shape, dtype);
    std::seed_seq seed_sequence{
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32),
    };
    std::mt19937 engine(seed_sequence);
    FillRandomCpu(t, engine);

    return t;
}

Tensor Tensor::RangeN(const std::vector<size_t>& shape, DataType dtype) {
    Tensor t(shape, dtype);
    const size_t num_elements = t.NumElements();

    switch (dtype) {
        case DataType::Float32: {
            float* data = t.Data<float>();
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<float>(i);
            }
            break;
        }
        case DataType::Float64: {
            double* data = t.Data<double>();
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<double>(i);
            }
            break;
        }
        case DataType::Int32: {
            int32_t* data = t.Data<int32_t>();
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<int32_t>(i);
            }
            break;
        }
        case DataType::Int64: {
            int64_t* data = t.Data<int64_t>();
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<int64_t>(i);
            }
            break;
        }
        case DataType::UInt8: {
            uint8_t* data = t.Data<uint8_t>();
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<uint8_t>(i % 256);
            }
            break;
        }
    }

    return t;
}

Tensor Tensor::Reshape(const std::vector<size_t>& new_shape) const {
    size_t new_elements = 1;
    for (size_t dim : new_shape) {
        size_t next = 0;
        if (!tensor_utils::SafeMultiply(new_elements, dim, next)) {
            throw std::overflow_error("Tensor::Reshape: integer overflow in dimension product");
        }
        new_elements = next;
    }

    if (new_elements != NumElements()) {
        throw std::runtime_error("Tensor::Reshape: new shape must have same number of elements");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (device_current_ && af_array_ &&
        new_shape.size() <= 4 &&
        device_layout_ == TensorDeviceLayout::ArrayFireNative) {
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < new_shape.size(); i++) {
            dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(new_shape[i]);
        }

        Tensor result;
        result.shape_ = new_shape;
        result.dtype_ = dtype_;
        result.device_ = device_;
        result.af_array_ = std::make_unique<af::array>(af::moddims(*af_array_, dims));
        result.host_current_ = false;
        result.device_current_ = true;
        result.device_layout_ = TensorDeviceLayout::ArrayFireNative;
        return result;
    }

    if (device_current_ && af_array_ &&
        new_elements > 0 &&
        new_shape.size() <= 4 &&
        (device_layout_ == TensorDeviceLayout::RowMajor2D ||
         device_layout_ == TensorDeviceLayout::RowMajor3D)) {
        try {
            const af::array row_major_linear =
                device_layout_ == TensorDeviceLayout::RowMajor2D
                    ? af::flat(af::transpose(*af_array_))
                    : af::flat(af::reorder(*af_array_, 2, 1, 0));

            Tensor result;
            result.shape_ = new_shape;
            result.dtype_ = dtype_;
            result.device_ = device_;
            result.host_current_ = false;
            result.device_current_ = true;

            if (new_shape.size() == 2) {
                const af::dim4 swapped_dims(
                    static_cast<dim_t>(new_shape[1]),
                    static_cast<dim_t>(new_shape[0]),
                    1,
                    1);
                result.af_array_ = std::make_unique<af::array>(
                    af::transpose(af::moddims(row_major_linear, swapped_dims)));
                result.device_layout_ = TensorDeviceLayout::RowMajor2D;
            } else if (new_shape.size() == 3) {
                const af::dim4 reversed_dims(
                    static_cast<dim_t>(new_shape[2]),
                    static_cast<dim_t>(new_shape[1]),
                    static_cast<dim_t>(new_shape[0]),
                    1);
                result.af_array_ = std::make_unique<af::array>(
                    af::reorder(
                        af::moddims(row_major_linear, reversed_dims),
                        2,
                        1,
                        0));
                result.device_layout_ = TensorDeviceLayout::RowMajor3D;
            } else {
                af::dim4 dims(1, 1, 1, 1);
                for (size_t i = 0; i < new_shape.size(); ++i) {
                    dims[static_cast<unsigned int>(i)] =
                        static_cast<dim_t>(new_shape[i]);
                }
                result.af_array_ = std::make_unique<af::array>(
                    af::moddims(row_major_linear, dims));
                result.device_layout_ = TensorDeviceLayout::ArrayFireNative;
            }
            return result;
        } catch (const af::exception& e) {
            RecordTensorCoreArrayFireFallback(
                "Tensor::Reshape",
                *this,
                new_shape,
                "op=reshape;layout=row_major_cross_rank",
                e.what());
        }
    }
#endif

    return Tensor(new_shape, Data(), dtype_);
}

Tensor Tensor::Transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Tensor::Transpose currently only supports 2D tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (dtype_ == DataType::Float32 || dtype_ == DataType::Float64) {
        try {
            return Tensor::FromArrayRowMajor2D(af::transpose(GetArrayRowMajor2D()));
        } catch (const af::exception& e) {
            RecordTensorCoreArrayFireFallback(
                "Tensor::Transpose",
                *this,
                {shape_[1], shape_[0]},
                "op=transpose;rank=2",
                e.what());
        }
    }
#endif

    const size_t rows = shape_[0];
    const size_t cols = shape_[1];
    Tensor result({cols, rows}, dtype_);

    switch (dtype_) {
        case DataType::Float32: {
            const float* src = Data<float>();
            float* dst = result.Data<float>();
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            break;
        }
        case DataType::Float64: {
            const double* src = Data<double>();
            double* dst = result.Data<double>();
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* src = Data<int32_t>();
            int32_t* dst = result.Data<int32_t>();
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* src = Data<int64_t>();
            int64_t* dst = result.Data<int64_t>();
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* src = Data<uint8_t>();
            uint8_t* dst = result.Data<uint8_t>();
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            break;
        }
    }

    return result;
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
    if (shape_.size() == 2 &&
        (dtype_ == DataType::Float32 || dtype_ == DataType::Float64)) {
        try {
            return Tensor::FromArrayRowMajor2D(GetArrayRowMajor2D() + other.GetArrayRowMajor2D());
        } catch (const af::exception& e) {
            RecordTensorCoreBinaryArrayFireFallback(
                "Tensor::operator+",
                *this,
                other,
                "op=add;layout=row_major_2d",
                e.what());
        }
    }

    try {
        af::array a_arr = GetArray();
        af::array b_arr = other.GetArray();

        // Perform GPU-accelerated addition
        af::array result_arr = a_arr + b_arr;

        return Tensor(result_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreBinaryArrayFireFallback(
            "Tensor::operator+",
            *this,
            other,
            "op=add;layout=arrayfire_native",
            e.what());
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
        af::array a_arr = GetArray();
        af::array b_arr = other.GetArray();

        af::array result_arr = a_arr - b_arr;

        return Tensor(result_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreBinaryArrayFireFallback(
            "Tensor::operator-",
            *this,
            other,
            "op=subtract;layout=arrayfire_native",
            e.what());
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
    if (shape_.size() == 2 &&
        (dtype_ == DataType::Float32 || dtype_ == DataType::Float64)) {
        try {
            return Tensor::FromArrayRowMajor2D(GetArrayRowMajor2D() * other.GetArrayRowMajor2D());
        } catch (const af::exception& e) {
            RecordTensorCoreBinaryArrayFireFallback(
                "Tensor::operator*",
                *this,
                other,
                "op=multiply;layout=row_major_2d",
                e.what());
        }
    }

    try {
        af::array a_arr = GetArray();
        af::array b_arr = other.GetArray();

        af::array result_arr = a_arr * b_arr;

        return Tensor(result_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreBinaryArrayFireFallback(
            "Tensor::operator*",
            *this,
            other,
            "op=multiply;layout=arrayfire_native",
            e.what());
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
        af::array a_arr = GetArray();
        af::array b_arr = other.GetArray();

        af::array result_arr = a_arr / b_arr;

        return Tensor(result_arr);
    } catch (const af::exception& e) {
        RecordTensorCoreBinaryArrayFireFallback(
            "Tensor::operator/",
            *this,
            other,
            "op=divide;layout=arrayfire_native",
            e.what());
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
