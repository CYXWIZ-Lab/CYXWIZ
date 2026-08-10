#pragma once

#include "api_export.h"
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <memory>
#include <initializer_list>

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace af {
    class array;
}
#endif

namespace cyxwiz {
    class Device;
}

namespace cyxwiz {

class TensorHostBuffer;

enum class DataType {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    UInt8 = 4
};

class CYXWIZ_API Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    Tensor(const std::vector<size_t>& shape, const void* data, DataType dtype = DataType::Float32);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Construct from ArrayFire array (takes ownership)
    explicit Tensor(const af::array& arr);
#endif
    ~Tensor();

    // Copy/Move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Shape and metadata
    const std::vector<size_t>& Shape() const { return shape_; }
    size_t NumElements() const;
    size_t NumBytes() const;
    DataType GetDataType() const { return dtype_; }
    int NumDimensions() const { return static_cast<int>(shape_.size()); }

    // Explicit host access. ReadData preserves a current device copy;
    // MutableData invalidates it because the caller may modify host memory.
    const void* ReadData() const;
    void* MutableData();
    template<typename T>
    const T* ReadData() const { return static_cast<const T*>(ReadData()); }
    template<typename T>
    T* MutableData() { return static_cast<T*>(MutableData()); }

    // Compatibility aliases. New code should use explicit host access.
    void* Data();
    const void* Data() const;
    template<typename T>
    T* Data() { return static_cast<T*>(Data()); }
    template<typename T>
    const T* Data() const { return static_cast<const T*>(Data()); }

    // Device metadata
    Device* GetDevice() const { return device_; }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire array access (creates if needed, syncs from CPU data)
    af::array GetArray() const;
    // 2D row-major Tensor view as semantic ArrayFire [rows, cols].
    af::array GetArrayRowMajor2D() const;
    // 3D row-major Tensor view as semantic ArrayFire [dim0, dim1, dim2].
    af::array GetArrayRowMajor3D() const;
    // Rank-aware semantic view: row-major for rank 2/3, native otherwise.
    af::array GetSemanticArray() const;
    // Set from ArrayFire array, keeping device data resident until host data is requested.
    void SetFromArray(const af::array& arr);
    // Set from semantic row-major ArrayFire views, keeping device data resident.
    void SetFromArrayRowMajor2D(const af::array& arr);
    void SetFromArrayRowMajor3D(const af::array& arr);
    // Store a semantic ArrayFire result with an explicit logical shape.
    void SetFromSemanticArray(
        const af::array& arr,
        std::vector<size_t> semantic_shape);
    // Build a row-major 2D Tensor from semantic ArrayFire [rows, cols].
    static Tensor FromArrayRowMajor2D(const af::array& arr);
    // Build a row-major 3D Tensor from semantic ArrayFire [dim0, dim1, dim2].
    static Tensor FromArrayRowMajor3D(const af::array& arr);
    static Tensor FromSemanticArray(
        const af::array& arr,
        std::vector<size_t> semantic_shape);
#endif

    // Operations
    Tensor Clone() const;
    Tensor Reshape(const std::vector<size_t>& new_shape) const;
    Tensor View(const std::vector<size_t>& new_shape) const;
    Tensor Squeeze(int dim = -1) const;
    Tensor Unsqueeze(int dim) const;
    Tensor Flatten() const;
    Tensor Flatten(int start_dim, int end_dim = -1) const;
    Tensor Transpose() const;
    Tensor Transpose(int dim0, int dim1) const;
    Tensor Permute(const std::vector<int>& dims) const;

    // Indexing and slicing
    float At(size_t idx) const;
    float At(size_t i, size_t j) const;
    float At(size_t i, size_t j, size_t k) const;
    float At(size_t i, size_t j, size_t k, size_t l) const;
    void Set(size_t idx, float value);
    void Set(size_t i, size_t j, float value);
    void Set(size_t i, size_t j, size_t k, float value);
    void Set(size_t i, size_t j, size_t k, size_t l, float value);
    template<typename T>
    T AtAs(size_t idx) const {
        EnsureDataType<T>();
        if (idx >= NumElements()) {
            throw std::out_of_range("Tensor::AtAs: index out of range");
        }
        return ReadData<T>()[idx];
    }
    template<typename T>
    T AtAs(size_t i, size_t j) const {
        return AtAs<T>(CheckedTypedLinearIndex({i, j}));
    }
    template<typename T>
    T AtAs(size_t i, size_t j, size_t k) const {
        return AtAs<T>(CheckedTypedLinearIndex({i, j, k}));
    }
    template<typename T>
    T AtAs(size_t i, size_t j, size_t k, size_t l) const {
        return AtAs<T>(CheckedTypedLinearIndex({i, j, k, l}));
    }
    template<typename T>
    void SetAs(size_t idx, T value) {
        EnsureDataType<T>();
        if (idx >= NumElements()) {
            throw std::out_of_range("Tensor::SetAs: index out of range");
        }
        MutableData<T>()[idx] = value;
    }
    template<typename T>
    void SetAs(size_t i, size_t j, T value) {
        SetAs<T>(CheckedTypedLinearIndex({i, j}), value);
    }
    template<typename T>
    void SetAs(size_t i, size_t j, size_t k, T value) {
        SetAs<T>(CheckedTypedLinearIndex({i, j, k}), value);
    }
    template<typename T>
    void SetAs(size_t i, size_t j, size_t k, size_t l, T value) {
        SetAs<T>(CheckedTypedLinearIndex({i, j, k, l}), value);
    }
    Tensor Slice(int dim, int start, int end = -1, int step = 1) const;
    Tensor IndexSelect(int dim, const std::vector<int>& indices) const;

    // Concatenation and splitting
    static Tensor Cat(const std::vector<Tensor>& tensors, int dim = 0);
    static Tensor Stack(const std::vector<Tensor>& tensors, int dim = 0);
    std::vector<Tensor> Split(int split_size, int dim = 0) const;
    std::vector<Tensor> Split(const std::vector<int>& sizes, int dim = 0) const;
    std::vector<Tensor> Chunk(int chunks, int dim = 0) const;

    // Reductions
    Tensor Sum() const;
    Tensor Sum(int dim, bool keepdim = false) const;
    Tensor Mean() const;
    Tensor Mean(int dim, bool keepdim = false) const;
    Tensor Max() const;
    Tensor Max(int dim, bool keepdim = false) const;
    Tensor Min() const;
    Tensor Min(int dim, bool keepdim = false) const;
    Tensor Prod() const;
    Tensor Prod(int dim, bool keepdim = false) const;
    Tensor Var() const;
    Tensor Var(int dim, bool keepdim = false) const;
    Tensor Std() const;
    Tensor Std(int dim, bool keepdim = false) const;

    // Math operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    Tensor Pow(float exponent) const;
    Tensor Pow(const Tensor& exponent) const;
    Tensor Sqrt() const;
    Tensor Exp() const;
    Tensor Log() const;
    Tensor Abs() const;
    Tensor Sign() const;
    Tensor Clip(float min_val, float max_val) const;
    Tensor operator-() const;

    // Linear algebra
    Tensor Dot(const Tensor& other) const;
    Tensor BatchMatMul(const Tensor& other) const;

    // Comparisons return UInt8 masks with 1 for true and 0 for false.
    Tensor operator>(const Tensor& other) const;
    Tensor operator>(float scalar) const;
    Tensor operator>=(const Tensor& other) const;
    Tensor operator>=(float scalar) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator<(float scalar) const;
    Tensor operator<=(const Tensor& other) const;
    Tensor operator<=(float scalar) const;
    Tensor operator==(const Tensor& other) const;
    Tensor operator==(float scalar) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator!=(float scalar) const;

    // Logical operations treat zero as false and nonzero as true.
    Tensor operator&&(const Tensor& other) const;
    Tensor operator||(const Tensor& other) const;
    Tensor operator!() const;

    // Broadcasting
    static bool IsBroadcastable(const std::vector<size_t>& shape1,
                                const std::vector<size_t>& shape2);
    static std::vector<size_t> BroadcastShape(const std::vector<size_t>& shape1,
                                              const std::vector<size_t>& shape2);
    Tensor BroadcastTo(const std::vector<size_t>& target_shape) const;
    Tensor Expand(const std::vector<size_t>& target_shape) const;

    // Static factory methods
    static Tensor Zeros(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    static Tensor Ones(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    static Tensor Random(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    static Tensor RandomSeeded(const std::vector<size_t>& shape, uint64_t seed,
                               DataType dtype = DataType::Float32);
    static Tensor RangeN(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);

private:
    template<typename>
    struct AlwaysFalse : std::false_type {};

    template<typename T>
    static constexpr DataType NativeDataType() {
        if constexpr (std::is_same_v<T, float>) {
            return DataType::Float32;
        } else if constexpr (std::is_same_v<T, double>) {
            return DataType::Float64;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return DataType::Int32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return DataType::Int64;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return DataType::UInt8;
        } else {
            static_assert(AlwaysFalse<T>::value, "Unsupported Tensor accessor type");
        }
    }

    template<typename T>
    void EnsureDataType() const {
        if (dtype_ != NativeDataType<T>()) {
            throw std::runtime_error("Tensor typed accessor data type mismatch");
        }
    }

    size_t CheckedTypedLinearIndex(std::initializer_list<size_t> indices) const {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Tensor typed accessor rank mismatch");
        }

        size_t stride = NumElements();
        size_t linear = 0;
        size_t axis = 0;
        for (size_t index : indices) {
            const size_t dim = shape_[axis];
            if (index >= dim) {
                throw std::out_of_range("Tensor typed accessor index out of range");
            }
            stride /= dim;
            linear += index * stride;
            axis++;
        }
        return linear;
    }

    std::vector<size_t> shape_;
    DataType dtype_;
    Device* device_;
    mutable std::unique_ptr<TensorHostBuffer> host_buffer_;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    enum class TensorDeviceLayout {
        None,
        ArrayFireNative,
        RowMajor2D,
        RowMajor3D
    };

    mutable std::unique_ptr<af::array> af_array_;
    mutable bool host_current_;
    mutable bool device_current_;
    mutable TensorDeviceLayout device_layout_ = TensorDeviceLayout::None;

    void EnsureHostCurrent() const;
    void MarkHostModified() const;
    void ClearDeviceCache() const;
#endif
};

} // namespace cyxwiz
