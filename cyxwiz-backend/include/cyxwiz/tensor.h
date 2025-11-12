#pragma once

#include "api_export.h"
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

    // Data access
    void* Data();
    const void* Data() const;
    template<typename T>
    T* Data() { return static_cast<T*>(Data()); }
    template<typename T>
    const T* Data() const { return static_cast<const T*>(Data()); }

    // Device management
    void ToDevice(Device* device);
    void ToCPU();
    Device* GetDevice() const { return device_; }

    // Operations
    Tensor Clone() const;
    Tensor Reshape(const std::vector<size_t>& new_shape) const;
    Tensor Transpose() const;

    // Math operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // Static factory methods
    static Tensor Zeros(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    static Tensor Ones(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    static Tensor Random(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);
    static Tensor RangeN(const std::vector<size_t>& shape, DataType dtype = DataType::Float32);

private:
    std::vector<size_t> shape_;
    DataType dtype_;
    Device* device_;
    void* data_;
    bool owns_data_;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array* af_array_;
#endif
};

} // namespace cyxwiz
