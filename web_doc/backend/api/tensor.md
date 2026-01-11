# Tensor API Reference

The `Tensor` class is the fundamental data structure in cyxwiz-backend, providing GPU-accelerated multi-dimensional array operations powered by ArrayFire.

## Class Definition

```cpp
namespace cyxwiz {

class CYXWIZ_API Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<float>& data, const std::vector<int>& shape);
    Tensor(const std::vector<double>& data, const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::Float32);

    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    // Shape and dimensions
    std::vector<int> Shape() const;
    int NumDimensions() const;
    int64_t NumElements() const;
    int Dim(int index) const;

    // Data access
    DataType DType() const;
    std::vector<float> ToVector() const;
    float* Data();
    const float* Data() const;

    // Reshape operations
    Tensor Reshape(const std::vector<int>& new_shape) const;
    Tensor Flatten() const;
    Tensor Squeeze(int dim = -1) const;
    Tensor Unsqueeze(int dim) const;
    Tensor Transpose(int dim1 = -2, int dim2 = -1) const;
    Tensor Permute(const std::vector<int>& dims) const;

    // Arithmetic operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // Scalar operators
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // In-place operators
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Comparison
    Tensor operator==(const Tensor& other) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator>(const Tensor& other) const;
    Tensor operator<=(const Tensor& other) const;
    Tensor operator>=(const Tensor& other) const;

    // Indexing
    Tensor operator[](int index) const;
    Tensor Slice(int dim, int start, int end) const;
    Tensor Index(const std::vector<int>& indices) const;

    // Reduction operations
    Tensor Sum(int dim = -1, bool keepdim = false) const;
    Tensor Mean(int dim = -1, bool keepdim = false) const;
    Tensor Max(int dim = -1, bool keepdim = false) const;
    Tensor Min(int dim = -1, bool keepdim = false) const;
    Tensor ArgMax(int dim = -1, bool keepdim = false) const;
    Tensor ArgMin(int dim = -1, bool keepdim = false) const;

    // Math operations
    Tensor Abs() const;
    Tensor Sqrt() const;
    Tensor Exp() const;
    Tensor Log() const;
    Tensor Pow(float exponent) const;
    Tensor Sin() const;
    Tensor Cos() const;
    Tensor Tanh() const;

    // Matrix operations
    Tensor MatMul(const Tensor& other) const;
    Tensor Dot(const Tensor& other) const;

    // Device operations
    Tensor ToDevice(DeviceType device) const;
    Tensor ToCPU() const;
    Tensor ToGPU() const;
    bool IsOnGPU() const;
    DeviceType Device() const;

    // Gradient support
    void RequiresGrad(bool requires);
    bool RequiresGrad() const;
    Tensor Grad() const;
    void ZeroGrad();
    void Backward();

    // Cloning
    Tensor Clone() const;
    Tensor Detach() const;

    // Utility
    void Print() const;
    std::string ToString() const;
    bool IsContiguous() const;
    Tensor Contiguous() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cyxwiz
```

## Data Types

```cpp
enum class DataType {
    Float16,   // Half precision
    Float32,   // Single precision (default)
    Float64,   // Double precision
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    Bool
};
```

## Construction

### From Data

```cpp
// From vector with shape
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
Tensor t1(data, {2, 3});  // 2x3 matrix

// From double vector
std::vector<double> ddata = {1.0, 2.0, 3.0};
Tensor t2(ddata, {3});  // 1D vector
```

### From Shape (Uninitialized)

```cpp
// Create tensor with specific shape
Tensor t3({32, 64, 128});  // 3D tensor, uninitialized

// With specific dtype
Tensor t4({100, 100}, DataType::Float64);
```

### Factory Functions

```cpp
namespace cyxwiz {

// Zeros
Tensor Zeros(const std::vector<int>& shape, DataType dtype = DataType::Float32);

// Ones
Tensor Ones(const std::vector<int>& shape, DataType dtype = DataType::Float32);

// Random uniform [0, 1)
Tensor Rand(const std::vector<int>& shape, DataType dtype = DataType::Float32);

// Random normal (mean=0, std=1)
Tensor Randn(const std::vector<int>& shape, DataType dtype = DataType::Float32);

// Range
Tensor Arange(float start, float end, float step = 1.0f);

// Linspace
Tensor Linspace(float start, float end, int num);

// Identity matrix
Tensor Eye(int n, DataType dtype = DataType::Float32);

// Full with value
Tensor Full(const std::vector<int>& shape, float value, DataType dtype = DataType::Float32);

// From file
Tensor LoadTensor(const std::string& path);

// Save to file
void SaveTensor(const Tensor& tensor, const std::string& path);

}
```

## Usage Examples

### Basic Operations

```cpp
#include <cyxwiz/tensor.h>

using namespace cyxwiz;

// Create tensors
Tensor a = Randn({100, 50});
Tensor b = Randn({50, 30});

// Matrix multiplication
Tensor c = a.MatMul(b);  // Shape: (100, 30)

// Element-wise operations
Tensor d = a * 2.0f + 1.0f;

// Reductions
Tensor sum = a.Sum();        // Scalar
Tensor col_sum = a.Sum(0);   // Sum along dim 0, shape: (50,)
Tensor row_sum = a.Sum(1);   // Sum along dim 1, shape: (100,)

// Statistics
Tensor mean = a.Mean();
Tensor max_val = a.Max();
Tensor min_val = a.Min();
```

### Shape Manipulation

```cpp
Tensor t = Randn({2, 3, 4, 5});

// Reshape
Tensor reshaped = t.Reshape({6, 20});

// Flatten
Tensor flat = t.Flatten();  // Shape: (120,)

// Transpose
Tensor transposed = t.Transpose(-2, -1);  // Swap last two dims

// Permute
Tensor permuted = t.Permute({0, 2, 1, 3});  // Reorder dimensions

// Squeeze/Unsqueeze
Tensor squeezed = Randn({1, 10, 1}).Squeeze();  // Shape: (10,)
Tensor unsqueezed = Randn({10}).Unsqueeze(0);   // Shape: (1, 10)
```

### Indexing and Slicing

```cpp
Tensor t = Randn({10, 20, 30});

// Single index
Tensor first = t[0];  // Shape: (20, 30)

// Slice
Tensor sliced = t.Slice(0, 2, 5);  // t[2:5, :, :], shape: (3, 20, 30)

// Multiple slices
Tensor multi = t.Slice(0, 0, 5).Slice(1, 10, 15);  // t[0:5, 10:15, :]
```

### Device Management

```cpp
// Create on default device
Tensor cpu_tensor = Randn({1000, 1000});

// Move to GPU
Tensor gpu_tensor = cpu_tensor.ToGPU();

// Move back to CPU
Tensor back_to_cpu = gpu_tensor.ToCPU();

// Check device
if (gpu_tensor.IsOnGPU()) {
    std::cout << "Tensor is on GPU" << std::endl;
}

// Explicit device
Tensor cuda_tensor = cpu_tensor.ToDevice(DeviceType::CUDA);
```

### Gradient Computation

```cpp
// Create tensor with gradient tracking
Tensor x = Randn({10, 5});
x.RequiresGrad(true);

// Forward pass
Tensor y = x.MatMul(Randn({5, 3}));
Tensor loss = y.Sum();

// Backward pass
loss.Backward();

// Access gradient
Tensor grad = x.Grad();

// Zero gradients
x.ZeroGrad();

// Detach from computation graph
Tensor detached = y.Detach();
```

## Broadcasting Rules

CyxWiz follows NumPy/PyTorch broadcasting rules:

1. If tensors have different numbers of dimensions, prepend 1s to smaller tensor's shape
2. Dimensions are compatible if they are equal or one of them is 1
3. The output shape is the maximum of each dimension

```cpp
Tensor a({3, 4, 5});
Tensor b({4, 5});     // Broadcasts to (1, 4, 5)
Tensor c = a + b;     // Result shape: (3, 4, 5)

Tensor d({3, 1, 5});
Tensor e({1, 4, 1});
Tensor f = d * e;     // Result shape: (3, 4, 5)
```

## Memory Management

### RAII Pattern

```cpp
{
    Tensor t = Randn({1000, 1000});  // Allocates memory
    // ... use tensor
}  // Memory automatically freed when t goes out of scope
```

### Clone vs View

```cpp
Tensor original = Randn({10, 10});

// Clone creates a copy (independent data)
Tensor cloned = original.Clone();
cloned[0] = Zeros({10});  // Does not affect original

// Reshape creates a view (shared data in some cases)
Tensor view = original.Reshape({100});
// Note: Contiguous tensors may share memory
```

### Contiguous Memory

```cpp
Tensor t = Randn({10, 20});
Tensor transposed = t.Transpose();

// Check if contiguous
if (!transposed.IsContiguous()) {
    // Make contiguous copy
    Tensor contiguous = transposed.Contiguous();
}
```

## Python Bindings

```python
import pycyxwiz as cyx

# Create tensor
t = cyx.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])

# Factory functions
zeros = cyx.zeros([10, 10])
ones = cyx.ones([5, 5])
rand = cyx.rand([100, 50])
randn = cyx.randn([64, 128])

# Operations
result = t + t * 2.0
matmul = cyx.matmul(t, t.T)

# Device
gpu_t = t.to_gpu()
cpu_t = gpu_t.to_cpu()

# NumPy conversion
import numpy as np
np_array = t.numpy()
from_np = cyx.from_numpy(np_array)
```

## Performance Tips

1. **Batch operations**: Operate on batches rather than individual samples
2. **Minimize transfers**: Keep data on GPU when possible
3. **Use in-place**: Use `+=`, `-=` etc. when the original is no longer needed
4. **Contiguous memory**: Ensure tensors are contiguous before heavy computation
5. **Appropriate dtype**: Use Float32 for most cases, Float16 for large models

---

**Next**: [Device API](device.md) | [Layer API](layers.md)
