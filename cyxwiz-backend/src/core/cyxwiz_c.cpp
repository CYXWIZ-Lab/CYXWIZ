#include "cyxwiz/cyxwiz_c.h"
#include "cyxwiz/cyxwiz.h"
#include "cyxwiz/linear_algebra.h"
#include <string>
#include <cstring>
#include <vector>

// Thread-local error storage
thread_local std::string g_last_error;

static void SetLastError(const std::string& error) {
    g_last_error = error;
}

static void ClearLastError() {
    g_last_error.clear();
}

static bool TryConvertDataType(CyxWizDataType dtype, cyxwiz::DataType& out) {
    switch (dtype) {
        case CYXWIZ_DTYPE_FLOAT32:
            out = cyxwiz::DataType::Float32;
            return true;
        case CYXWIZ_DTYPE_FLOAT64:
            out = cyxwiz::DataType::Float64;
            return true;
        case CYXWIZ_DTYPE_INT32:
            out = cyxwiz::DataType::Int32;
            return true;
        case CYXWIZ_DTYPE_INT64:
            out = cyxwiz::DataType::Int64;
            return true;
        case CYXWIZ_DTYPE_UINT8:
            out = cyxwiz::DataType::UInt8;
            return true;
    }

    SetLastError("Invalid tensor data type");
    return false;
}

static bool TryBuildShape(const size_t* shape, size_t ndim, std::vector<size_t>& out) {
    if (ndim > 0 && !shape) {
        SetLastError("Null tensor shape pointer");
        return false;
    }

    if (ndim == 0) {
        out.clear();
        return true;
    }

    out.assign(shape, shape + ndim);
    return true;
}

// Initialization
bool cyxwiz_initialize(void) {
    try {
        return cyxwiz::Initialize();
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return false;
    }
}

void cyxwiz_shutdown(void) {
    cyxwiz::Shutdown();
}

const char* cyxwiz_get_version(void) {
    return cyxwiz::GetVersionString();
}

// Device Management
CyxWizDevice* cyxwiz_device_create(CyxWizDeviceType type, int device_id) {
    try {
        auto* device = new cyxwiz::Device(
            static_cast<cyxwiz::DeviceType>(type),
            device_id
        );
        return reinterpret_cast<CyxWizDevice*>(device);
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

void cyxwiz_device_destroy(CyxWizDevice* device) {
    delete reinterpret_cast<cyxwiz::Device*>(device);
}

void cyxwiz_device_set_active(CyxWizDevice* device) {
    if (device) {
        reinterpret_cast<cyxwiz::Device*>(device)->SetActive();
    }
}

int cyxwiz_device_get_count(CyxWizDeviceType type) {
    auto devices = cyxwiz::Device::GetAvailableDevices();
    int count = 0;
    for (const auto& dev : devices) {
        if (static_cast<int>(dev.type) == type) {
            count++;
        }
    }
    return count;
}

// Tensor Operations
CyxWizTensor* cyxwiz_tensor_create(const size_t* shape, size_t ndim, CyxWizDataType dtype) {
    try {
        std::vector<size_t> shape_vec;
        cyxwiz::DataType tensor_dtype;
        if (!TryBuildShape(shape, ndim, shape_vec) || !TryConvertDataType(dtype, tensor_dtype)) {
            return nullptr;
        }

        auto* tensor = new cyxwiz::Tensor(shape_vec, tensor_dtype);
        return reinterpret_cast<CyxWizTensor*>(tensor);
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_create_with_data(const size_t* shape, size_t ndim,
                                               const void* data, CyxWizDataType dtype) {
    try {
        if (!data) {
            SetLastError("Null tensor data pointer");
            return nullptr;
        }

        std::vector<size_t> shape_vec;
        cyxwiz::DataType tensor_dtype;
        if (!TryBuildShape(shape, ndim, shape_vec) || !TryConvertDataType(dtype, tensor_dtype)) {
            return nullptr;
        }

        auto* tensor = new cyxwiz::Tensor(shape_vec, data, tensor_dtype);
        return reinterpret_cast<CyxWizTensor*>(tensor);
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

void cyxwiz_tensor_destroy(CyxWizTensor* tensor) {
    delete reinterpret_cast<cyxwiz::Tensor*>(tensor);
}

CyxWizTensor* cyxwiz_tensor_zeros(const size_t* shape, size_t ndim, CyxWizDataType dtype) {
    try {
        std::vector<size_t> shape_vec;
        cyxwiz::DataType tensor_dtype;
        if (!TryBuildShape(shape, ndim, shape_vec) || !TryConvertDataType(dtype, tensor_dtype)) {
            return nullptr;
        }

        auto tensor = cyxwiz::Tensor::Zeros(shape_vec, tensor_dtype);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(tensor)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_ones(const size_t* shape, size_t ndim, CyxWizDataType dtype) {
    try {
        std::vector<size_t> shape_vec;
        cyxwiz::DataType tensor_dtype;
        if (!TryBuildShape(shape, ndim, shape_vec) || !TryConvertDataType(dtype, tensor_dtype)) {
            return nullptr;
        }

        auto tensor = cyxwiz::Tensor::Ones(shape_vec, tensor_dtype);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(tensor)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_random(const size_t* shape, size_t ndim, CyxWizDataType dtype) {
    try {
        std::vector<size_t> shape_vec;
        cyxwiz::DataType tensor_dtype;
        if (!TryBuildShape(shape, ndim, shape_vec) || !TryConvertDataType(dtype, tensor_dtype)) {
            return nullptr;
        }

        auto tensor = cyxwiz::Tensor::Random(shape_vec, tensor_dtype);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(tensor)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

size_t cyxwiz_tensor_num_elements(const CyxWizTensor* tensor) {
    if (!tensor) return 0;
    return reinterpret_cast<const cyxwiz::Tensor*>(tensor)->NumElements();
}

size_t cyxwiz_tensor_num_bytes(const CyxWizTensor* tensor) {
    if (!tensor) return 0;
    return reinterpret_cast<const cyxwiz::Tensor*>(tensor)->NumBytes();
}

int cyxwiz_tensor_num_dimensions(const CyxWizTensor* tensor) {
    if (!tensor) return 0;
    return reinterpret_cast<const cyxwiz::Tensor*>(tensor)->NumDimensions();
}

void cyxwiz_tensor_get_shape(const CyxWizTensor* tensor, size_t* shape_out) {
    if (!tensor || !shape_out) return;
    const auto& shape = reinterpret_cast<const cyxwiz::Tensor*>(tensor)->Shape();
    std::copy(shape.begin(), shape.end(), shape_out);
}

void* cyxwiz_tensor_data(CyxWizTensor* tensor) {
    if (!tensor) return nullptr;
    return reinterpret_cast<cyxwiz::Tensor*>(tensor)->Data();
}

const void* cyxwiz_tensor_data_const(const CyxWizTensor* tensor) {
    if (!tensor) return nullptr;
    return reinterpret_cast<const cyxwiz::Tensor*>(tensor)->Data();
}

// Tensor Math Operations
CyxWizTensor* cyxwiz_tensor_add(const CyxWizTensor* a, const CyxWizTensor* b) {
    try {
        if (!a || !b) {
            SetLastError("Null tensor pointer");
            return nullptr;
        }
        auto result = *reinterpret_cast<const cyxwiz::Tensor*>(a) +
                      *reinterpret_cast<const cyxwiz::Tensor*>(b);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(result)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_sub(const CyxWizTensor* a, const CyxWizTensor* b) {
    try {
        if (!a || !b) {
            SetLastError("Null tensor pointer");
            return nullptr;
        }
        auto result = *reinterpret_cast<const cyxwiz::Tensor*>(a) -
                      *reinterpret_cast<const cyxwiz::Tensor*>(b);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(result)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_mul(const CyxWizTensor* a, const CyxWizTensor* b) {
    try {
        if (!a || !b) {
            SetLastError("Null tensor pointer");
            return nullptr;
        }
        auto result = *reinterpret_cast<const cyxwiz::Tensor*>(a) *
                      *reinterpret_cast<const cyxwiz::Tensor*>(b);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(result)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_div(const CyxWizTensor* a, const CyxWizTensor* b) {
    try {
        if (!a || !b) {
            SetLastError("Null tensor pointer");
            return nullptr;
        }
        auto result = *reinterpret_cast<const cyxwiz::Tensor*>(a) /
                      *reinterpret_cast<const cyxwiz::Tensor*>(b);
        return reinterpret_cast<CyxWizTensor*>(new cyxwiz::Tensor(std::move(result)));
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

CyxWizTensor* cyxwiz_tensor_matmul(const CyxWizTensor* a, const CyxWizTensor* b) {
    try {
        if (!a || !b) {
            SetLastError("Null tensor pointer");
            return nullptr;
        }

        auto result = cyxwiz::LinearAlgebra::Multiply(
            *reinterpret_cast<const cyxwiz::Tensor*>(a),
            *reinterpret_cast<const cyxwiz::Tensor*>(b)
        );
        if (!result.success) {
            SetLastError(result.error_message.empty()
                ? "Matrix multiplication failed"
                : result.error_message);
            return nullptr;
        }

        return reinterpret_cast<CyxWizTensor*>(
            new cyxwiz::Tensor(std::move(result.tensor))
        );
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

// Optimizer
CyxWizOptimizer* cyxwiz_optimizer_create(CyxWizOptimizerType type, double learning_rate) {
    try {
        auto optimizer = cyxwiz::CreateOptimizer(
            static_cast<cyxwiz::OptimizerType>(type),
            learning_rate
        );
        return reinterpret_cast<CyxWizOptimizer*>(optimizer.release());
    } catch (const std::exception& e) {
        SetLastError(e.what());
        return nullptr;
    }
}

void cyxwiz_optimizer_destroy(CyxWizOptimizer* optimizer) {
    delete reinterpret_cast<cyxwiz::Optimizer*>(optimizer);
}

void cyxwiz_optimizer_set_learning_rate(CyxWizOptimizer* optimizer, double lr) {
    if (optimizer) {
        reinterpret_cast<cyxwiz::Optimizer*>(optimizer)->SetLearningRate(lr);
    }
}

double cyxwiz_optimizer_get_learning_rate(const CyxWizOptimizer* optimizer) {
    if (!optimizer) return 0.0;
    return reinterpret_cast<const cyxwiz::Optimizer*>(optimizer)->GetLearningRate();
}

// Memory Management
size_t cyxwiz_memory_get_allocated_bytes(void) {
    return cyxwiz::MemoryManager::GetAllocatedBytes();
}

size_t cyxwiz_memory_get_peak_bytes(void) {
    return cyxwiz::MemoryManager::GetPeakBytes();
}

void cyxwiz_memory_reset_peak(void) {
    cyxwiz::MemoryManager::ResetPeak();
}

// Error Handling
const char* cyxwiz_get_last_error(void) {
    return g_last_error.c_str();
}

void cyxwiz_clear_last_error(void) {
    ClearLastError();
}
