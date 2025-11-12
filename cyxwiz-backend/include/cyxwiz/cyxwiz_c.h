#pragma once

/**
 * CyxWiz C API
 *
 * This header provides a pure C interface to the CyxWiz backend library.
 * Use this when integrating with C code or languages with C FFI.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Platform-specific exports
#ifdef _WIN32
    #ifdef CYXWIZ_BACKEND_EXPORTS
        #define CYXWIZ_C_API __declspec(dllexport)
    #else
        #define CYXWIZ_C_API __declspec(dllimport)
    #endif
#else
    #define CYXWIZ_C_API __attribute__((visibility("default")))
#endif

// Opaque handles
typedef struct CyxWizTensor CyxWizTensor;
typedef struct CyxWizDevice CyxWizDevice;
typedef struct CyxWizOptimizer CyxWizOptimizer;
typedef struct CyxWizModel CyxWizModel;

// Enums
typedef enum {
    CYXWIZ_DEVICE_CPU = 0,
    CYXWIZ_DEVICE_CUDA = 1,
    CYXWIZ_DEVICE_OPENCL = 2,
    CYXWIZ_DEVICE_METAL = 3,
    CYXWIZ_DEVICE_VULKAN = 4
} CyxWizDeviceType;

typedef enum {
    CYXWIZ_DTYPE_FLOAT32 = 0,
    CYXWIZ_DTYPE_FLOAT64 = 1,
    CYXWIZ_DTYPE_INT32 = 2,
    CYXWIZ_DTYPE_INT64 = 3,
    CYXWIZ_DTYPE_UINT8 = 4
} CyxWizDataType;

typedef enum {
    CYXWIZ_OPTIMIZER_SGD = 0,
    CYXWIZ_OPTIMIZER_ADAM = 1,
    CYXWIZ_OPTIMIZER_ADAMW = 2,
    CYXWIZ_OPTIMIZER_RMSPROP = 3,
    CYXWIZ_OPTIMIZER_ADAGRAD = 4
} CyxWizOptimizerType;

// Initialization
CYXWIZ_C_API bool cyxwiz_initialize(void);
CYXWIZ_C_API void cyxwiz_shutdown(void);
CYXWIZ_C_API const char* cyxwiz_get_version(void);

// Device Management
CYXWIZ_C_API CyxWizDevice* cyxwiz_device_create(CyxWizDeviceType type, int device_id);
CYXWIZ_C_API void cyxwiz_device_destroy(CyxWizDevice* device);
CYXWIZ_C_API void cyxwiz_device_set_active(CyxWizDevice* device);
CYXWIZ_C_API int cyxwiz_device_get_count(CyxWizDeviceType type);

// Tensor Operations
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_create(const size_t* shape, size_t ndim, CyxWizDataType dtype);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_create_with_data(const size_t* shape, size_t ndim,
                                                            const void* data, CyxWizDataType dtype);
CYXWIZ_C_API void cyxwiz_tensor_destroy(CyxWizTensor* tensor);

CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_zeros(const size_t* shape, size_t ndim, CyxWizDataType dtype);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_ones(const size_t* shape, size_t ndim, CyxWizDataType dtype);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_random(const size_t* shape, size_t ndim, CyxWizDataType dtype);

CYXWIZ_C_API size_t cyxwiz_tensor_num_elements(const CyxWizTensor* tensor);
CYXWIZ_C_API size_t cyxwiz_tensor_num_bytes(const CyxWizTensor* tensor);
CYXWIZ_C_API int cyxwiz_tensor_num_dimensions(const CyxWizTensor* tensor);
CYXWIZ_C_API void cyxwiz_tensor_get_shape(const CyxWizTensor* tensor, size_t* shape_out);

CYXWIZ_C_API void* cyxwiz_tensor_data(CyxWizTensor* tensor);
CYXWIZ_C_API const void* cyxwiz_tensor_data_const(const CyxWizTensor* tensor);

// Tensor Math Operations
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_add(const CyxWizTensor* a, const CyxWizTensor* b);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_sub(const CyxWizTensor* a, const CyxWizTensor* b);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_mul(const CyxWizTensor* a, const CyxWizTensor* b);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_div(const CyxWizTensor* a, const CyxWizTensor* b);
CYXWIZ_C_API CyxWizTensor* cyxwiz_tensor_matmul(const CyxWizTensor* a, const CyxWizTensor* b);

// Optimizer
CYXWIZ_C_API CyxWizOptimizer* cyxwiz_optimizer_create(CyxWizOptimizerType type, double learning_rate);
CYXWIZ_C_API void cyxwiz_optimizer_destroy(CyxWizOptimizer* optimizer);
CYXWIZ_C_API void cyxwiz_optimizer_set_learning_rate(CyxWizOptimizer* optimizer, double lr);
CYXWIZ_C_API double cyxwiz_optimizer_get_learning_rate(const CyxWizOptimizer* optimizer);

// Memory Management
CYXWIZ_C_API size_t cyxwiz_memory_get_allocated_bytes(void);
CYXWIZ_C_API size_t cyxwiz_memory_get_peak_bytes(void);
CYXWIZ_C_API void cyxwiz_memory_reset_peak(void);

// Error Handling
CYXWIZ_C_API const char* cyxwiz_get_last_error(void);
CYXWIZ_C_API void cyxwiz_clear_last_error(void);

#ifdef __cplusplus
}
#endif
