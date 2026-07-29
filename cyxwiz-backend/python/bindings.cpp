#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "cyxwiz/cyxwiz.h"
#include <pybind11/functional.h>
#include <atomic>
#include <cstring>
#include <functional>
#include "cyxwiz/sequential.h"
#include "cyxwiz/tokenizer.h"
#include "cyxwiz/audio_processing.h"
#include "cyxwiz/rl_interface.h"
#include "cyxwiz/data_loader.h"
#include "bindings_tensor.h"
// Distributed training
#include "cyxwiz/distributed/process_group.h"
#include "cyxwiz/distributed/cpu_backend.h"
#include "cyxwiz/distributed/ddp.h"
#include "cyxwiz/distributed/distributed_sampler.h"
#include "cyxwiz/distributed/distributed_trainer.h"

using namespace pybind11::literals;  // For _a suffix

namespace py = pybind11;

// Helper function to convert NumPy dtype to CyxWiz DataType
cyxwiz::DataType numpy_dtype_to_cyxwiz(const py::dtype& dt) {
    if (dt.is(py::dtype::of<float>())) {
        return cyxwiz::DataType::Float32;
    } else if (dt.is(py::dtype::of<double>())) {
        return cyxwiz::DataType::Float64;
    } else if (dt.is(py::dtype::of<int32_t>())) {
        return cyxwiz::DataType::Int32;
    } else if (dt.is(py::dtype::of<int64_t>())) {
        return cyxwiz::DataType::Int64;
    } else if (dt.is(py::dtype::of<uint8_t>())) {
        return cyxwiz::DataType::UInt8;
    } else {
        throw std::runtime_error("Unsupported NumPy dtype");
    }
}

// Helper function to get NumPy dtype format string from CyxWiz DataType
std::string cyxwiz_dtype_to_numpy_format(cyxwiz::DataType dt) {
    switch (dt) {
        case cyxwiz::DataType::Float32: return py::format_descriptor<float>::format();
        case cyxwiz::DataType::Float64: return py::format_descriptor<double>::format();
        case cyxwiz::DataType::Int32: return py::format_descriptor<int32_t>::format();
        case cyxwiz::DataType::Int64: return py::format_descriptor<int64_t>::format();
        case cyxwiz::DataType::UInt8: return py::format_descriptor<uint8_t>::format();
        default: throw std::runtime_error("Unsupported CyxWiz DataType");
    }
}

// Helper function to get element size from DataType
size_t get_dtype_size(cyxwiz::DataType dt) {
    switch (dt) {
        case cyxwiz::DataType::Float32: return sizeof(float);
        case cyxwiz::DataType::Float64: return sizeof(double);
        case cyxwiz::DataType::Int32: return sizeof(int32_t);
        case cyxwiz::DataType::Int64: return sizeof(int64_t);
        case cyxwiz::DataType::UInt8: return sizeof(uint8_t);
        default: throw std::runtime_error("Unknown DataType");
    }
}

using NumpyArrayDouble = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Convert a 2D NumPy array to std::vector<std::vector<double>>
std::vector<std::vector<double>> numpy_2d_to_matrix(const NumpyArrayDouble& arr, const char* name) {
    if (arr.ndim() != 2) {
        throw std::runtime_error(std::string(name) + " must be a 2D NumPy array");
    }

    const py::ssize_t rows = arr.shape(0);
    const py::ssize_t cols = arr.shape(1);
    const double* src = arr.data();

    std::vector<std::vector<double>> out(static_cast<size_t>(rows),
                                         std::vector<double>(static_cast<size_t>(cols)));

    for (py::ssize_t r = 0; r < rows; ++r) {
        std::memcpy(out[static_cast<size_t>(r)].data(),
                    src + r * cols,
                    static_cast<size_t>(cols) * sizeof(double));
    }

    return out;
}

// Convert a 1D or 2D NumPy array to matrix form (1D becomes column vector)
std::vector<std::vector<double>> numpy_vector_or_2d_to_matrix(const NumpyArrayDouble& arr, const char* name) {
    if (arr.ndim() == 2) {
        return numpy_2d_to_matrix(arr, name);
    }

    if (arr.ndim() == 1) {
        const py::ssize_t n = arr.shape(0);
        const double* src = arr.data();
        std::vector<std::vector<double>> out(static_cast<size_t>(n), std::vector<double>(1));
        for (py::ssize_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)][0] = src[i];
        }
        return out;
    }

    throw std::runtime_error(std::string(name) + " must be a 1D or 2D NumPy array");
}

std::vector<double> matrix_single_column_to_vector(
    const std::vector<std::vector<double>>& mat,
    const char* name) {
    if (mat.empty()) {
        return {};
    }

    const size_t cols = mat[0].size();
    if (cols != 1) {
        throw std::runtime_error(std::string(name) + " expected single-column matrix result");
    }

    std::vector<double> out(mat.size());
    for (size_t r = 0; r < mat.size(); ++r) {
        if (mat[r].size() != cols) {
            throw std::runtime_error(std::string(name) + " has inconsistent row sizes");
        }
        out[r] = mat[r][0];
    }

    return out;
}

py::array_t<double> matrix_to_numpy(const std::vector<std::vector<double>>& mat) {
    if (mat.empty()) {
        return py::array_t<double>({0, 0});
    }

    const py::ssize_t rows = static_cast<py::ssize_t>(mat.size());
    const py::ssize_t cols = static_cast<py::ssize_t>(mat[0].size());

    py::array_t<double> out({rows, cols});
    double* dst = out.mutable_data();

    for (py::ssize_t r = 0; r < rows; ++r) {
        const auto& row = mat[static_cast<size_t>(r)];
        if (static_cast<py::ssize_t>(row.size()) != cols) {
            throw std::runtime_error("Matrix rows have inconsistent sizes");
        }
        std::memcpy(dst + r * cols, row.data(), static_cast<size_t>(cols) * sizeof(double));
    }

    return out;
}

py::array_t<double> vector_to_numpy(const std::vector<double>& vec) {
    py::array_t<double> out({static_cast<py::ssize_t>(vec.size())});
    if (!vec.empty()) {
        std::memcpy(out.mutable_data(), vec.data(), vec.size() * sizeof(double));
    }
    return out;
}

py::array_t<int> int_vector_to_numpy(const std::vector<int>& vec) {
    py::array_t<int> out({static_cast<py::ssize_t>(vec.size())});
    if (!vec.empty()) {
        std::memcpy(out.mutable_data(), vec.data(), vec.size() * sizeof(int));
    }
    return out;
}

py::array_t<std::complex<double>> complex_vector_to_numpy(const std::vector<std::complex<double>>& vec) {
    py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(vec.size())});
    if (!vec.empty()) {
        std::memcpy(out.mutable_data(), vec.data(), vec.size() * sizeof(std::complex<double>));
    }
    return out;
}

py::array_t<std::complex<double>> complex_matrix_to_numpy(
    const std::vector<std::vector<std::complex<double>>>& mat) {
    if (mat.empty()) {
        return py::array_t<std::complex<double>>({0, 0});
    }

    const py::ssize_t rows = static_cast<py::ssize_t>(mat.size());
    const py::ssize_t cols = static_cast<py::ssize_t>(mat[0].size());

    py::array_t<std::complex<double>> out({rows, cols});
    std::complex<double>* dst = out.mutable_data();

    for (py::ssize_t r = 0; r < rows; ++r) {
        const auto& row = mat[static_cast<size_t>(r)];
        if (static_cast<py::ssize_t>(row.size()) != cols) {
            throw std::runtime_error("Complex matrix rows have inconsistent sizes");
        }
        std::memcpy(dst + r * cols, row.data(),
                    static_cast<size_t>(cols) * sizeof(std::complex<double>));
    }

    return out;
}

PYBIND11_MODULE(pycyxwiz, m) {
    m.doc() = "CyxWiz Python Bindings - High-performance ML compute library";

    // Initialization
    m.def("initialize", &cyxwiz::Initialize, "Initialize the CyxWiz backend");
    m.def("shutdown", &cyxwiz::Shutdown, "Shutdown the CyxWiz backend");
    m.def("get_version", &cyxwiz::GetVersionString, "Get version string");

    // DeviceType enum
    py::enum_<cyxwiz::DeviceType>(m, "DeviceType")
        .value("CPU", cyxwiz::DeviceType::CPU)
        .value("CUDA", cyxwiz::DeviceType::CUDA)
        .value("OPENCL", cyxwiz::DeviceType::OPENCL)
        .value("METAL", cyxwiz::DeviceType::METAL)
        .value("VULKAN", cyxwiz::DeviceType::VULKAN)
        .export_values();

    // DataType enum
    py::enum_<cyxwiz::DataType>(m, "DataType")
        .value("Float32", cyxwiz::DataType::Float32)
        .value("Float64", cyxwiz::DataType::Float64)
        .value("Int32", cyxwiz::DataType::Int32)
        .value("Int64", cyxwiz::DataType::Int64)
        .value("UInt8", cyxwiz::DataType::UInt8)
        .export_values();

    // DeviceInfo
    py::class_<cyxwiz::DeviceInfo>(m, "DeviceInfo")
        .def_readonly("type", &cyxwiz::DeviceInfo::type)
        .def_readonly("device_id", &cyxwiz::DeviceInfo::device_id)
        .def_readonly("name", &cyxwiz::DeviceInfo::name)
        .def_readonly("memory_total", &cyxwiz::DeviceInfo::memory_total)
        .def_readonly("memory_available", &cyxwiz::DeviceInfo::memory_available)
        .def_readonly("compute_units", &cyxwiz::DeviceInfo::compute_units)
        .def_readonly("supports_fp64", &cyxwiz::DeviceInfo::supports_fp64)
        .def_readonly("supports_fp16", &cyxwiz::DeviceInfo::supports_fp16);

    // Device
    py::class_<cyxwiz::Device>(m, "Device")
        .def(py::init<cyxwiz::DeviceType, int>(),
             py::arg("type"), py::arg("device_id") = 0)
        .def("get_type", &cyxwiz::Device::GetType)
        .def("get_device_id", &cyxwiz::Device::GetDeviceId)
        .def("get_info", &cyxwiz::Device::GetInfo)
        .def("set_active", &cyxwiz::Device::SetActive)
        .def("is_active", &cyxwiz::Device::IsActive)
        .def_static("get_available_devices", &cyxwiz::Device::GetAvailableDevices)
        .def_static("get_current_device", &cyxwiz::Device::GetCurrentDevice,
                   py::return_value_policy::reference);

    BindTensor(m);

    // OptimizerType enum
    // Note: Not using .export_values() to avoid conflicts with class names like SGD, Adam
    py::enum_<cyxwiz::OptimizerType>(m, "OptimizerType")
        .value("SGD", cyxwiz::OptimizerType::SGD)
        .value("Adam", cyxwiz::OptimizerType::Adam)
        .value("AdamW", cyxwiz::OptimizerType::AdamW)
        .value("RMSprop", cyxwiz::OptimizerType::RMSprop)
        .value("AdaGrad", cyxwiz::OptimizerType::AdaGrad)
        .value("NAdam", cyxwiz::OptimizerType::NAdam)
        .value("Adadelta", cyxwiz::OptimizerType::Adadelta)
        .value("LAMB", cyxwiz::OptimizerType::LAMB);

    // WarmupType enum for learning rate warmup
    py::enum_<cyxwiz::WarmupType>(m, "WarmupType")
        .value("None_", cyxwiz::WarmupType::None)
        .value("Linear", cyxwiz::WarmupType::Linear)
        .value("Cosine", cyxwiz::WarmupType::Cosine)
        .export_values();

    // Optimizer base class
    py::class_<cyxwiz::Optimizer>(m, "Optimizer")
        .def("step", &cyxwiz::Optimizer::Step,
             py::arg("parameters"),
             py::arg("gradients"),
             "Update parameters using gradients")
        .def("zero_grad", &cyxwiz::Optimizer::ZeroGrad,
             "Clear optimizer state")
        .def("set_learning_rate", &cyxwiz::Optimizer::SetLearningRate,
             py::arg("lr"),
             "Set learning rate")
        .def("get_learning_rate", &cyxwiz::Optimizer::GetLearningRate,
             "Get current learning rate");

    m.def("create_optimizer", &cyxwiz::CreateOptimizer,
          py::arg("type"), py::arg("learning_rate") = 0.001,
          "Create an optimizer instance");

    // SGD Optimizer
    py::class_<cyxwiz::SGDOptimizer, cyxwiz::Optimizer>(m, "SGD")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("momentum") = 0.0,
             "SGD optimizer with optional momentum")
        .def("step", &cyxwiz::SGDOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::SGDOptimizer::ZeroGrad);

    // Adam Optimizer
    py::class_<cyxwiz::AdamOptimizer, cyxwiz::Optimizer>(m, "Adam")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             "Adam optimizer")
        .def("step", &cyxwiz::AdamOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::AdamOptimizer::ZeroGrad);

    // AdamW Optimizer
    py::class_<cyxwiz::AdamWOptimizer, cyxwiz::AdamOptimizer>(m, "AdamW")
        .def(py::init<double, double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("weight_decay") = 0.01,
             "AdamW optimizer (Adam with decoupled weight decay)")
        .def("step", &cyxwiz::AdamWOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"));

    // RMSprop Optimizer
    py::class_<cyxwiz::RMSpropOptimizer, cyxwiz::Optimizer>(m, "RMSprop")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("alpha") = 0.99,
             py::arg("epsilon") = 1e-8,
             py::arg("momentum") = 0.0,
             "RMSprop optimizer")
        .def("step", &cyxwiz::RMSpropOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::RMSpropOptimizer::ZeroGrad);

    // AdaGrad Optimizer
    py::class_<cyxwiz::AdaGradOptimizer, cyxwiz::Optimizer>(m, "AdaGrad")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("epsilon") = 1e-10,
             "AdaGrad optimizer")
        .def("step", &cyxwiz::AdaGradOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::AdaGradOptimizer::ZeroGrad);

    // NAdam Optimizer
    py::class_<cyxwiz::NAdamOptimizer, cyxwiz::Optimizer>(m, "NAdam")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate") = 0.002,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             "NAdam optimizer (Nesterov-accelerated Adam)")
        .def("step", &cyxwiz::NAdamOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::NAdamOptimizer::ZeroGrad);

    // Adadelta Optimizer
    py::class_<cyxwiz::AdadeltaOptimizer, cyxwiz::Optimizer>(m, "Adadelta")
        .def(py::init<double, double>(),
             py::arg("rho") = 0.9,
             py::arg("epsilon") = 1e-6,
             "Adadelta optimizer - no learning rate required")
        .def("step", &cyxwiz::AdadeltaOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::AdadeltaOptimizer::ZeroGrad)
        .def("get_rho", &cyxwiz::AdadeltaOptimizer::GetRho,
             "Get decay rate (rho)");

    // LAMB Optimizer
    py::class_<cyxwiz::LAMBOptimizer, cyxwiz::Optimizer>(m, "LAMB")
        .def(py::init<double, double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-6,
             py::arg("weight_decay") = 0.01,
             "LAMB optimizer for large batch training (e.g., BERT)")
        .def("step", &cyxwiz::LAMBOptimizer::Step,
             py::arg("parameters"), py::arg("gradients"))
        .def("zero_grad", &cyxwiz::LAMBOptimizer::ZeroGrad)
        .def("get_beta1", &cyxwiz::LAMBOptimizer::GetBeta1)
        .def("get_beta2", &cyxwiz::LAMBOptimizer::GetBeta2)
        .def("get_weight_decay", &cyxwiz::LAMBOptimizer::GetWeightDecay);

    // Learning Rate Warmup
    // Note: Use create_lr_warmup() factory function to create instances
    py::class_<cyxwiz::LRWarmup>(m, "LRWarmup",
        "Learning rate warmup wrapper. Use create_lr_warmup() to create instances.")
        .def("step", &cyxwiz::LRWarmup::Step,
             py::arg("parameters"), py::arg("gradients"),
             "Perform optimizer step with warmup-adjusted learning rate")
        .def("zero_grad", &cyxwiz::LRWarmup::ZeroGrad,
             "Clear optimizer state")
        .def("get_current_lr", &cyxwiz::LRWarmup::GetCurrentLR,
             "Get current learning rate (after warmup adjustment)")
        .def("get_warmup_progress", &cyxwiz::LRWarmup::GetWarmupProgress,
             "Get warmup progress (0.0 to 1.0)")
        .def("is_warmup_complete", &cyxwiz::LRWarmup::IsWarmupComplete,
             "Check if warmup phase is complete");

    // Factory function for LRWarmup
    m.def("create_lr_warmup", [](cyxwiz::OptimizerType type, double learning_rate,
                                  int warmup_steps, cyxwiz::WarmupType warmup_type) {
        auto optimizer = cyxwiz::CreateOptimizer(type, learning_rate);
        return std::make_unique<cyxwiz::LRWarmup>(std::move(optimizer), warmup_steps, warmup_type);
    },
    py::arg("optimizer_type"),
    py::arg("learning_rate") = 0.001,
    py::arg("warmup_steps") = 1000,
    py::arg("warmup_type") = cyxwiz::WarmupType::Linear,
    "Create an optimizer with learning rate warmup");

    // Layer base class
    py::class_<cyxwiz::Layer>(m, "Layer")
        .def("forward", &cyxwiz::Layer::Forward,
             py::arg("input"),
             "Forward pass through the layer")
        .def("backward", &cyxwiz::Layer::Backward,
             py::arg("grad_output"),
             "Backward pass (compute gradients)")
        .def("get_parameters", &cyxwiz::Layer::GetParameters,
             "Get layer parameters as dict")
        .def("set_parameters", &cyxwiz::Layer::SetParameters,
             py::arg("params"),
             "Set layer parameters from dict");

    // LinearLayer (fully-connected / dense layer)
    py::class_<cyxwiz::LinearLayer, cyxwiz::Layer>(m, "LinearLayer")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("use_bias") = true,
             "Create a Linear (fully-connected) layer")
        .def("forward", &cyxwiz::LinearLayer::Forward,
             py::arg("input"),
             "Forward pass: output = input @ weight.T + bias")
        .def("backward", &cyxwiz::LinearLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::LinearLayer::GetParameters,
             "Get parameters {'weight': Tensor, 'bias': Tensor}")
        .def("set_parameters", &cyxwiz::LinearLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict")
        .def("get_gradients", &cyxwiz::LinearLayer::GetGradients,
             "Get parameter gradients")
        .def("initialize_weights", &cyxwiz::LinearLayer::InitializeWeights,
             "Re-initialize weights with Xavier initialization")
        .def_property_readonly("in_features", &cyxwiz::LinearLayer::GetInFeatures,
                              "Number of input features")
        .def_property_readonly("out_features", &cyxwiz::LinearLayer::GetOutFeatures,
                              "Number of output features")
        .def_property_readonly("has_bias", &cyxwiz::LinearLayer::HasBias,
                              "Whether layer has bias term");

    // Dense alias for LinearLayer (code generator uses cx.Dense)
    m.attr("Dense") = m.attr("LinearLayer");

    // Conv2D Layer
    py::class_<cyxwiz::Conv2DLayer, cyxwiz::Layer>(m, "Conv2D")
        .def(py::init<int, int, int, int, int, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("use_bias") = true,
             "Create a 2D Convolutional layer")
        .def("forward", &cyxwiz::Conv2DLayer::Forward,
             py::arg("input"),
             "Forward pass: apply 2D convolution")
        .def("backward", &cyxwiz::Conv2DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::Conv2DLayer::GetParameters,
             "Get parameters {'weight': Tensor, 'bias': Tensor}")
        .def("set_parameters", &cyxwiz::Conv2DLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict")
        .def_property_readonly("in_channels", &cyxwiz::Conv2DLayer::GetInChannels)
        .def_property_readonly("out_channels", &cyxwiz::Conv2DLayer::GetOutChannels)
        .def_property_readonly("kernel_size", &cyxwiz::Conv2DLayer::GetKernelSize)
        .def_property_readonly("stride", &cyxwiz::Conv2DLayer::GetStride)
        .def_property_readonly("padding", &cyxwiz::Conv2DLayer::GetPadding);

    // MaxPool2D Layer
    py::class_<cyxwiz::MaxPool2DLayer, cyxwiz::Layer>(m, "MaxPool2D")
        .def(py::init<int, int, int>(),
             py::arg("pool_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0,
             "Create a 2D Max Pooling layer (stride defaults to pool_size)")
        .def("forward", &cyxwiz::MaxPool2DLayer::Forward,
             py::arg("input"),
             "Forward pass: apply max pooling")
        .def("backward", &cyxwiz::MaxPool2DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::MaxPool2DLayer::GetParameters)
        .def("set_parameters", &cyxwiz::MaxPool2DLayer::SetParameters,
             py::arg("params"));

    // AvgPool2D Layer
    py::class_<cyxwiz::AvgPool2DLayer, cyxwiz::Layer>(m, "AvgPool2D")
        .def(py::init<int, int, int>(),
             py::arg("pool_size"),
             py::arg("stride") = -1,
             py::arg("padding") = 0,
             "Create a 2D Average Pooling layer")
        .def("forward", &cyxwiz::AvgPool2DLayer::Forward,
             py::arg("input"),
             "Forward pass: apply average pooling")
        .def("backward", &cyxwiz::AvgPool2DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::AvgPool2DLayer::GetParameters)
        .def("set_parameters", &cyxwiz::AvgPool2DLayer::SetParameters,
             py::arg("params"));

    // GlobalAvgPool2D Layer
    py::class_<cyxwiz::GlobalAvgPool2DLayer, cyxwiz::Layer>(m, "GlobalAvgPool2D")
        .def(py::init<>(),
             "Create a Global Average Pooling layer")
        .def("forward", &cyxwiz::GlobalAvgPool2DLayer::Forward,
             py::arg("input"),
             "Forward pass: reduce spatial dims to single value per channel")
        .def("backward", &cyxwiz::GlobalAvgPool2DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::GlobalAvgPool2DLayer::GetParameters)
        .def("set_parameters", &cyxwiz::GlobalAvgPool2DLayer::SetParameters,
             py::arg("params"));

    // BatchNorm2D Layer
    py::class_<cyxwiz::BatchNorm2DLayer, cyxwiz::Layer>(m, "BatchNorm2D")
        .def(py::init<int, float, float>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5f,
             py::arg("momentum") = 0.1f,
             "Create a 2D Batch Normalization layer")
        .def("forward", &cyxwiz::BatchNorm2DLayer::Forward,
             py::arg("input"),
             "Forward pass: normalize batch")
        .def("backward", &cyxwiz::BatchNorm2DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::BatchNorm2DLayer::GetParameters,
             "Get parameters {'gamma': Tensor, 'beta': Tensor}")
        .def("set_parameters", &cyxwiz::BatchNorm2DLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict");

    // BatchNorm alias (code generator uses cx.BatchNorm)
    m.attr("BatchNorm") = m.attr("BatchNorm2D");
    // LayerNorm Layer
    py::class_<cyxwiz::LayerNormLayer, cyxwiz::Layer>(m, "LayerNorm")
        .def(py::init<const std::vector<int>&, float, bool>(),
             py::arg("normalized_shape"),
             py::arg("eps") = 1e-5f,
             py::arg("elementwise_affine") = true,
             "Create a Layer Normalization layer")
        .def("forward", &cyxwiz::LayerNormLayer::Forward,
             py::arg("input"),
             "Forward pass: normalize across normalized dimensions")
        .def("backward", &cyxwiz::LayerNormLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::LayerNormLayer::GetParameters,
             "Get parameters {'gamma': Tensor, 'beta': Tensor}")
        .def("set_parameters", &cyxwiz::LayerNormLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict");

    // InstanceNorm2D Layer
    py::class_<cyxwiz::InstanceNorm2DLayer, cyxwiz::Layer>(m, "InstanceNorm2D")
        .def(py::init<int, float, bool>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5f,
             py::arg("affine") = false,
             "Create a 2D Instance Normalization layer")
        .def("forward", &cyxwiz::InstanceNorm2DLayer::Forward,
             py::arg("input"),
             "Forward pass: normalize per instance")
        .def("backward", &cyxwiz::InstanceNorm2DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::InstanceNorm2DLayer::GetParameters,
             "Get parameters {'gamma': Tensor, 'beta': Tensor} if affine=True")
        .def("set_parameters", &cyxwiz::InstanceNorm2DLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict");

    // GroupNorm Layer
    py::class_<cyxwiz::GroupNormLayer, cyxwiz::Layer>(m, "GroupNorm")
        .def(py::init<int, int, float, bool>(),
             py::arg("num_groups"),
             py::arg("num_channels"),
             py::arg("eps") = 1e-5f,
             py::arg("affine") = true,
             "Create a Group Normalization layer")
        .def("forward", &cyxwiz::GroupNormLayer::Forward,
             py::arg("input"),
             "Forward pass: normalize per group")
        .def("backward", &cyxwiz::GroupNormLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::GroupNormLayer::GetParameters,
             "Get parameters {'gamma': Tensor, 'beta': Tensor}")
        .def("set_parameters", &cyxwiz::GroupNormLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict");

    // Conv1D Layer
    py::class_<cyxwiz::Conv1DLayer, cyxwiz::Layer>(m, "Conv1D")
        .def(py::init<int, int, int, int, int, int, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("dilation") = 1,
             py::arg("use_bias") = true,
             "Create a 1D Convolutional layer")
        .def("forward", &cyxwiz::Conv1DLayer::Forward,
             py::arg("input"),
             "Forward pass: apply 1D convolution")
        .def("backward", &cyxwiz::Conv1DLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::Conv1DLayer::GetParameters,
             "Get parameters {'weights': Tensor, 'bias': Tensor}")
        .def("set_parameters", &cyxwiz::Conv1DLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict")
        .def_property_readonly("in_channels", &cyxwiz::Conv1DLayer::GetInChannels)
        .def_property_readonly("out_channels", &cyxwiz::Conv1DLayer::GetOutChannels)
        .def_property_readonly("kernel_size", &cyxwiz::Conv1DLayer::GetKernelSize)
        .def_property_readonly("stride", &cyxwiz::Conv1DLayer::GetStride)
        .def_property_readonly("padding", &cyxwiz::Conv1DLayer::GetPadding)
        .def_property_readonly("dilation", &cyxwiz::Conv1DLayer::GetDilation);

    // Embedding Layer
    py::class_<cyxwiz::EmbeddingLayer, cyxwiz::Layer>(m, "Embedding")
        .def(py::init<int, int, int, float>(),
             py::arg("num_embeddings"),
             py::arg("embedding_dim"),
             py::arg("padding_idx") = -1,
             py::arg("max_norm") = 0.0f,
             "Create an Embedding layer (lookup table for token embeddings)")
        .def("forward", &cyxwiz::EmbeddingLayer::Forward,
             py::arg("input"),
             "Forward pass: lookup embeddings for input indices")
        .def("backward", &cyxwiz::EmbeddingLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: accumulate gradients for used embeddings")
        .def("get_parameters", &cyxwiz::EmbeddingLayer::GetParameters,
             "Get parameters {'weight': Tensor}")
        .def("set_parameters", &cyxwiz::EmbeddingLayer::SetParameters,
             py::arg("params"),
             "Set parameters from dict")
        .def("get_embedding", &cyxwiz::EmbeddingLayer::GetEmbedding,
             py::arg("index"),
             "Get embedding vector for a specific index")
        .def("set_embedding", &cyxwiz::EmbeddingLayer::SetEmbedding,
             py::arg("index"), py::arg("embedding"),
             "Set embedding vector for a specific index")
        .def("load_pretrained_weights", &cyxwiz::EmbeddingLayer::LoadPretrainedWeights,
             py::arg("weights"), py::arg("freeze") = false,
             "Load pretrained embedding weights")
        .def_property_readonly("num_embeddings", &cyxwiz::EmbeddingLayer::GetNumEmbeddings)
        .def_property_readonly("embedding_dim", &cyxwiz::EmbeddingLayer::GetEmbeddingDim)
        .def_property_readonly("padding_idx", &cyxwiz::EmbeddingLayer::GetPaddingIdx)
        .def_property("frozen", &cyxwiz::EmbeddingLayer::IsFrozen, &cyxwiz::EmbeddingLayer::SetFrozen,
                     "Whether embeddings are frozen (not updated during training)");

    // LSTM Layer
    py::class_<cyxwiz::LSTMLayer, cyxwiz::Layer>(m, "LSTM")
        .def(py::init<int, int, int, bool, bool, float>(),
             py::arg("input_size"),
             py::arg("hidden_size"),
             py::arg("num_layers") = 1,
             py::arg("batch_first") = true,
             py::arg("bidirectional") = false,
             py::arg("dropout") = 0.0f,
             "Create an LSTM layer")
        .def("forward", &cyxwiz::LSTMLayer::Forward,
             py::arg("input"),
             "Forward pass: process sequence through LSTM")
        .def("backward", &cyxwiz::LSTMLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: backpropagation through time")
        .def("get_parameters", &cyxwiz::LSTMLayer::GetParameters,
             "Get LSTM parameters")
        .def("set_parameters", &cyxwiz::LSTMLayer::SetParameters,
             py::arg("params"),
             "Set LSTM parameters")
        .def("reset_state", &cyxwiz::LSTMLayer::ResetState,
             "Reset hidden and cell states to zeros")
        .def("set_hidden_state", &cyxwiz::LSTMLayer::SetHiddenState,
             py::arg("h0"),
             "Set initial hidden state")
        .def("set_cell_state", &cyxwiz::LSTMLayer::SetCellState,
             py::arg("c0"),
             "Set initial cell state")
        .def("get_hidden_state", &cyxwiz::LSTMLayer::GetHiddenState,
             "Get current hidden state")
        .def("get_cell_state", &cyxwiz::LSTMLayer::GetCellState,
             "Get current cell state")
        .def_property_readonly("input_size", &cyxwiz::LSTMLayer::GetInputSize)
        .def_property_readonly("hidden_size", &cyxwiz::LSTMLayer::GetHiddenSize)
        .def_property_readonly("num_layers", &cyxwiz::LSTMLayer::GetNumLayers)
        .def_property_readonly("batch_first", &cyxwiz::LSTMLayer::IsBatchFirst)
        .def_property_readonly("bidirectional", &cyxwiz::LSTMLayer::IsBidirectional)
        .def_property_readonly("num_directions", &cyxwiz::LSTMLayer::GetNumDirections);

    // GRU Layer
    py::class_<cyxwiz::GRULayer, cyxwiz::Layer>(m, "GRU")
        .def(py::init<int, int, int, bool, bool, float>(),
             py::arg("input_size"),
             py::arg("hidden_size"),
             py::arg("num_layers") = 1,
             py::arg("batch_first") = true,
             py::arg("bidirectional") = false,
             py::arg("dropout") = 0.0f,
             "Create a GRU layer")
        .def("forward", &cyxwiz::GRULayer::Forward,
             py::arg("input"),
             "Forward pass: process sequence through GRU")
        .def("backward", &cyxwiz::GRULayer::Backward,
             py::arg("grad_output"),
             "Backward pass: backpropagation through time")
        .def("get_parameters", &cyxwiz::GRULayer::GetParameters,
             "Get GRU parameters")
        .def("set_parameters", &cyxwiz::GRULayer::SetParameters,
             py::arg("params"),
             "Set GRU parameters")
        .def("reset_state", &cyxwiz::GRULayer::ResetState,
             "Reset hidden state to zeros")
        .def("set_hidden_state", &cyxwiz::GRULayer::SetHiddenState,
             py::arg("h0"),
             "Set initial hidden state")
        .def("get_hidden_state", &cyxwiz::GRULayer::GetHiddenState,
             "Get current hidden state")
        .def_property_readonly("input_size", &cyxwiz::GRULayer::GetInputSize)
        .def_property_readonly("hidden_size", &cyxwiz::GRULayer::GetHiddenSize)
        .def_property_readonly("num_layers", &cyxwiz::GRULayer::GetNumLayers)
        .def_property_readonly("batch_first", &cyxwiz::GRULayer::IsBatchFirst)
        .def_property_readonly("bidirectional", &cyxwiz::GRULayer::IsBidirectional);

    // MultiHeadAttention Layer
    py::class_<cyxwiz::MultiHeadAttentionLayer, cyxwiz::Layer>(m, "MultiHeadAttention")
        .def(py::init<int, int, float, bool>(),
             py::arg("embed_dim"),
             py::arg("num_heads"),
             py::arg("dropout") = 0.0f,
             py::arg("use_bias") = true,
             "Create a Multi-Head Attention layer")
        .def("forward", static_cast<cyxwiz::Tensor (cyxwiz::MultiHeadAttentionLayer::*)(const cyxwiz::Tensor&)>(&cyxwiz::MultiHeadAttentionLayer::Forward),
             py::arg("input"),
             "Self-attention forward pass")
        .def("forward_qkv", [](cyxwiz::MultiHeadAttentionLayer& self,
                               const cyxwiz::Tensor& query,
                               const cyxwiz::Tensor& key,
                               const cyxwiz::Tensor& value,
                               const cyxwiz::Tensor* attn_mask) {
            return self.Forward(query, key, value, attn_mask);
        },
             py::arg("query"), py::arg("key"), py::arg("value"),
             py::arg("attn_mask") = nullptr,
             "Full attention forward with separate Q, K, V")
        .def("backward", &cyxwiz::MultiHeadAttentionLayer::Backward,
             py::arg("grad_output"),
             "Backward pass")
        .def("get_parameters", &cyxwiz::MultiHeadAttentionLayer::GetParameters,
             "Get attention parameters (W_q, W_k, W_v, W_o, biases)")
        .def("set_parameters", &cyxwiz::MultiHeadAttentionLayer::SetParameters,
             py::arg("params"),
             "Set attention parameters")
        .def("get_attention_weights", &cyxwiz::MultiHeadAttentionLayer::GetAttentionWeights,
             "Get last computed attention weights (for visualization)")
        .def_property_readonly("embed_dim", &cyxwiz::MultiHeadAttentionLayer::GetEmbedDim)
        .def_property_readonly("num_heads", &cyxwiz::MultiHeadAttentionLayer::GetNumHeads)
        .def_property_readonly("head_dim", &cyxwiz::MultiHeadAttentionLayer::GetHeadDim);

    // TransformerEncoderLayer
    py::class_<cyxwiz::TransformerEncoderLayer, cyxwiz::Layer>(m, "TransformerEncoderLayer")
        .def(py::init<int, int, int, float, bool>(),
             py::arg("d_model"),
             py::arg("nhead"),
             py::arg("dim_feedforward") = 2048,
             py::arg("dropout") = 0.1f,
             py::arg("norm_first") = false,
             "Create a Transformer Encoder layer")
        .def("forward", static_cast<cyxwiz::Tensor (cyxwiz::TransformerEncoderLayer::*)(const cyxwiz::Tensor&)>(&cyxwiz::TransformerEncoderLayer::Forward),
             py::arg("input"),
             "Forward pass without mask")
        .def("forward_with_mask", [](cyxwiz::TransformerEncoderLayer& self,
                                     const cyxwiz::Tensor& input,
                                     const cyxwiz::Tensor* src_mask) {
            return self.Forward(input, src_mask);
        },
             py::arg("input"), py::arg("src_mask") = nullptr,
             "Forward pass with optional attention mask")
        .def("backward", &cyxwiz::TransformerEncoderLayer::Backward,
             py::arg("grad_output"),
             "Backward pass")
        .def("get_parameters", &cyxwiz::TransformerEncoderLayer::GetParameters,
             "Get all layer parameters")
        .def("set_parameters", &cyxwiz::TransformerEncoderLayer::SetParameters,
             py::arg("params"),
             "Set layer parameters")
        .def("set_training", &cyxwiz::TransformerEncoderLayer::SetTraining,
             py::arg("training"),
             "Set training mode (affects dropout)");

    // TransformerDecoderLayer
    py::class_<cyxwiz::TransformerDecoderLayer, cyxwiz::Layer>(m, "TransformerDecoderLayer")
        .def(py::init<int, int, int, float, bool>(),
             py::arg("d_model"),
             py::arg("nhead"),
             py::arg("dim_feedforward") = 2048,
             py::arg("dropout") = 0.1f,
             py::arg("norm_first") = false,
             "Create a Transformer Decoder layer")
        .def("forward", static_cast<cyxwiz::Tensor (cyxwiz::TransformerDecoderLayer::*)(const cyxwiz::Tensor&)>(&cyxwiz::TransformerDecoderLayer::Forward),
             py::arg("input"),
             "Self-attention only forward pass")
        .def("forward_with_memory", [](cyxwiz::TransformerDecoderLayer& self,
                                       const cyxwiz::Tensor& tgt,
                                       const cyxwiz::Tensor& memory,
                                       const cyxwiz::Tensor* tgt_mask,
                                       const cyxwiz::Tensor* memory_mask) {
            return self.Forward(tgt, memory, tgt_mask, memory_mask);
        },
             py::arg("tgt"), py::arg("memory"),
             py::arg("tgt_mask") = nullptr, py::arg("memory_mask") = nullptr,
             "Full decoder forward with encoder memory and masks")
        .def("backward", &cyxwiz::TransformerDecoderLayer::Backward,
             py::arg("grad_output"),
             "Backward pass")
        .def("get_last_memory_gradient", &cyxwiz::TransformerDecoderLayer::GetLastMemoryGradient,
             "Get gradient with respect to encoder memory after backward")
        .def("get_parameters", &cyxwiz::TransformerDecoderLayer::GetParameters,
             "Get all layer parameters")
        .def("set_parameters", &cyxwiz::TransformerDecoderLayer::SetParameters,
             py::arg("params"),
             "Set layer parameters")
        .def("set_training", &cyxwiz::TransformerDecoderLayer::SetTraining,
             py::arg("training"),
             "Set training mode (affects dropout)")
        .def_static("generate_causal_mask", &cyxwiz::TransformerDecoderLayer::GenerateCausalMask,
             py::arg("size"),
             "Generate causal mask for autoregressive decoding");

    // Flatten Layer
    py::class_<cyxwiz::FlattenLayer, cyxwiz::Layer>(m, "Flatten")
        .def(py::init<>(),
             "Create a Flatten layer")
        .def("forward", &cyxwiz::FlattenLayer::Forward,
             py::arg("input"),
             "Forward pass: flatten spatial dimensions")
        .def("backward", &cyxwiz::FlattenLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: restore original shape")
        .def("get_parameters", &cyxwiz::FlattenLayer::GetParameters)
        .def("set_parameters", &cyxwiz::FlattenLayer::SetParameters,
             py::arg("params"));

    // Dropout Layer
    py::class_<cyxwiz::DropoutLayer, cyxwiz::Layer>(m, "Dropout")
        .def(py::init<float>(),
             py::arg("p") = 0.5f,
             "Create a Dropout layer (p = probability of dropping)")
        .def("forward", &cyxwiz::DropoutLayer::Forward,
             py::arg("input"),
             "Forward pass: randomly drop units during training")
        .def("backward", &cyxwiz::DropoutLayer::Backward,
             py::arg("grad_output"),
             "Backward pass: compute gradients")
        .def("get_parameters", &cyxwiz::DropoutLayer::GetParameters)
        .def("set_parameters", &cyxwiz::DropoutLayer::SetParameters,
             py::arg("params"));

    // UpsampleMode enum
    py::enum_<cyxwiz::UpsampleMode>(m, "UpsampleMode")
        .value("Nearest", cyxwiz::UpsampleMode::Nearest)
        .value("Bilinear", cyxwiz::UpsampleMode::Bilinear);

    // ConvTranspose2D Layer
    py::class_<cyxwiz::ConvTranspose2DLayer, cyxwiz::Layer>(m, "ConvTranspose2D")
        .def(py::init<int, int, int, int, int, int, bool>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0,
             py::arg("output_padding") = 0, py::arg("use_bias") = true,
             "Create a 2D transposed convolution layer")
        .def("forward", &cyxwiz::ConvTranspose2DLayer::Forward, py::arg("input"))
        .def("backward", &cyxwiz::ConvTranspose2DLayer::Backward, py::arg("grad_output"))
        .def("get_parameters", &cyxwiz::ConvTranspose2DLayer::GetParameters)
        .def("set_parameters", &cyxwiz::ConvTranspose2DLayer::SetParameters, py::arg("params"))
        .def_property_readonly("in_channels", &cyxwiz::ConvTranspose2DLayer::GetInChannels)
        .def_property_readonly("out_channels", &cyxwiz::ConvTranspose2DLayer::GetOutChannels)
        .def_property_readonly("kernel_size", &cyxwiz::ConvTranspose2DLayer::GetKernelSize)
        .def_property_readonly("stride", &cyxwiz::ConvTranspose2DLayer::GetStride)
        .def_property_readonly("padding", &cyxwiz::ConvTranspose2DLayer::GetPadding)
        .def_property_readonly("output_padding", &cyxwiz::ConvTranspose2DLayer::GetOutputPadding);

    // Upsample2D Layer
    py::class_<cyxwiz::Upsample2DLayer, cyxwiz::Layer>(m, "Upsample2D")
        .def(py::init<int, cyxwiz::UpsampleMode>(),
             py::arg("scale_factor") = 2,
             py::arg("mode") = cyxwiz::UpsampleMode::Nearest,
             "Create a 2D upsampling layer (no learnable parameters)")
        .def("forward", &cyxwiz::Upsample2DLayer::Forward, py::arg("input"))
        .def("backward", &cyxwiz::Upsample2DLayer::Backward, py::arg("grad_output"))
        .def("get_parameters", &cyxwiz::Upsample2DLayer::GetParameters)
        .def("set_parameters", &cyxwiz::Upsample2DLayer::SetParameters, py::arg("params"))
        .def_property_readonly("scale_factor", &cyxwiz::Upsample2DLayer::GetScaleFactor)
        .def_property_readonly("mode", &cyxwiz::Upsample2DLayer::GetMode);

    // PixelShuffle Layer
    py::class_<cyxwiz::PixelShuffleLayer, cyxwiz::Layer>(m, "PixelShuffle")
        .def(py::init<int>(),
             py::arg("upscale_factor"),
             "Create a PixelShuffle layer (depth to space rearrangement)")
        .def("forward", &cyxwiz::PixelShuffleLayer::Forward, py::arg("input"))
        .def("backward", &cyxwiz::PixelShuffleLayer::Backward, py::arg("grad_output"))
        .def("get_parameters", &cyxwiz::PixelShuffleLayer::GetParameters)
        .def("set_parameters", &cyxwiz::PixelShuffleLayer::SetParameters, py::arg("params"))
        .def_property_readonly("upscale_factor", &cyxwiz::PixelShuffleLayer::GetUpscaleFactor);

    // Base Loss class (abstract - for type hierarchy)
    py::class_<cyxwiz::Loss>(m, "Loss")
        .def("forward", &cyxwiz::Loss::Forward,
             py::arg("predictions"), py::arg("targets"),
             "Compute loss value")
        .def("backward", &cyxwiz::Loss::Backward,
             py::arg("predictions"), py::arg("targets"),
             "Compute loss gradients");

    // MSE Loss (concrete implementation)
    py::class_<cyxwiz::MSELoss, cyxwiz::Loss>(m, "MSELoss")
        .def(py::init<>(),
             "Create MSE Loss: mean((predictions - targets)^2)")
        .def("forward", &cyxwiz::MSELoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute MSE loss")
        .def("backward", &cyxwiz::MSELoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: dL/dy = 2*(predictions - targets)/N");

    // CrossEntropy Loss (concrete implementation)
    py::class_<cyxwiz::CrossEntropyLoss, cyxwiz::Loss>(m, "CrossEntropyLoss")
        .def(py::init<cyxwiz::Reduction, int, std::vector<float>, float>(),
             py::arg("reduction") = cyxwiz::Reduction::Mean,
             py::arg("ignore_index") = -100,
             py::arg("class_weights") = std::vector<float>{},
             py::arg("label_smoothing") = 0.0f,
             "Create CrossEntropy Loss with optional per-class weights and label smoothing")
        .def("forward", &cyxwiz::CrossEntropyLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute cross entropy loss (predictions are logits)")
        .def("backward", &cyxwiz::CrossEntropyLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t logits")
        .def_property_readonly("class_weights",
             &cyxwiz::CrossEntropyLoss::GetClassWeights,
             "Per-class loss weights")
        .def_property_readonly("label_smoothing",
             &cyxwiz::CrossEntropyLoss::GetLabelSmoothing,
             "Label smoothing factor");

    // BCEWithLogits Loss (for binary / multi-label classification)
    py::class_<cyxwiz::BCEWithLogitsLoss, cyxwiz::Loss>(m, "BCEWithLogitsLoss")
        .def(py::init<cyxwiz::Reduction, float>(),
             py::arg("reduction") = cyxwiz::Reduction::Mean,
             py::arg("pos_weight") = 1.0f,
             "Create BCEWithLogits Loss with optional positive-class weight")
        .def("forward", &cyxwiz::BCEWithLogitsLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute binary cross entropy from logits")
        .def("backward", &cyxwiz::BCEWithLogitsLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t logits")
        .def_property("pos_weight",
             &cyxwiz::BCEWithLogitsLoss::GetPosWeight,
             &cyxwiz::BCEWithLogitsLoss::SetPosWeight,
             "Positive-class weighting factor");

    py::class_<cyxwiz::SoftDiceLoss, cyxwiz::Loss>(m, "SoftDiceLoss")
        .def(py::init<cyxwiz::Reduction, float>(),
             py::arg("reduction") = cyxwiz::Reduction::Mean,
             py::arg("smooth") = 1.0f,
             "Create Soft Dice loss for probability masks")
        .def("forward", &cyxwiz::SoftDiceLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute Soft Dice loss")
        .def("backward", &cyxwiz::SoftDiceLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t predictions")
        .def_property_readonly("smooth",
             &cyxwiz::SoftDiceLoss::GetSmooth,
             "Smoothing constant");

    py::class_<cyxwiz::TverskyLoss, cyxwiz::Loss>(m, "TverskyLoss")
        .def(py::init<cyxwiz::Reduction, float, float, float>(),
             py::arg("reduction") = cyxwiz::Reduction::Mean,
             py::arg("alpha") = 0.5f,
             py::arg("beta") = 0.5f,
             py::arg("smooth") = 1.0f,
             "Create Tversky loss for probability masks")
        .def("forward", &cyxwiz::TverskyLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute Tversky loss")
        .def("backward", &cyxwiz::TverskyLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t predictions")
        .def_property_readonly("alpha",
             &cyxwiz::TverskyLoss::GetAlpha,
             "False-positive penalty")
        .def_property_readonly("beta",
             &cyxwiz::TverskyLoss::GetBeta,
             "False-negative penalty")
        .def_property_readonly("smooth",
             &cyxwiz::TverskyLoss::GetSmooth,
             "Smoothing constant");

    py::class_<cyxwiz::JaccardLoss, cyxwiz::Loss>(m, "JaccardLoss")
        .def(py::init<cyxwiz::Reduction, float>(),
             py::arg("reduction") = cyxwiz::Reduction::Mean,
             py::arg("smooth") = 1.0f,
             "Create Jaccard/IoU loss for probability masks")
        .def("forward", &cyxwiz::JaccardLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute Jaccard loss")
        .def("backward", &cyxwiz::JaccardLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t predictions")
        .def_property_readonly("smooth",
             &cyxwiz::JaccardLoss::GetSmooth,
             "Smoothing constant");

    // Focal Loss (for class imbalance)
    py::class_<cyxwiz::FocalLoss, cyxwiz::Loss>(m, "FocalLoss")
        .def(py::init<float, float>(),
             py::arg("alpha") = 0.25f,
             py::arg("gamma") = 2.0f,
             "Create Focal Loss: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)")
        .def("forward", &cyxwiz::FocalLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute focal loss (predictions are logits)")
        .def("backward", &cyxwiz::FocalLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t logits")
        .def_property("alpha", &cyxwiz::FocalLoss::GetAlpha, &cyxwiz::FocalLoss::SetAlpha,
                     "Class balance weight (default: 0.25)")
        .def_property("gamma", &cyxwiz::FocalLoss::GetGamma, &cyxwiz::FocalLoss::SetGamma,
                     "Focusing parameter - higher = more focus on hard examples (default: 2.0)");

    // Triplet Loss (for metric learning)
    py::enum_<cyxwiz::TripletLoss::DistanceType>(m, "TripletDistanceType")
        .value("Euclidean", cyxwiz::TripletLoss::DistanceType::Euclidean)
        .value("Cosine", cyxwiz::TripletLoss::DistanceType::Cosine);

    py::class_<cyxwiz::TripletLoss, cyxwiz::Loss>(m, "TripletLoss")
        .def(py::init<float, cyxwiz::TripletLoss::DistanceType>(),
             py::arg("margin") = 1.0f,
             py::arg("distance_type") = cyxwiz::TripletLoss::DistanceType::Euclidean,
             "Create Triplet Loss: L = max(d(a,p) - d(a,n) + margin, 0)")
        .def("forward", &cyxwiz::TripletLoss::Forward,
             py::arg("anchor"),
             py::arg("positive"),
             "Forward: compute triplet loss (set negative via set_negative first)")
        .def("backward", &cyxwiz::TripletLoss::Backward,
             py::arg("anchor"),
             py::arg("positive"),
             "Backward: gradient w.r.t anchor")
        .def("set_negative", &cyxwiz::TripletLoss::SetNegative,
             py::arg("negative"),
             "Set negative samples for triplet computation")
        .def_property("margin", &cyxwiz::TripletLoss::GetMargin, &cyxwiz::TripletLoss::SetMargin,
                     "Margin for triplet loss (default: 1.0)");

    // Contrastive Loss (for similarity learning)
    py::class_<cyxwiz::ContrastiveLoss, cyxwiz::Loss>(m, "ContrastiveLoss")
        .def(py::init<float>(),
             py::arg("margin") = 1.0f,
             "Create Contrastive Loss: L = (1-y)*d^2 + y*max(margin-d,0)^2")
        .def("forward", &cyxwiz::ContrastiveLoss::Forward,
             py::arg("x1"),
             py::arg("x2"),
             "Forward: compute contrastive loss (set labels via set_labels first)")
        .def("backward", &cyxwiz::ContrastiveLoss::Backward,
             py::arg("x1"),
             py::arg("x2"),
             "Backward: gradient w.r.t x1")
        .def("set_labels", &cyxwiz::ContrastiveLoss::SetLabels,
             py::arg("labels"),
             "Set labels: 0=similar (minimize distance), 1=dissimilar (push apart)")
        .def_property("margin", &cyxwiz::ContrastiveLoss::GetMargin, &cyxwiz::ContrastiveLoss::SetMargin,
                     "Margin for dissimilar pairs (default: 1.0)");

    // ============================================================================
    // FUNCTIONAL API (lowercase functions)
    // ============================================================================

    // Activation functions
    m.def("relu", [](const cyxwiz::Tensor& x) {
        cyxwiz::ReLU act;
        return act.Forward(x);
    }, py::arg("input"), "Apply ReLU activation: f(x) = max(0, x)");

    m.def("sigmoid", [](const cyxwiz::Tensor& x) {
        cyxwiz::Sigmoid act;
        return act.Forward(x);
    }, py::arg("input"), "Apply Sigmoid activation: f(x) = 1 / (1 + exp(-x))");

    m.def("tanh", [](const cyxwiz::Tensor& x) {
        cyxwiz::Tanh act;
        return act.Forward(x);
    }, py::arg("input"), "Apply Tanh activation: f(x) = tanh(x)");

    m.def("softmax", [](const cyxwiz::Tensor& x, int dim) {
        cyxwiz::SoftmaxActivation act(dim);
        return act.Forward(x);
    }, py::arg("input"), py::arg("dim") = -1, "Apply Softmax activation");

    m.def("gelu", [](const cyxwiz::Tensor& x) {
        cyxwiz::GELUActivation act;
        return act.Forward(x);
    }, py::arg("input"), "Apply GELU activation");

    m.def("leaky_relu", [](const cyxwiz::Tensor& x, float negative_slope) {
        cyxwiz::LeakyReLUActivation act(negative_slope);
        return act.Forward(x);
    }, py::arg("input"), py::arg("negative_slope") = 0.01f, "Apply LeakyReLU activation");

    m.def("elu", [](const cyxwiz::Tensor& x, float alpha) {
        cyxwiz::ELUActivation act(alpha);
        return act.Forward(x);
    }, py::arg("input"), py::arg("alpha") = 1.0f, "Apply ELU activation");

    m.def("swish", [](const cyxwiz::Tensor& x) {
        cyxwiz::SwishActivation act;
        return act.Forward(x);
    }, py::arg("input"), "Apply Swish activation: f(x) = x * sigmoid(x)");

    m.def("silu", [](const cyxwiz::Tensor& x) {
        cyxwiz::SwishActivation act;
        return act.Forward(x);
    }, py::arg("input"), "Apply SiLU activation (alias for Swish)");

    m.def("mish", [](const cyxwiz::Tensor& x) {
        cyxwiz::MishActivation act;
        return act.Forward(x);
    }, py::arg("input"), "Apply Mish activation");

    // Layer-like functional operations
    m.def("flatten", [](const cyxwiz::Tensor& x) {
        cyxwiz::FlattenLayer layer;
        return layer.Forward(x);
    }, py::arg("input"), "Flatten spatial dimensions");

    m.def("dropout", [](const cyxwiz::Tensor& x, float p, bool training) {
        cyxwiz::DropoutLayer layer(p);
        layer.SetTraining(training);
        return layer.Forward(x);
    }, py::arg("input"), py::arg("p") = 0.5f, py::arg("training") = true,
    "Apply dropout during training");

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    m.def("cuda_available", []() {
        auto devices = cyxwiz::Device::GetAvailableDevices();
        for (const auto& d : devices) {
            if (d.type == cyxwiz::DeviceType::CUDA) return true;
        }
        return false;
    }, "Check if CUDA is available");

    m.def("opencl_available", []() {
        auto devices = cyxwiz::Device::GetAvailableDevices();
        for (const auto& d : devices) {
            if (d.type == cyxwiz::DeviceType::OPENCL) return true;
        }
        return false;
    }, "Check if OpenCL is available");

    m.def("metal_available", []() {
        auto devices = cyxwiz::Device::GetAvailableDevices();
        for (const auto& d : devices) {
            if (d.type == cyxwiz::DeviceType::METAL) return true;
        }
        return false;
    }, "Check if Metal is available");

    m.def("get_device", [](cyxwiz::DeviceType type, int device_id) {
        return cyxwiz::Device(type, device_id);
    }, py::arg("type"), py::arg("device_id") = 0, "Get a device by type and ID");

    m.def("set_device", [](cyxwiz::Device& device) {
        device.SetActive();
    }, py::arg("device"), "Set the active device");

    m.def("get_available_devices", &cyxwiz::Device::GetAvailableDevices,
          "Get list of all available devices");

    // ============================================================================
    // LINEAR ALGEBRA SUBMODULE
    // ============================================================================
    auto linalg = m.def_submodule("linalg", "Linear algebra functions (MATLAB-style)");

    // Matrix creation
    linalg.def("eye", [](int n) {
        auto result = cyxwiz::LinearAlgebra::Identity(n);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create square identity matrix", py::arg("n"));

    linalg.def("eye", [](int rows, int cols) {
        auto result = cyxwiz::LinearAlgebra::Identity(rows, cols);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create non-square identity matrix (1s on diagonal)", py::arg("rows"), py::arg("cols"));

    linalg.def("zeros", [](int n) {
        auto result = cyxwiz::LinearAlgebra::Zeros(n);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create square zero matrix", py::arg("n"));

    linalg.def("zeros", [](int rows, int cols) {
        auto result = cyxwiz::LinearAlgebra::Zeros(rows, cols);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create zero matrix", py::arg("rows"), py::arg("cols"));

    linalg.def("ones", [](int n) {
        auto result = cyxwiz::LinearAlgebra::Ones(n);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create square ones matrix", py::arg("n"));

    linalg.def("ones", [](int rows, int cols) {
        auto result = cyxwiz::LinearAlgebra::Ones(rows, cols);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create ones matrix", py::arg("rows"), py::arg("cols"));

    linalg.def("diag", [](const NumpyArrayDouble& d) {
        if (d.ndim() != 1) {
            throw std::runtime_error("d must be a 1D NumPy array");
        }
        std::vector<double> vec(static_cast<size_t>(d.shape(0)));
        if (!vec.empty()) {
            std::memcpy(vec.data(), d.data(), vec.size() * sizeof(double));
        }
        auto result = cyxwiz::LinearAlgebra::Diagonal(vec);
        if (!result.success) throw std::runtime_error(result.error_message);
        return matrix_to_numpy(result.matrix);
    }, "Create diagonal matrix from NumPy vector", py::arg("d"));

    linalg.def("diag", [](const std::vector<double>& d) {
        auto result = cyxwiz::LinearAlgebra::Diagonal(d);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create diagonal matrix from vector", py::arg("d"));

    // Decompositions
    linalg.def("svd", [](const NumpyArrayDouble& A, bool full_matrices) {
        auto result = cyxwiz::LinearAlgebra::SVD(numpy_2d_to_matrix(A, "A"), full_matrices);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(
            matrix_to_numpy(result.U),
            vector_to_numpy(result.S),
            matrix_to_numpy(result.Vt));
    }, "Singular Value Decomposition: U, S, Vt = svd(A)",
       py::arg("A"), py::arg("full_matrices") = false);

    linalg.def("svd", [](const std::vector<std::vector<double>>& A, bool full_matrices) {
        auto result = cyxwiz::LinearAlgebra::SVD(A, full_matrices);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.U, result.S, result.Vt);
    }, "Singular Value Decomposition: U, S, Vt = svd(A)", py::arg("A"), py::arg("full_matrices") = false);

    linalg.def("eig", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::Eigen(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(
            complex_vector_to_numpy(result.eigenvalues),
            complex_matrix_to_numpy(result.eigenvectors));
    }, "Eigenvalue decomposition: eigenvalues, eigenvectors = eig(A)", py::arg("A"));

    linalg.def("eig", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Eigen(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.eigenvalues, result.eigenvectors);
    }, "Eigenvalue decomposition: eigenvalues, eigenvectors = eig(A)");

    linalg.def("qr", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::QR(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(matrix_to_numpy(result.Q), matrix_to_numpy(result.R));
    }, "QR decomposition: Q, R = qr(A)", py::arg("A"));

    linalg.def("qr", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::QR(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.Q, result.R);
    }, "QR decomposition: Q, R = qr(A)");

    linalg.def("chol", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::Cholesky(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return matrix_to_numpy(result.L);
    }, "Cholesky decomposition: L = chol(A) where A = L @ L.T", py::arg("A"));

    linalg.def("chol", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Cholesky(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.L;
    }, "Cholesky decomposition: L = chol(A) where A = L @ L.T");

    linalg.def("lu", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::LU(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(
            matrix_to_numpy(result.L),
            matrix_to_numpy(result.U),
            int_vector_to_numpy(result.P));
    }, "LU decomposition: L, U, P = lu(A)", py::arg("A"));

    linalg.def("lu", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::LU(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.L, result.U, result.P);
    }, "LU decomposition: L, U, P = lu(A)");

    // Matrix properties
    linalg.def("det", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::Determinant(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Compute determinant", py::arg("A"));

    linalg.def("det", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Determinant(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Compute determinant");

    linalg.def("rank", [](const NumpyArrayDouble& A, double tol) {
        auto result = cyxwiz::LinearAlgebra::Rank(numpy_2d_to_matrix(A, "A"), tol);
        if (!result.success) throw std::runtime_error(result.error_message);
        return static_cast<int>(result.value);
    }, "Compute matrix rank", py::arg("A"), py::arg("tol") = 1e-10);

    linalg.def("rank", [](const std::vector<std::vector<double>>& A, double tol) {
        auto result = cyxwiz::LinearAlgebra::Rank(A, tol);
        if (!result.success) throw std::runtime_error(result.error_message);
        return static_cast<int>(result.value);
    }, "Compute matrix rank", py::arg("A"), py::arg("tol") = 1e-10);

    linalg.def("trace", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::Trace(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Compute trace (sum of diagonal)", py::arg("A"));

    linalg.def("trace", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Trace(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Compute trace (sum of diagonal)");

    linalg.def("norm", [](const cyxwiz::Tensor& A) {
        auto result = cyxwiz::LinearAlgebra::FrobeniusNorm(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Frobenius norm");

    linalg.def("norm", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::FrobeniusNorm(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Frobenius norm", py::arg("A"));

    linalg.def("norm", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::FrobeniusNorm(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Frobenius norm");

    linalg.def("cond", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::ConditionNumber(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Condition number", py::arg("A"));

    linalg.def("cond", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::ConditionNumber(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Condition number");

    // Matrix operations
    linalg.def("inv", [](const cyxwiz::Tensor& A) {
        auto result = cyxwiz::LinearAlgebra::Inverse(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.tensor;
    }, "Matrix inverse");

    linalg.def("inv", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::Inverse(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return matrix_to_numpy(result.matrix);
    }, "Matrix inverse", py::arg("A"));

    linalg.def("inv", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Inverse(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Matrix inverse");

    linalg.def("transpose", [](const cyxwiz::Tensor& A) {
        auto result = cyxwiz::LinearAlgebra::Transpose(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.tensor;
    }, "Matrix transpose");

    linalg.def("transpose", [](const NumpyArrayDouble& A) {
        auto result = cyxwiz::LinearAlgebra::Transpose(numpy_2d_to_matrix(A, "A"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return matrix_to_numpy(result.matrix);
    }, "Matrix transpose", py::arg("A"));

    linalg.def("transpose", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Transpose(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Matrix transpose");

    linalg.def("solve", [](const cyxwiz::Tensor& A, const cyxwiz::Tensor& b) {
        auto result = cyxwiz::LinearAlgebra::Solve(A, b);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.tensor;
    }, "Solve Ax = b");

    linalg.def("solve", [](const NumpyArrayDouble& A, const NumpyArrayDouble& b) -> py::object {
        auto result = cyxwiz::LinearAlgebra::Solve(
            numpy_2d_to_matrix(A, "A"),
            numpy_vector_or_2d_to_matrix(b, "b"));
        if (!result.success) throw std::runtime_error(result.error_message);
        if (b.ndim() == 1) {
            return vector_to_numpy(matrix_single_column_to_vector(result.matrix, "solve"));
        }
        return matrix_to_numpy(result.matrix);
    }, "Solve Ax = b", py::arg("A"), py::arg("b"));

    linalg.def("solve", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
        auto result = cyxwiz::LinearAlgebra::Solve(A, b);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Solve Ax = b");

    linalg.def("lstsq", [](const cyxwiz::Tensor& A, const cyxwiz::Tensor& b) {
        auto result = cyxwiz::LinearAlgebra::LeastSquares(A, b);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.tensor;
    }, "Least squares solution");

    linalg.def("lstsq", [](const NumpyArrayDouble& A, const NumpyArrayDouble& b) -> py::object {
        auto result = cyxwiz::LinearAlgebra::LeastSquares(
            numpy_2d_to_matrix(A, "A"),
            numpy_vector_or_2d_to_matrix(b, "b"));
        if (!result.success) throw std::runtime_error(result.error_message);
        if (b.ndim() == 1) {
            return vector_to_numpy(matrix_single_column_to_vector(result.matrix, "lstsq"));
        }
        return matrix_to_numpy(result.matrix);
    }, "Least squares solution", py::arg("A"), py::arg("b"));

    linalg.def("lstsq", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
        auto result = cyxwiz::LinearAlgebra::LeastSquares(A, b);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Least squares solution");

    linalg.def("matmul", [](const cyxwiz::Tensor& A, const cyxwiz::Tensor& B) {
        auto result = cyxwiz::LinearAlgebra::Multiply(A, B);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.tensor;
    }, "Matrix multiplication");

    linalg.def("matmul", [](const NumpyArrayDouble& A, const NumpyArrayDouble& B) {
        auto result = cyxwiz::LinearAlgebra::Multiply(
            numpy_2d_to_matrix(A, "A"),
            numpy_2d_to_matrix(B, "B"));
        if (!result.success) throw std::runtime_error(result.error_message);
        return matrix_to_numpy(result.matrix);
    }, "Matrix multiplication", py::arg("A"), py::arg("B"));

    linalg.def("matmul", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
        auto result = cyxwiz::LinearAlgebra::Multiply(A, B);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Matrix multiplication");

    // ============================================================================
    // SIGNAL PROCESSING SUBMODULE
    // ============================================================================
    auto signal = m.def_submodule("signal", "Signal processing functions (MATLAB-style)");

    signal.def("fft", [](const std::vector<double>& x, double sample_rate) {
        auto result = cyxwiz::SignalProcessing::FFT(x, sample_rate);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "magnitude"_a = result.magnitude,
            "phase"_a = result.phase,
            "frequencies"_a = result.frequencies,
            "complex"_a = result.complex_output
        );
    }, "Fast Fourier Transform", py::arg("x"), py::arg("sample_rate") = 1.0);

    signal.def("ifft", [](const std::vector<std::complex<double>>& X) {
        return cyxwiz::SignalProcessing::IFFT(X);
    }, "Inverse FFT");

    signal.def("conv", [](const std::vector<double>& x, const std::vector<double>& h, const std::string& mode) {
        auto result = cyxwiz::SignalProcessing::Convolve1D(x, h, mode);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.output;
    }, "1D Convolution", py::arg("x"), py::arg("h"), py::arg("mode") = "same");

    signal.def("conv2", [](const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& h, const std::string& mode) {
        auto result = cyxwiz::SignalProcessing::Convolve2D(x, h, mode);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.output;
    }, "2D Convolution", py::arg("x"), py::arg("h"), py::arg("mode") = "same");

    signal.def("spectrogram", [](const std::vector<double>& x, int window_size, int hop_size, double sample_rate, const std::string& window) {
        auto result = cyxwiz::SignalProcessing::ComputeSpectrogram(x, window_size, hop_size, sample_rate, window);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "S"_a = result.spectrogram,
            "frequencies"_a = result.frequencies,
            "times"_a = result.times
        );
    }, "Compute spectrogram (STFT)", py::arg("x"), py::arg("window_size") = 256, py::arg("hop_size") = 128, py::arg("sample_rate") = 1.0, py::arg("window") = "hann");

    signal.def("lowpass", [](double cutoff, double fs, int order) {
        auto result = cyxwiz::SignalProcessing::DesignLowpass(cutoff, fs, order);
        return py::dict("b"_a = result.b, "a"_a = result.a);
    }, "Design lowpass filter", py::arg("cutoff"), py::arg("fs"), py::arg("order") = 4);

    signal.def("highpass", [](double cutoff, double fs, int order) {
        auto result = cyxwiz::SignalProcessing::DesignHighpass(cutoff, fs, order);
        return py::dict("b"_a = result.b, "a"_a = result.a);
    }, "Design highpass filter", py::arg("cutoff"), py::arg("fs"), py::arg("order") = 4);

    signal.def("bandpass", [](double low, double high, double fs, int order) {
        auto result = cyxwiz::SignalProcessing::DesignBandpass(low, high, fs, order);
        return py::dict("b"_a = result.b, "a"_a = result.a);
    }, "Design bandpass filter", py::arg("low"), py::arg("high"), py::arg("fs"), py::arg("order") = 4);

    signal.def("filter", [](const std::vector<double>& x, const std::vector<double>& b, const std::vector<double>& a) {
        cyxwiz::FilterCoefficients coeffs;
        coeffs.b = b;
        coeffs.a = a;
        return cyxwiz::SignalProcessing::ApplyFilter(x, coeffs);
    }, "Apply filter to signal", py::arg("x"), py::arg("b"), py::arg("a"));

    signal.def("findpeaks", [](const std::vector<double>& x, double min_height, int min_distance) {
        auto peaks = cyxwiz::SignalProcessing::FindPeaks(x, min_height, min_distance);
        std::vector<int> indices;
        std::vector<double> values;
        for (const auto& p : peaks) {
            indices.push_back(p.index);
            values.push_back(p.value);
        }
        return py::dict("indices"_a = indices, "values"_a = values);
    }, "Find peaks in signal", py::arg("x"), py::arg("min_height") = 0.0, py::arg("min_distance") = 1);

    // Signal generation
    signal.def("sine", [](double freq, double fs, int n, double amp, double phase) {
        return cyxwiz::SignalProcessing::GenerateSineWave(freq, fs, n, amp, phase);
    }, "Generate sine wave", py::arg("freq"), py::arg("fs"), py::arg("n"), py::arg("amp") = 1.0, py::arg("phase") = 0.0);

    signal.def("square", [](double freq, double fs, int n, double amp) {
        return cyxwiz::SignalProcessing::GenerateSquareWave(freq, fs, n, amp);
    }, "Generate square wave", py::arg("freq"), py::arg("fs"), py::arg("n"), py::arg("amp") = 1.0);

    signal.def("noise", [](int n, double amp) {
        return cyxwiz::SignalProcessing::GenerateWhiteNoise(n, amp);
    }, "Generate white noise", py::arg("n"), py::arg("amp") = 1.0);

    // ============================================================================
    // STATISTICS/CLUSTERING SUBMODULE
    // ============================================================================
    auto stats = m.def_submodule("stats", "Statistics and clustering functions");

    // Clustering
    stats.def("kmeans", [](const std::vector<std::vector<double>>& data, int k, int max_iter, const std::string& init) {
        auto result = cyxwiz::Clustering::KMeans(data, k, max_iter, init);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "labels"_a = result.labels,
            "centroids"_a = result.centroids,
            "inertia"_a = result.inertia,
            "n_iterations"_a = result.n_iterations,
            "converged"_a = result.converged
        );
    }, "K-Means clustering", py::arg("data"), py::arg("k"), py::arg("max_iter") = 300, py::arg("init") = "kmeans++");

    stats.def("dbscan", [](const std::vector<std::vector<double>>& data, double eps, int min_samples) {
        auto result = cyxwiz::Clustering::DBSCAN(data, eps, min_samples);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "labels"_a = result.labels,
            "n_clusters"_a = result.n_clusters,
            "n_noise"_a = result.n_noise_points
        );
    }, "DBSCAN clustering", py::arg("data"), py::arg("eps"), py::arg("min_samples") = 5);

    stats.def("gmm", [](const std::vector<std::vector<double>>& data, int n_components, const std::string& cov_type) {
        auto result = cyxwiz::Clustering::GMM(data, n_components, cov_type);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "labels"_a = result.labels,
            "means"_a = result.means,
            "weights"_a = result.weights,
            "aic"_a = result.aic,
            "bic"_a = result.bic
        );
    }, "Gaussian Mixture Model", py::arg("data"), py::arg("n_components"), py::arg("cov_type") = "full");

    // Dimensionality reduction
    stats.def("pca", [](const std::vector<std::vector<double>>& data, int n_components) {
        auto result = cyxwiz::DimensionalityReduction::ComputePCA(data, n_components);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "transformed"_a = result.transformed,
            "components"_a = result.components,
            "explained_variance"_a = result.explained_variance_ratio
        );
    }, "Principal Component Analysis", py::arg("data"), py::arg("n_components") = 2);

    stats.def("tsne", [](const std::vector<std::vector<double>>& data, int n_dims, int perplexity) {
        auto result = cyxwiz::DimensionalityReduction::ComputetSNE(data, n_dims, perplexity);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.embeddings;
    }, "t-SNE embedding", py::arg("data"), py::arg("n_dims") = 2, py::arg("perplexity") = 30);

    // Evaluation metrics
    stats.def("silhouette", [](const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
        return cyxwiz::Clustering::ComputeSilhouetteScore(data, labels);
    }, "Silhouette score");

    stats.def("confusion_matrix", [](const std::vector<int>& y_true, const std::vector<int>& y_pred) {
        auto result = cyxwiz::ModelEvaluation::ComputeConfusionMatrix(y_true, y_pred);
        return py::dict(
            "matrix"_a = result.matrix,
            "accuracy"_a = result.accuracy,
            "precision"_a = result.precision,
            "recall"_a = result.recall,
            "f1"_a = result.f1_scores
        );
    }, "Compute confusion matrix");

    stats.def("roc", [](const std::vector<int>& y_true, const std::vector<double>& y_scores) {
        auto result = cyxwiz::ModelEvaluation::ComputeROC(y_true, y_scores);
        return py::dict(
            "fpr"_a = result.fpr,
            "tpr"_a = result.tpr,
            "auc"_a = result.auc
        );
    }, "ROC curve and AUC");

    // ============================================================================
    // TIME SERIES SUBMODULE
    // ============================================================================
    auto timeseries = m.def_submodule("timeseries", "Time series analysis functions");

    timeseries.def("acf", [](const std::vector<double>& data, int max_lag) {
        auto result = cyxwiz::TimeSeries::ComputeACF(data, max_lag);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "acf"_a = result.acf,
            "lags"_a = result.lags,
            "confidence_upper"_a = result.confidence_upper,
            "confidence_lower"_a = result.confidence_lower
        );
    }, "Autocorrelation function", py::arg("data"), py::arg("max_lag") = -1);

    timeseries.def("pacf", [](const std::vector<double>& data, int max_lag) {
        auto result = cyxwiz::TimeSeries::ComputePACF(data, max_lag);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.pacf;
    }, "Partial autocorrelation function", py::arg("data"), py::arg("max_lag") = -1);

    timeseries.def("decompose", [](const std::vector<double>& data, int period, const std::string& method) {
        auto result = cyxwiz::TimeSeries::Decompose(data, period, method);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "trend"_a = result.trend,
            "seasonal"_a = result.seasonal,
            "residual"_a = result.residual
        );
    }, "Time series decomposition", py::arg("data"), py::arg("period"), py::arg("method") = "additive");

    timeseries.def("stationarity", [](const std::vector<double>& data) {
        auto result = cyxwiz::TimeSeries::TestStationarity(data);
        return py::dict(
            "is_stationary"_a = result.is_stationary,
            "adf_statistic"_a = result.adf_statistic,
            "adf_pvalue"_a = result.adf_pvalue,
            "kpss_statistic"_a = result.kpss_statistic,
            "kpss_pvalue"_a = result.kpss_pvalue,
            "suggested_differencing"_a = result.suggested_differencing
        );
    }, "Test for stationarity (ADF + KPSS)");

    timeseries.def("arima", [](const std::vector<double>& data, int horizon, int p, int d, int q) {
        auto result = cyxwiz::TimeSeries::ARIMA(data, horizon, p, d, q);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::dict(
            "forecast"_a = result.forecast,
            "lower"_a = result.lower_bound,
            "upper"_a = result.upper_bound,
            "mse"_a = result.mse,
            "aic"_a = result.aic
        );
    }, "ARIMA forecasting", py::arg("data"), py::arg("horizon"), py::arg("p") = -1, py::arg("d") = -1, py::arg("q") = -1);

    timeseries.def("diff", [](const std::vector<double>& data, int order) {
        return cyxwiz::TimeSeries::Difference(data, order);
    }, "Difference series", py::arg("data"), py::arg("order") = 1);

    timeseries.def("rolling_mean", [](const std::vector<double>& data, int window) {
        return cyxwiz::TimeSeries::RollingMean(data, window);
    }, "Rolling mean", py::arg("data"), py::arg("window"));

    timeseries.def("rolling_std", [](const std::vector<double>& data, int window) {
        return cyxwiz::TimeSeries::RollingStd(data, window);
    }, "Rolling standard deviation", py::arg("data"), py::arg("window"));

    // WindowConfig struct
    py::class_<cyxwiz::TimeSeries::WindowConfig>(timeseries, "WindowConfig")
        .def(py::init<>())
        .def_readwrite("window_size", &cyxwiz::TimeSeries::WindowConfig::window_size)
        .def_readwrite("forecast_horizon", &cyxwiz::TimeSeries::WindowConfig::forecast_horizon)
        .def_readwrite("stride", &cyxwiz::TimeSeries::WindowConfig::stride)
        .def_readwrite("lag_values", &cyxwiz::TimeSeries::WindowConfig::lag_values)
        .def_readwrite("rolling_windows", &cyxwiz::TimeSeries::WindowConfig::rolling_windows)
        .def_readwrite("add_diff_features", &cyxwiz::TimeSeries::WindowConfig::add_diff_features)
        .def_readwrite("normalize", &cyxwiz::TimeSeries::WindowConfig::normalize);

    // WindowResult struct
    py::class_<cyxwiz::TimeSeries::WindowResult>(timeseries, "WindowResult")
        .def_readonly("X", &cyxwiz::TimeSeries::WindowResult::X)
        .def_readonly("y", &cyxwiz::TimeSeries::WindowResult::y)
        .def_readonly("num_windows", &cyxwiz::TimeSeries::WindowResult::num_windows)
        .def_readonly("input_features", &cyxwiz::TimeSeries::WindowResult::input_features)
        .def_readonly("target_features", &cyxwiz::TimeSeries::WindowResult::target_features)
        .def_readonly("success", &cyxwiz::TimeSeries::WindowResult::success)
        .def_readonly("error_message", &cyxwiz::TimeSeries::WindowResult::error_message);

    timeseries.def("create_windows", [](const std::vector<double>& data,
                                        const cyxwiz::TimeSeries::WindowConfig& config) {
        return cyxwiz::TimeSeries::CreateWindows(data, config);
    }, "Create sliding windows for ML", py::arg("data"), py::arg("config"));

    timeseries.def("create_multivariate_windows", [](const std::vector<std::vector<double>>& data,
                                                     int target_col,
                                                     const cyxwiz::TimeSeries::WindowConfig& config) {
        return cyxwiz::TimeSeries::CreateMultivariateWindows(data, target_col, config);
    }, "Create multivariate sliding windows", py::arg("data"), py::arg("target_col"), py::arg("config"));

    timeseries.def("add_features", [](const std::vector<double>& data,
                                      const std::vector<int>& lags,
                                      const std::vector<int>& rolling,
                                      bool add_diff) {
        return cyxwiz::TimeSeries::AddFeatures(data, lags, rolling, add_diff);
    }, "Add engineered time-series features",
       py::arg("data"), py::arg("lag_values") = std::vector<int>{},
       py::arg("rolling_windows") = std::vector<int>{}, py::arg("add_diff") = false);

    timeseries.def("chronological_split", [](size_t n, double train, double val) {
        return cyxwiz::TimeSeries::ChronologicalSplit(n, train, val);
    }, "Chronological train/val/test split",
       py::arg("num_samples"), py::arg("train_ratio") = 0.7, py::arg("val_ratio") = 0.15);

    // ============================================================================
    // Audio Processing
    // ============================================================================

    py::class_<cyxwiz::AudioData>(m, "AudioData")
        .def(py::init<>())
        .def_readwrite("samples", &cyxwiz::AudioData::samples)
        .def_readwrite("sample_rate", &cyxwiz::AudioData::sample_rate)
        .def_readwrite("num_samples", &cyxwiz::AudioData::num_samples)
        .def_readwrite("duration_seconds", &cyxwiz::AudioData::duration_seconds)
        .def_readonly("valid", &cyxwiz::AudioData::valid)
        .def_readonly("error_message", &cyxwiz::AudioData::error_message);

    py::class_<cyxwiz::SpectrogramConfig>(m, "SpectrogramConfig")
        .def(py::init<>())
        .def_readwrite("n_fft", &cyxwiz::SpectrogramConfig::n_fft)
        .def_readwrite("hop_length", &cyxwiz::SpectrogramConfig::hop_length)
        .def_readwrite("win_length", &cyxwiz::SpectrogramConfig::win_length)
        .def_readwrite("center", &cyxwiz::SpectrogramConfig::center)
        .def_readwrite("window_type", &cyxwiz::SpectrogramConfig::window_type);

    py::class_<cyxwiz::MelConfig, cyxwiz::SpectrogramConfig>(m, "MelConfig")
        .def(py::init<>())
        .def_readwrite("n_mels", &cyxwiz::MelConfig::n_mels)
        .def_readwrite("fmin", &cyxwiz::MelConfig::fmin)
        .def_readwrite("fmax", &cyxwiz::MelConfig::fmax);

    py::class_<cyxwiz::MFCCConfig, cyxwiz::MelConfig>(m, "MFCCConfig")
        .def(py::init<>())
        .def_readwrite("n_mfcc", &cyxwiz::MFCCConfig::n_mfcc)
        .def_readwrite("use_energy", &cyxwiz::MFCCConfig::use_energy);

    py::class_<cyxwiz::AudioFeatures>(m, "AudioFeatures")
        .def_readonly("data", &cyxwiz::AudioFeatures::data)
        .def_readonly("rows", &cyxwiz::AudioFeatures::rows)
        .def_readonly("cols", &cyxwiz::AudioFeatures::cols)
        .def_readonly("valid", &cyxwiz::AudioFeatures::valid)
        .def_readonly("error_message", &cyxwiz::AudioFeatures::error_message);

    py::class_<cyxwiz::AudioProcessing>(m, "AudioProcessing")
        .def_static("load_audio", &cyxwiz::AudioProcessing::LoadAudio,
                    py::arg("filepath"), py::arg("target_sr") = 16000)
        .def_static("compute_spectrogram", &cyxwiz::AudioProcessing::ComputeSpectrogram,
                    py::arg("audio"), py::arg("config") = cyxwiz::SpectrogramConfig{})
        .def_static("compute_mel_spectrogram", &cyxwiz::AudioProcessing::ComputeMelSpectrogram,
                    py::arg("audio"), py::arg("config") = cyxwiz::MelConfig{})
        .def_static("compute_mfcc", &cyxwiz::AudioProcessing::ComputeMFCC,
                    py::arg("audio"), py::arg("config") = cyxwiz::MFCCConfig{})
        .def_static("add_noise", &cyxwiz::AudioProcessing::AddNoise,
                    py::arg("audio"), py::arg("snr_db") = 20.0f)
        .def_static("time_stretch", &cyxwiz::AudioProcessing::TimeStretch,
                    py::arg("audio"), py::arg("rate") = 1.0f)
        .def_static("pitch_shift", &cyxwiz::AudioProcessing::PitchShift,
                    py::arg("audio"), py::arg("semitones") = 0.0f)
        .def_static("resample", &cyxwiz::AudioProcessing::Resample,
                    py::arg("audio"), py::arg("target_sr"))
        .def_static("normalize", &cyxwiz::AudioProcessing::Normalize,
                    py::arg("audio"))
        .def_static("trim_silence", &cyxwiz::AudioProcessing::TrimSilence,
                    py::arg("audio"), py::arg("threshold_db") = -40.0f);

    // ============================================================================
    // Activation Functions
    // ============================================================================

    // ActivationType enum
    // Note: Not using .export_values() to avoid conflicts with class names like ReLU, Sigmoid
    py::enum_<cyxwiz::ActivationType>(m, "ActivationType")
        .value("ReLU", cyxwiz::ActivationType::ReLU)
        .value("Sigmoid", cyxwiz::ActivationType::Sigmoid)
        .value("Tanh", cyxwiz::ActivationType::Tanh)
        .value("Softmax", cyxwiz::ActivationType::Softmax)
        .value("LeakyReLU", cyxwiz::ActivationType::LeakyReLU)
        .value("ELU", cyxwiz::ActivationType::ELU)
        .value("GELU", cyxwiz::ActivationType::GELU)
        .value("Swish", cyxwiz::ActivationType::Swish)
        .value("SiLU", cyxwiz::ActivationType::SiLU)
        .value("Mish", cyxwiz::ActivationType::Mish)
        .value("Hardswish", cyxwiz::ActivationType::Hardswish)
        .value("SELU", cyxwiz::ActivationType::SELU)
        .value("PReLU", cyxwiz::ActivationType::PReLU);

    // Activation base class
    py::class_<cyxwiz::Activation>(m, "Activation")
        .def("forward", &cyxwiz::Activation::Forward,
             py::arg("input"),
             "Apply activation function")
        .def("backward", &cyxwiz::Activation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Compute gradient of activation")
        .def("get_name", &cyxwiz::Activation::GetName,
             "Get activation name");

    // Factory function
    m.def("create_activation", &cyxwiz::CreateActivation,
          py::arg("type"), py::arg("alpha") = 0.01f,
          "Create an activation function by type");

    // ReLU Activation
    py::class_<cyxwiz::ReLUActivation, cyxwiz::Activation>(m, "ReLU")
        .def(py::init<>(), "Create ReLU activation")
        .def("forward", &cyxwiz::ReLUActivation::Forward)
        .def("backward", &cyxwiz::ReLUActivation::Backward);

    // LeakyReLU Activation
    py::class_<cyxwiz::LeakyReLUActivation, cyxwiz::Activation>(m, "LeakyReLU")
        .def(py::init<float>(), py::arg("alpha") = 0.01f,
             "Create LeakyReLU activation with negative slope alpha")
        .def("forward", &cyxwiz::LeakyReLUActivation::Forward)
        .def("backward", &cyxwiz::LeakyReLUActivation::Backward)
        .def("get_alpha", &cyxwiz::LeakyReLUActivation::GetAlpha);

    // ELU Activation
    py::class_<cyxwiz::ELUActivation, cyxwiz::Activation>(m, "ELU")
        .def(py::init<float>(), py::arg("alpha") = 1.0f,
             "Create ELU activation")
        .def("forward", &cyxwiz::ELUActivation::Forward)
        .def("backward", &cyxwiz::ELUActivation::Backward)
        .def("get_alpha", &cyxwiz::ELUActivation::GetAlpha);

    // GELU Activation
    py::class_<cyxwiz::GELUActivation, cyxwiz::Activation>(m, "GELU")
        .def(py::init<>(), "Create GELU activation")
        .def("forward", &cyxwiz::GELUActivation::Forward)
        .def("backward", &cyxwiz::GELUActivation::Backward);

    // Swish/SiLU Activation
    py::class_<cyxwiz::SwishActivation, cyxwiz::Activation>(m, "Swish")
        .def(py::init<>(), "Create Swish (SiLU) activation")
        .def("forward", &cyxwiz::SwishActivation::Forward)
        .def("backward", &cyxwiz::SwishActivation::Backward);

    m.attr("SiLU") = m.attr("Swish");  // Alias

    // Sigmoid Activation
    py::class_<cyxwiz::SigmoidActivation, cyxwiz::Activation>(m, "Sigmoid")
        .def(py::init<>(), "Create Sigmoid activation")
        .def("forward", &cyxwiz::SigmoidActivation::Forward)
        .def("backward", &cyxwiz::SigmoidActivation::Backward);

    // Tanh Activation
    py::class_<cyxwiz::TanhActivation, cyxwiz::Activation>(m, "Tanh")
        .def(py::init<>(), "Create Tanh activation")
        .def("forward", &cyxwiz::TanhActivation::Forward)
        .def("backward", &cyxwiz::TanhActivation::Backward);

    // Softmax Activation
    py::class_<cyxwiz::SoftmaxActivation, cyxwiz::Activation>(m, "Softmax")
        .def(py::init<int>(), py::arg("axis") = -1,
             "Create Softmax activation")
        .def("forward", &cyxwiz::SoftmaxActivation::Forward)
        .def("backward", &cyxwiz::SoftmaxActivation::Backward);

    // Mish Activation
    py::class_<cyxwiz::MishActivation, cyxwiz::Activation>(m, "Mish")
        .def(py::init<>(), "Create Mish activation")
        .def("forward", &cyxwiz::MishActivation::Forward)
        .def("backward", &cyxwiz::MishActivation::Backward);

    // Hardswish Activation
    py::class_<cyxwiz::HardswishActivation, cyxwiz::Activation>(m, "Hardswish")
        .def(py::init<>(), "Create Hardswish activation")
        .def("forward", &cyxwiz::HardswishActivation::Forward)
        .def("backward", &cyxwiz::HardswishActivation::Backward);

    // SELU Activation (Scaled Exponential Linear Unit)
    py::class_<cyxwiz::SELUActivation, cyxwiz::Activation>(m, "SELU")
        .def(py::init<>(), "Create SELU activation (self-normalizing)")
        .def("forward", &cyxwiz::SELUActivation::Forward)
        .def("backward", &cyxwiz::SELUActivation::Backward)
        .def_readonly_static("ALPHA", &cyxwiz::SELUActivation::ALPHA)
        .def_readonly_static("SCALE", &cyxwiz::SELUActivation::SCALE);

    // PReLU Activation (Parametric ReLU)
    py::class_<cyxwiz::PReLUActivation, cyxwiz::Activation>(m, "PReLU")
        .def(py::init<int, float>(),
             py::arg("num_parameters") = 1,
             py::arg("init") = 0.25f,
             "Create PReLU activation with learnable alpha")
        .def("forward", &cyxwiz::PReLUActivation::Forward)
        .def("backward", &cyxwiz::PReLUActivation::Backward)
        .def("get_alpha", &cyxwiz::PReLUActivation::GetAlpha,
             "Get learnable alpha parameter")
        .def("set_alpha", &cyxwiz::PReLUActivation::SetAlpha,
             py::arg("alpha"),
             "Set alpha parameter")
        .def("get_alpha_gradient", &cyxwiz::PReLUActivation::GetAlphaGradient,
             "Get gradient for alpha parameter");

    // ============================================================================
    // SEQUENTIAL MODEL
    // ============================================================================

    // ModuleType enum
    py::enum_<cyxwiz::ModuleType>(m, "ModuleType")
        .value("Linear", cyxwiz::ModuleType::Linear)
        .value("ReLU", cyxwiz::ModuleType::ReLU)
        .value("Sigmoid", cyxwiz::ModuleType::Sigmoid)
        .value("Tanh", cyxwiz::ModuleType::Tanh)
        .value("Softmax", cyxwiz::ModuleType::Softmax)
        .value("Dropout", cyxwiz::ModuleType::Dropout)
        .value("BatchNorm", cyxwiz::ModuleType::BatchNorm)
        .value("Flatten", cyxwiz::ModuleType::Flatten)
        .value("LeakyReLU", cyxwiz::ModuleType::LeakyReLU)
        .value("ELU", cyxwiz::ModuleType::ELU)
        .value("GELU", cyxwiz::ModuleType::GELU)
        .value("Swish", cyxwiz::ModuleType::Swish)
        .value("Mish", cyxwiz::ModuleType::Mish)
        .export_values();

    // Module base class
    py::class_<cyxwiz::Module>(m, "Module")
        .def("forward", &cyxwiz::Module::Forward,
             py::arg("input"),
             "Forward pass through the module")
        .def("backward", &cyxwiz::Module::Backward,
             py::arg("grad_output"),
             "Backward pass")
        .def("get_parameters", &cyxwiz::Module::GetParameters,
             "Get module parameters")
        .def("set_parameters", &cyxwiz::Module::SetParameters,
             py::arg("params"),
             "Set module parameters")
        .def("get_gradients", &cyxwiz::Module::GetGradients,
             "Get parameter gradients")
        .def("has_parameters", &cyxwiz::Module::HasParameters,
             "Check if module has trainable parameters")
        .def("get_name", &cyxwiz::Module::GetName,
             "Get module name")
        .def("set_training", &cyxwiz::Module::SetTraining,
             py::arg("training"),
             "Set training mode")
        .def("is_training", &cyxwiz::Module::IsTraining,
             "Check if in training mode")
        .def("freeze", &cyxwiz::Module::Freeze,
             "Freeze module (disable parameter updates)")
        .def("unfreeze", &cyxwiz::Module::Unfreeze,
             "Unfreeze module (enable parameter updates)")
        .def("is_trainable", &cyxwiz::Module::IsTrainable,
             "Check if module is trainable");

    // LinearModule
    py::class_<cyxwiz::LinearModule, cyxwiz::Module>(m, "LinearModule")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("use_bias") = true,
             "Create a Linear module");

    // ReLUModule
    py::class_<cyxwiz::ReLUModule, cyxwiz::Module>(m, "ReLUModule")
        .def(py::init<>(), "Create a ReLU module");

    // SigmoidModule
    py::class_<cyxwiz::SigmoidModule, cyxwiz::Module>(m, "SigmoidModule")
        .def(py::init<>(), "Create a Sigmoid module");

    // TanhModule
    py::class_<cyxwiz::TanhModule, cyxwiz::Module>(m, "TanhModule")
        .def(py::init<>(), "Create a Tanh module");

    // SoftmaxModule
    py::class_<cyxwiz::SoftmaxModule, cyxwiz::Module>(m, "SoftmaxModule")
        .def(py::init<int>(), py::arg("dim") = -1,
             "Create a Softmax module");

    // DropoutModule
    py::class_<cyxwiz::DropoutModule, cyxwiz::Module>(m, "DropoutModule")
        .def(py::init<float>(), py::arg("p") = 0.5f,
             "Create a Dropout module");

    // FlattenModule
    py::class_<cyxwiz::FlattenModule, cyxwiz::Module>(m, "FlattenModule")
        .def(py::init<int>(), py::arg("start_dim") = 1,
             "Create a Flatten module");

    // LeakyReLUModule
    py::class_<cyxwiz::LeakyReLUModule, cyxwiz::Module>(m, "LeakyReLUModule")
        .def(py::init<float>(), py::arg("negative_slope") = 0.01f,
             "Create a LeakyReLU module");

    // ELUModule
    py::class_<cyxwiz::ELUModule, cyxwiz::Module>(m, "ELUModule")
        .def(py::init<float>(), py::arg("alpha") = 1.0f,
             "Create an ELU module");

    // GELUModule
    py::class_<cyxwiz::GELUModule, cyxwiz::Module>(m, "GELUModule")
        .def(py::init<>(), "Create a GELU module");

    // SwishModule
    py::class_<cyxwiz::SwishModule, cyxwiz::Module>(m, "SwishModule")
        .def(py::init<>(), "Create a Swish module");

    // MishModule
    py::class_<cyxwiz::MishModule, cyxwiz::Module>(m, "MishModule")
        .def(py::init<>(), "Create a Mish module");

    // SequentialModel - the main model class
    py::class_<cyxwiz::SequentialModel>(m, "Sequential",
        "Sequential model container for building neural networks")
        .def(py::init<>(), "Create an empty Sequential model")

        // Layer addition methods (since templates can't be directly exposed)
        .def("add_linear", [](cyxwiz::SequentialModel& self, size_t in_features, size_t out_features, bool use_bias) {
            self.Add<cyxwiz::LinearModule>(in_features, out_features, use_bias);
        }, py::arg("in_features"), py::arg("out_features"), py::arg("use_bias") = true,
           "Add a Linear layer")

        .def("add_relu", [](cyxwiz::SequentialModel& self) {
            self.Add<cyxwiz::ReLUModule>();
        }, "Add a ReLU activation")

        .def("add_sigmoid", [](cyxwiz::SequentialModel& self) {
            self.Add<cyxwiz::SigmoidModule>();
        }, "Add a Sigmoid activation")

        .def("add_tanh", [](cyxwiz::SequentialModel& self) {
            self.Add<cyxwiz::TanhModule>();
        }, "Add a Tanh activation")

        .def("add_softmax", [](cyxwiz::SequentialModel& self, int dim) {
            self.Add<cyxwiz::SoftmaxModule>(dim);
        }, py::arg("dim") = -1, "Add a Softmax activation")

        .def("add_dropout", [](cyxwiz::SequentialModel& self, float p) {
            self.Add<cyxwiz::DropoutModule>(p);
        }, py::arg("p") = 0.5f, "Add a Dropout layer")

        .def("add_flatten", [](cyxwiz::SequentialModel& self, int start_dim) {
            self.Add<cyxwiz::FlattenModule>(start_dim);
        }, py::arg("start_dim") = 1, "Add a Flatten layer")

        .def("add_leaky_relu", [](cyxwiz::SequentialModel& self, float negative_slope) {
            self.Add<cyxwiz::LeakyReLUModule>(negative_slope);
        }, py::arg("negative_slope") = 0.01f, "Add a LeakyReLU activation")

        .def("add_elu", [](cyxwiz::SequentialModel& self, float alpha) {
            self.Add<cyxwiz::ELUModule>(alpha);
        }, py::arg("alpha") = 1.0f, "Add an ELU activation")

        .def("add_gelu", [](cyxwiz::SequentialModel& self) {
            self.Add<cyxwiz::GELUModule>();
        }, "Add a GELU activation")

        .def("add_swish", [](cyxwiz::SequentialModel& self) {
            self.Add<cyxwiz::SwishModule>();
        }, "Add a Swish activation")

        .def("add_mish", [](cyxwiz::SequentialModel& self) {
            self.Add<cyxwiz::MishModule>();
        }, "Add a Mish activation")

        // Core methods
        .def("forward", &cyxwiz::SequentialModel::Forward,
             py::arg("input"),
             "Forward pass through all layers")
        .def("backward", &cyxwiz::SequentialModel::Backward,
             py::arg("grad_output"),
             "Backward pass through all layers (reverse order)")
        .def("get_parameters", &cyxwiz::SequentialModel::GetParameters,
             "Get all trainable parameters")
        .def("set_parameters", &cyxwiz::SequentialModel::SetParameters,
             py::arg("params"),
             "Set all trainable parameters")
        .def("get_gradients", &cyxwiz::SequentialModel::GetGradients,
             "Get all parameter gradients")
        .def("update_parameters", &cyxwiz::SequentialModel::UpdateParameters,
             py::arg("optimizer"),
             "Apply optimizer to all parameters")

        // Training control
        .def("set_training", &cyxwiz::SequentialModel::SetTraining,
             py::arg("training"),
             "Set training mode for all modules")
        .def("train", [](cyxwiz::SequentialModel& self) {
            self.SetTraining(true);
        }, "Set model to training mode")
        .def("eval", [](cyxwiz::SequentialModel& self) {
            self.SetTraining(false);
        }, "Set model to evaluation mode")

        // Model info
        .def("size", &cyxwiz::SequentialModel::Size,
             "Get number of modules")
        .def("summary", &cyxwiz::SequentialModel::Summary,
             "Print model summary")
        .def("__len__", &cyxwiz::SequentialModel::Size)

        // Persistence
        .def("save", &cyxwiz::SequentialModel::Save,
             py::arg("path"),
             "Save model to file")
        .def("load", &cyxwiz::SequentialModel::Load,
             py::arg("path"),
             "Load model weights from file")

        // Metadata
        .def("set_name", &cyxwiz::SequentialModel::SetName,
             py::arg("name"),
             "Set model name")
        .def("get_name", &cyxwiz::SequentialModel::GetName,
             "Get model name")
        .def("set_description", &cyxwiz::SequentialModel::SetDescription,
             py::arg("description"),
             "Set model description")
        .def("get_description", &cyxwiz::SequentialModel::GetDescription,
             "Get model description")

        // Transfer learning
        .def("freeze_layer", &cyxwiz::SequentialModel::FreezeLayer,
             py::arg("layer_idx"),
             "Freeze a specific layer by index")
        .def("freeze_up_to", &cyxwiz::SequentialModel::FreezeUpTo,
             py::arg("layer_idx"),
             "Freeze all layers up to (not including) the given index")
        .def("freeze_except_last", &cyxwiz::SequentialModel::FreezeExceptLast,
             py::arg("n"),
             "Freeze all layers except the last N layers")
        .def("unfreeze_all", &cyxwiz::SequentialModel::UnfreezeAll,
             "Unfreeze all layers")
        .def("is_layer_trainable", &cyxwiz::SequentialModel::IsLayerTrainable,
             py::arg("layer_idx"),
             "Check if a layer is trainable");

    // Factory function for creating modules from enum
    m.def("create_module", &cyxwiz::CreateModule,
          py::arg("type"), py::arg("params") = std::map<std::string, std::string>{},
          "Create a module from type enum");

    // ============================================================================
    // DATA LOADER (DuckDB Integration)
    // ============================================================================

    // DataLoaderConfig struct
    py::class_<cyxwiz::DataLoaderConfig>(m, "DataLoaderConfig",
        "Configuration for DataLoader")
        .def(py::init<>())
        .def_readwrite("batch_size", &cyxwiz::DataLoaderConfig::batch_size,
                       "Default batch size for iterators (default: 1024)")
        .def_readwrite("memory_limit_mb", &cyxwiz::DataLoaderConfig::memory_limit_mb,
                       "Memory limit in MB before warning (default: 4096)")
        .def_readwrite("num_threads", &cyxwiz::DataLoaderConfig::num_threads,
                       "Number of threads for parallel operations (default: 4)")
        .def_readwrite("verbose", &cyxwiz::DataLoaderConfig::verbose,
                       "Print verbose logging (default: false)");

    // ColumnInfo struct
    py::class_<cyxwiz::ColumnInfo>(m, "ColumnInfo",
        "Information about a column in a dataset")
        .def(py::init<>())
        .def_readwrite("name", &cyxwiz::ColumnInfo::name)
        .def_readwrite("type", &cyxwiz::ColumnInfo::type)
        .def_readwrite("nullable", &cyxwiz::ColumnInfo::nullable)
        .def_readwrite("index", &cyxwiz::ColumnInfo::index)
        .def("__repr__", [](const cyxwiz::ColumnInfo& c) {
            return "<ColumnInfo name='" + c.name + "' type='" + c.type + "'>";
        });

    // BatchIterator class
    py::class_<cyxwiz::DataLoader::BatchIterator>(m, "BatchIterator",
        "Iterator for streaming large datasets in batches")
        .def("has_next", &cyxwiz::DataLoader::BatchIterator::HasNext,
             "Check if more batches are available")
        .def("next", [](cyxwiz::DataLoader::BatchIterator& self) {
            py::gil_scoped_release release;
            return self.Next();
        }, "Get next batch as Tensor")
        .def("reset", &cyxwiz::DataLoader::BatchIterator::Reset,
             "Reset iterator to beginning")
        .def("total_rows", &cyxwiz::DataLoader::BatchIterator::TotalRows,
             "Get total number of rows")
        .def("current_batch", &cyxwiz::DataLoader::BatchIterator::CurrentBatch,
             "Get current batch index (0-based)")
        .def("batch_size", &cyxwiz::DataLoader::BatchIterator::BatchSize,
             "Get batch size")
        // Python iterator protocol
        .def("__iter__", [](cyxwiz::DataLoader::BatchIterator& self) -> cyxwiz::DataLoader::BatchIterator& {
            return self;
        })
        .def("__next__", [](cyxwiz::DataLoader::BatchIterator& self) {
            if (!self.HasNext()) {
                throw py::stop_iteration();
            }
            py::gil_scoped_release release;
            return self.Next();
        });

    // DataLoader class
    py::class_<cyxwiz::DataLoader>(m, "DataLoader",
        R"doc(High-performance data loader using DuckDB.

Supports:
- Loading Parquet, CSV, JSON files directly into Tensors
- SQL queries on files (SELECT, JOIN, WHERE, etc.)
- Batch iteration for large datasets
- Schema inspection

Example:
    loader = DataLoader()
    data = loader.load_csv("data.csv")
    result = loader.query("SELECT * FROM 'data.parquet' WHERE x > 0")

    # Batch iteration
    for batch in loader.create_batch_iterator("SELECT * FROM 'large.parquet'", 1000):
        process(batch)
)doc")
        .def(py::init<>(), "Create DataLoader with default configuration")
        .def(py::init<const cyxwiz::DataLoaderConfig&>(),
             py::arg("config"),
             "Create DataLoader with custom configuration")

        // Static methods
        .def_static("is_available", &cyxwiz::DataLoader::IsAvailable,
                    "Check if DuckDB is available")
        .def_static("get_version", &cyxwiz::DataLoader::GetVersion,
                    "Get DuckDB version string")

        // File loading
        .def("load_parquet", [](cyxwiz::DataLoader& self, const std::string& path,
                                const std::vector<std::string>& columns) {
            py::gil_scoped_release release;
            return self.LoadParquet(path, columns);
        }, py::arg("path"), py::arg("columns") = std::vector<std::string>{},
           "Load Parquet file into Tensor")

        .def("load_csv", [](cyxwiz::DataLoader& self, const std::string& path,
                            const std::vector<std::string>& columns,
                            char delimiter, bool has_header) {
            py::gil_scoped_release release;
            return self.LoadCSV(path, columns, delimiter, has_header);
        }, py::arg("path"), py::arg("columns") = std::vector<std::string>{},
           py::arg("delimiter") = ',', py::arg("has_header") = true,
           "Load CSV file into Tensor")

        .def("load_json", [](cyxwiz::DataLoader& self, const std::string& path,
                             const std::vector<std::string>& columns) {
            py::gil_scoped_release release;
            return self.LoadJSON(path, columns);
        }, py::arg("path"), py::arg("columns") = std::vector<std::string>{},
           "Load JSON file into Tensor")

        // SQL queries
        .def("query", [](cyxwiz::DataLoader& self, const std::string& sql) {
            py::gil_scoped_release release;
            return self.Query(sql);
        }, py::arg("sql"),
           R"doc(Execute SQL query and return result as Tensor.

Example queries:
    "SELECT * FROM 'data.parquet'"
    "SELECT a, b FROM 'data.csv' WHERE c > 10"
    "SELECT * FROM 'a.parquet' JOIN 'b.parquet' ON a.id = b.id"
)doc")

        .def("query_columns", [](cyxwiz::DataLoader& self, const std::string& sql) {
            py::gil_scoped_release release;
            return self.QueryColumns(sql);
        }, py::arg("sql"),
           "Execute SQL query and return result as list of column Tensors")

        // Batch iteration
        .def("create_batch_iterator", [](cyxwiz::DataLoader& self, const std::string& sql,
                                         size_t batch_size) {
            return self.CreateBatchIterator(sql, batch_size);
        }, py::arg("sql"), py::arg("batch_size") = 0,
           "Create batch iterator for streaming large datasets")

        // Schema inspection
        .def("get_schema", &cyxwiz::DataLoader::GetSchema,
             py::arg("path"),
             "Get schema information for a file")
        .def("get_columns", &cyxwiz::DataLoader::GetColumns,
             py::arg("path"),
             "Get column names for a file")
        .def("get_row_count", [](cyxwiz::DataLoader& self, const std::string& path) {
            py::gil_scoped_release release;
            return self.GetRowCount(path);
        }, py::arg("path"),
           "Get row count for a file")

        // File conversion
        .def("convert_csv_to_parquet", [](cyxwiz::DataLoader& self,
                                          const std::string& csv_path,
                                          const std::string& parquet_path,
                                          const std::string& compression) {
            py::gil_scoped_release release;
            self.ConvertCSVToParquet(csv_path, parquet_path, compression);
        }, py::arg("csv_path"), py::arg("parquet_path"),
           py::arg("compression") = "snappy",
           "Convert CSV file to Parquet format")

        .def("convert_json_to_parquet", [](cyxwiz::DataLoader& self,
                                           const std::string& json_path,
                                           const std::string& parquet_path,
                                           const std::string& compression) {
            py::gil_scoped_release release;
            self.ConvertJSONToParquet(json_path, parquet_path, compression);
        }, py::arg("json_path"), py::arg("parquet_path"),
           py::arg("compression") = "snappy",
           "Convert JSON file to Parquet format")

        // Configuration
        .def("get_config", &cyxwiz::DataLoader::GetConfig,
             py::return_value_policy::reference_internal,
             "Get current configuration")
        .def("set_config", &cyxwiz::DataLoader::SetConfig,
             py::arg("config"),
             "Update configuration");

    // Module-level check
    m.def("duckdb_available", &cyxwiz::DataLoader::IsAvailable,
          "Check if DuckDB is available for data loading");

    // ========== Distributed Training Submodule ==========
    py::module_ distributed = m.def_submodule("distributed",
        "Distributed training support for data parallel training");

    // ReduceOp enum
    py::enum_<cyxwiz::ReduceOp>(distributed, "ReduceOp",
        "Reduction operations for collective communication")
        .value("SUM", cyxwiz::ReduceOp::SUM, "Element-wise sum")
        .value("PRODUCT", cyxwiz::ReduceOp::PRODUCT, "Element-wise product")
        .value("MIN", cyxwiz::ReduceOp::MIN, "Element-wise minimum")
        .value("MAX", cyxwiz::ReduceOp::MAX, "Element-wise maximum")
        .value("AVERAGE", cyxwiz::ReduceOp::AVERAGE, "Element-wise average")
        .export_values();

    // BackendType enum
    py::enum_<cyxwiz::BackendType>(distributed, "BackendType",
        "Backend types for distributed communication")
        .value("CPU", cyxwiz::BackendType::CPU, "TCP socket-based backend")
        .value("NCCL", cyxwiz::BackendType::NCCL, "NVIDIA NCCL backend (GPU)")
        .export_values();

    // DistributedConfig
    py::class_<cyxwiz::DistributedConfig>(distributed, "DistributedConfig",
        "Configuration for distributed training")
        .def(py::init<>())
        .def_readwrite("backend", &cyxwiz::DistributedConfig::backend,
            "Backend type (CPU or NCCL)")
        .def_readwrite("rank", &cyxwiz::DistributedConfig::rank,
            "Global rank (-1 = read from RANK env)")
        .def_readwrite("world_size", &cyxwiz::DistributedConfig::world_size,
            "Total number of processes (-1 = read from WORLD_SIZE env)")
        .def_readwrite("local_rank", &cyxwiz::DistributedConfig::local_rank,
            "Local rank for multi-GPU (-1 = read from LOCAL_RANK env)")
        .def_readwrite("master_addr", &cyxwiz::DistributedConfig::master_addr,
            "Master address (default: 127.0.0.1)")
        .def_readwrite("master_port", &cyxwiz::DistributedConfig::master_port,
            "Master port (default: 29500)")
        .def_readwrite("timeout_ms", &cyxwiz::DistributedConfig::timeout_ms,
            "Connection timeout in milliseconds")
        .def_static("from_environment", &cyxwiz::DistributedConfig::FromEnvironment,
            "Create config from environment variables (RANK, WORLD_SIZE, etc.)")
        .def("is_valid", &cyxwiz::DistributedConfig::IsValid,
            "Validate configuration")
        .def("__repr__", &cyxwiz::DistributedConfig::ToString);

    // Global distributed functions
    distributed.def("init", &cyxwiz::init_distributed,
        py::arg("config") = cyxwiz::DistributedConfig::FromEnvironment(),
        "Initialize distributed training");
    distributed.def("finalize", &cyxwiz::finalize_distributed,
        "Finalize distributed training");
    distributed.def("get_rank", &cyxwiz::get_rank,
        "Get rank of current process (-1 if not initialized)");
    distributed.def("get_world_size", &cyxwiz::get_world_size,
        "Get total number of processes (1 if not initialized)");
    distributed.def("get_local_rank", &cyxwiz::get_local_rank,
        "Get local rank (0 if not initialized)");
    distributed.def("is_distributed", &cyxwiz::is_distributed,
        "Check if distributed training is active");
    distributed.def("is_master", &cyxwiz::is_master,
        "Check if this is the master rank (rank 0)");

    // DDPConfig
    py::class_<cyxwiz::DDPConfig>(distributed, "DDPConfig",
        "Configuration for DistributedDataParallel")
        .def(py::init<>())
        .def_readwrite("broadcast_parameters", &cyxwiz::DDPConfig::broadcast_parameters,
            "Broadcast parameters from rank 0 at initialization")
        .def_readwrite("bucket_size_mb", &cyxwiz::DDPConfig::bucket_size_mb,
            "Gradient bucket size in MB (default: 25)")
        .def_readwrite("find_unused_parameters", &cyxwiz::DDPConfig::find_unused_parameters,
            "Warn about unused parameters");

    // DistributedDataParallel
    py::class_<cyxwiz::DistributedDataParallel>(distributed, "DistributedDataParallel",
        "Model wrapper for data parallel distributed training")
        .def(py::init<cyxwiz::SequentialModel*, cyxwiz::DDPConfig>(),
            py::arg("model"), py::arg("config") = cyxwiz::DDPConfig(),
            py::keep_alive<1, 2>(),  // Keep model alive
            "Wrap a model for distributed training")
        .def("forward", &cyxwiz::DistributedDataParallel::Forward,
            py::arg("input"),
            "Forward pass (delegates to wrapped model)")
        .def("backward", &cyxwiz::DistributedDataParallel::Backward,
            py::arg("grad_output"),
            "Backward pass (delegates to wrapped model)")
        .def("sync_gradients", &cyxwiz::DistributedDataParallel::SyncGradients,
            "Synchronize gradients across all ranks using AllReduce")
        .def("update_parameters", &cyxwiz::DistributedDataParallel::UpdateParameters,
            py::arg("optimizer"),
            "Sync gradients and update parameters")
        .def("broadcast_parameters", &cyxwiz::DistributedDataParallel::BroadcastParameters,
            py::arg("src_rank") = 0,
            "Broadcast parameters from source rank to all others")
        .def("get_model",
            static_cast<cyxwiz::SequentialModel* (cyxwiz::DistributedDataParallel::*)()>(&cyxwiz::DistributedDataParallel::GetModel),
            py::return_value_policy::reference,
            "Get the wrapped model")
        .def("is_master", &cyxwiz::DistributedDataParallel::IsMaster,
            "Check if this is the master rank")
        .def("get_rank", &cyxwiz::DistributedDataParallel::GetRank,
            "Get rank of this process")
        .def("get_world_size", &cyxwiz::DistributedDataParallel::GetWorldSize,
            "Get total number of processes");

    // DistributedSampler
    py::class_<cyxwiz::DistributedSampler>(distributed, "DistributedSampler",
        "Sampler that shards dataset indices across distributed workers")
        .def(py::init<size_t, bool, unsigned int, bool>(),
            py::arg("dataset_size"),
            py::arg("shuffle") = true,
            py::arg("seed") = 0,
            py::arg("drop_last") = false,
            "Create a distributed sampler")
        .def("set_epoch", &cyxwiz::DistributedSampler::SetEpoch,
            py::arg("epoch"),
            "Set epoch for deterministic shuffling")
        .def("get_epoch", &cyxwiz::DistributedSampler::GetEpoch,
            "Get current epoch")
        .def("get_indices", &cyxwiz::DistributedSampler::GetIndices,
            "Get indices for this rank's portion of the dataset")
        .def("local_size", &cyxwiz::DistributedSampler::LocalSize,
            "Get number of samples for this rank")
        .def("total_size", &cyxwiz::DistributedSampler::TotalSize,
            "Get total dataset size")
        .def("padded_size", &cyxwiz::DistributedSampler::PaddedSize,
            "Get padded size (divisible by world_size)")
        .def("get_rank", &cyxwiz::DistributedSampler::GetRank,
            "Get rank of this process")
        .def("get_world_size", &cyxwiz::DistributedSampler::GetWorldSize,
            "Get world size")
        .def("__len__", &cyxwiz::DistributedSampler::LocalSize,
            "Get number of samples for this rank");

    // DistributedBatchIterator
    py::class_<cyxwiz::DistributedBatchIterator>(distributed, "DistributedBatchIterator",
        "Iterator for batches in distributed training")
        .def(py::init<cyxwiz::DistributedSampler&, size_t>(),
            py::arg("sampler"), py::arg("batch_size"),
            py::keep_alive<1, 2>(),
            "Create a batch iterator")
        .def("reset", &cyxwiz::DistributedBatchIterator::Reset,
            py::arg("epoch"),
            "Reset iterator for new epoch")
        .def("has_next", &cyxwiz::DistributedBatchIterator::HasNext,
            "Check if there are more batches")
        .def("next", &cyxwiz::DistributedBatchIterator::Next,
            "Get next batch of indices")
        .def("num_batches", &cyxwiz::DistributedBatchIterator::NumBatches,
            "Get total number of batches")
        .def("current_batch", &cyxwiz::DistributedBatchIterator::CurrentBatch,
            "Get current batch index")
        .def("__iter__", [](cyxwiz::DistributedBatchIterator& self) -> cyxwiz::DistributedBatchIterator& {
            return self;
        })
        .def("__next__", [](cyxwiz::DistributedBatchIterator& self) {
            if (!self.HasNext()) {
                throw py::stop_iteration();
            }
            return self.Next();
        });

    // DistributedTrainingConfig
    py::class_<cyxwiz::DistributedTrainingConfig>(distributed, "DistributedTrainingConfig",
        "Configuration for distributed training")
        .def(py::init<>())
        .def_readwrite("epochs", &cyxwiz::DistributedTrainingConfig::epochs,
            "Number of training epochs")
        .def_readwrite("batch_size", &cyxwiz::DistributedTrainingConfig::batch_size,
            "Per-GPU batch size")
        .def_readwrite("shuffle", &cyxwiz::DistributedTrainingConfig::shuffle,
            "Shuffle training data each epoch")
        .def_readwrite("seed", &cyxwiz::DistributedTrainingConfig::seed,
            "Random seed for shuffling")
        .def_readwrite("save_on_master_only", &cyxwiz::DistributedTrainingConfig::save_on_master_only,
            "Only rank 0 saves checkpoints")
        .def_readwrite("checkpoint_every_n_epochs", &cyxwiz::DistributedTrainingConfig::checkpoint_every_n_epochs,
            "Save checkpoint every N epochs (0 = disabled)")
        .def_readwrite("checkpoint_dir", &cyxwiz::DistributedTrainingConfig::checkpoint_dir,
            "Directory for checkpoints")
        .def_readwrite("verbose", &cyxwiz::DistributedTrainingConfig::verbose,
            "Print training progress")
        .def_readwrite("log_every_n_batches", &cyxwiz::DistributedTrainingConfig::log_every_n_batches,
            "Print progress every N batches")
        .def_readwrite("validation_split", &cyxwiz::DistributedTrainingConfig::validation_split,
            "Validation data ratio");

    // DistributedTrainingHistory
    py::class_<cyxwiz::DistributedTrainingHistory>(distributed, "DistributedTrainingHistory",
        "Training history with metrics from distributed training")
        .def(py::init<>())
        .def_readonly("train_losses", &cyxwiz::DistributedTrainingHistory::train_losses,
            "Training loss per epoch")
        .def_readonly("train_accuracies", &cyxwiz::DistributedTrainingHistory::train_accuracies,
            "Training accuracy per epoch")
        .def_readonly("val_losses", &cyxwiz::DistributedTrainingHistory::val_losses,
            "Validation loss per epoch")
        .def_readonly("val_accuracies", &cyxwiz::DistributedTrainingHistory::val_accuracies,
            "Validation accuracy per epoch")
        .def_readonly("total_time_seconds", &cyxwiz::DistributedTrainingHistory::total_time_seconds,
            "Total training time in seconds")
        .def_readonly("samples_per_second", &cyxwiz::DistributedTrainingHistory::samples_per_second,
            "Throughput: samples processed per second")
        .def_readonly("effective_batch_size", &cyxwiz::DistributedTrainingHistory::effective_batch_size,
            "Effective batch size (batch_size * world_size)")
        .def_readonly("world_size", &cyxwiz::DistributedTrainingHistory::world_size,
            "Number of ranks used");

    // ProcessGroup (abstract base class for type recognition)
    py::class_<cyxwiz::ProcessGroup>(distributed, "ProcessGroup",
        "Abstract base class for distributed communication")
        .def("is_initialized", &cyxwiz::ProcessGroup::IsInitialized,
            "Check if process group is initialized")
        .def("get_rank", &cyxwiz::ProcessGroup::GetRank,
            "Get global rank")
        .def("get_world_size", &cyxwiz::ProcessGroup::GetWorldSize,
            "Get world size")
        .def("get_local_rank", &cyxwiz::ProcessGroup::GetLocalRank,
            "Get local rank")
        .def("barrier", &cyxwiz::ProcessGroup::Barrier,
            "Synchronization barrier")
        .def("get_backend_name", &cyxwiz::ProcessGroup::GetBackendName,
            "Get backend name (CPU or NCCL)");

    // Get default process group function
    distributed.def("get_default_process_group", &cyxwiz::GetDefaultProcessGroup,
        py::return_value_policy::reference,
        "Get the default (global) process group, or None if not initialized");

    // DistributedTrainer - using py::object for flexible type handling
    py::class_<cyxwiz::DistributedTrainer>(distributed, "DistributedTrainer",
        "High-level trainer for distributed data parallel training")
        .def(py::init([](cyxwiz::SequentialModel* model, py::object loss_obj,
                         py::object optimizer_obj, cyxwiz::ProcessGroup* pg) {
            // Cast through py::cast to properly handle inheritance
            cyxwiz::Loss* loss = loss_obj.cast<cyxwiz::Loss*>();
            cyxwiz::Optimizer* optimizer = optimizer_obj.cast<cyxwiz::Optimizer*>();
            return new cyxwiz::DistributedTrainer(model, loss, optimizer, pg);
        }),
            py::arg("model"), py::arg("loss"), py::arg("optimizer"),
            py::arg("process_group") = nullptr,
            py::keep_alive<1, 2>(),  // Keep model alive
            py::keep_alive<1, 3>(),  // Keep loss alive
            py::keep_alive<1, 4>(),  // Keep optimizer alive
            "Create a distributed trainer")
        .def("fit", py::overload_cast<const cyxwiz::Tensor&, const cyxwiz::Tensor&,
                                       const cyxwiz::DistributedTrainingConfig&>(
            &cyxwiz::DistributedTrainer::Fit),
            py::arg("X_train"), py::arg("y_train"), py::arg("config"),
            "Train the model")
        .def("fit", py::overload_cast<const cyxwiz::Tensor&, const cyxwiz::Tensor&,
                                       const cyxwiz::Tensor&, const cyxwiz::Tensor&,
                                       const cyxwiz::DistributedTrainingConfig&>(
            &cyxwiz::DistributedTrainer::Fit),
            py::arg("X_train"), py::arg("y_train"),
            py::arg("X_val"), py::arg("y_val"), py::arg("config"),
            "Train the model with validation data")
        .def("evaluate", &cyxwiz::DistributedTrainer::Evaluate,
            py::arg("X_test"), py::arg("y_test"),
            "Evaluate model on test data, returns (loss, accuracy)")
        .def("is_master", &cyxwiz::DistributedTrainer::IsMaster,
            "Check if this is the master rank")
        .def("get_rank", &cyxwiz::DistributedTrainer::GetRank,
            "Get current rank")
        .def("get_world_size", &cyxwiz::DistributedTrainer::GetWorldSize,
            "Get world size")
        .def("save_checkpoint", &cyxwiz::DistributedTrainer::SaveCheckpoint,
            py::arg("path"),
            "Save model checkpoint (only on master by default)")
        .def("load_checkpoint", &cyxwiz::DistributedTrainer::LoadCheckpoint,
            py::arg("path"),
            "Load model checkpoint (all ranks load)")
        .def("get_model", &cyxwiz::DistributedTrainer::GetModel,
            py::return_value_policy::reference,
            "Get the underlying model");

    // ====================================================================
    // Tokenizer Bindings
    // ====================================================================

    py::enum_<cyxwiz::TokenizerType>(m, "TokenizerType")
        .value("Whitespace", cyxwiz::TokenizerType::Whitespace)
        .value("Word", cyxwiz::TokenizerType::Word)
        .value("Character", cyxwiz::TokenizerType::Character)
        .export_values();

    py::class_<cyxwiz::Vocabulary>(m, "Vocabulary")
        .def(py::init<>())
        .def("build_from_documents", &cyxwiz::Vocabulary::BuildFromDocuments,
            py::arg("documents"),
            py::arg("min_freq") = 1,
            py::arg("max_vocab_size") = -1,
            py::arg("lowercase") = true,
            "Build vocabulary from a list of documents")
        .def("set_vocabulary", &cyxwiz::Vocabulary::SetVocabulary,
            py::arg("words"), "Set vocabulary from a list of words")
        .def("add_word", &cyxwiz::Vocabulary::AddWord,
            py::arg("word"), "Add a word, returns its index")
        .def("word_to_index", &cyxwiz::Vocabulary::WordToIndex,
            py::arg("word"), "Get index for a word ([UNK] if not found)")
        .def("index_to_word", &cyxwiz::Vocabulary::IndexToWord,
            py::arg("index"), "Get word for an index")
        .def("has_word", &cyxwiz::Vocabulary::HasWord,
            py::arg("word"), "Check if word is in vocabulary")
        .def("size", &cyxwiz::Vocabulary::Size, "Get vocabulary size")
        .def("save_to_file", &cyxwiz::Vocabulary::SaveToFile,
            py::arg("filepath"), "Save vocabulary to file")
        .def("load_from_file", &cyxwiz::Vocabulary::LoadFromFile,
            py::arg("filepath"), "Load vocabulary from file")
        .def_property_readonly("pad_index", &cyxwiz::Vocabulary::PadIndex)
        .def_property_readonly("unk_index", &cyxwiz::Vocabulary::UnkIndex)
        .def_property_readonly("bos_index", &cyxwiz::Vocabulary::BosIndex)
        .def_property_readonly("eos_index", &cyxwiz::Vocabulary::EosIndex);

    py::class_<cyxwiz::Tokenizer>(m, "Tokenizer")
        .def(py::init<cyxwiz::TokenizerType>(),
            py::arg("type") = cyxwiz::TokenizerType::Word,
            "Create a tokenizer")
        .def("encode", &cyxwiz::Tokenizer::Encode,
            py::arg("text"), "Encode text to token IDs")
        .def("decode", &cyxwiz::Tokenizer::Decode,
            py::arg("token_ids"), "Decode token IDs to text")
        .def("encode_batch", &cyxwiz::Tokenizer::EncodeBatch,
            py::arg("texts"), "Encode batch of texts")
        .def("decode_batch", &cyxwiz::Tokenizer::DecodeBatch,
            py::arg("batch"), "Decode batch of token IDs")
        .def("pad_batch", &cyxwiz::Tokenizer::PadBatch,
            py::arg("batch"), py::arg("max_length") = -1,
            "Pad a batch to uniform length")
        .def("train", &cyxwiz::Tokenizer::Train,
            py::arg("documents"),
            py::arg("min_freq") = 1,
            py::arg("max_vocab_size") = -1,
            "Train tokenizer on documents (builds vocabulary)")
        .def("set_vocabulary", &cyxwiz::Tokenizer::SetVocabulary,
            py::arg("vocab"), "Set the vocabulary")
        .def("get_vocabulary", static_cast<cyxwiz::Vocabulary& (cyxwiz::Tokenizer::*)()>(&cyxwiz::Tokenizer::GetVocabulary),
            py::return_value_policy::reference_internal,
            "Get the vocabulary")
        .def("set_lowercase", &cyxwiz::Tokenizer::SetLowercase,
            py::arg("value"), "Enable/disable lowercase normalization")
        .def("set_max_length", &cyxwiz::Tokenizer::SetMaxLength,
            py::arg("value"), "Set maximum token sequence length")
        .def("set_padding", &cyxwiz::Tokenizer::SetPadding,
            py::arg("value"), "Enable/disable padding")
        .def("set_truncation", &cyxwiz::Tokenizer::SetTruncation,
            py::arg("value"), "Enable/disable truncation")
        .def("set_add_bos", &cyxwiz::Tokenizer::SetAddBos,
            py::arg("value"), "Enable/disable BOS token")
        .def("set_add_eos", &cyxwiz::Tokenizer::SetAddEos,
            py::arg("value"), "Enable/disable EOS token")
        .def_property_readonly("vocab_size", [](const cyxwiz::Tokenizer& t) {
            return t.GetVocabulary().Size();
        });

    // ============================================================================
    // Reinforcement Learning
    // ============================================================================

    py::class_<cyxwiz::RLTransition>(m, "RLTransition")
        .def(py::init<>())
        .def_readwrite("state", &cyxwiz::RLTransition::state)
        .def_readwrite("action", &cyxwiz::RLTransition::action)
        .def_readwrite("reward", &cyxwiz::RLTransition::reward)
        .def_readwrite("next_state", &cyxwiz::RLTransition::next_state)
        .def_readwrite("done", &cyxwiz::RLTransition::done);

    py::class_<cyxwiz::RLBatch>(m, "RLBatch")
        .def_readonly("states", &cyxwiz::RLBatch::states)
        .def_readonly("actions", &cyxwiz::RLBatch::actions)
        .def_readonly("rewards", &cyxwiz::RLBatch::rewards)
        .def_readonly("next_states", &cyxwiz::RLBatch::next_states)
        .def_readonly("dones", &cyxwiz::RLBatch::dones)
        .def_readonly("size", &cyxwiz::RLBatch::size);

    py::class_<cyxwiz::StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("observation", &cyxwiz::StepResult::observation)
        .def_readwrite("reward", &cyxwiz::StepResult::reward)
        .def_readwrite("done", &cyxwiz::StepResult::done)
        .def_readwrite("truncated", &cyxwiz::StepResult::truncated)
        .def_readwrite("info", &cyxwiz::StepResult::info);

    py::class_<cyxwiz::EnvInfo>(m, "EnvInfo")
        .def(py::init<>())
        .def_readonly("name", &cyxwiz::EnvInfo::name)
        .def_readonly("observation_dim", &cyxwiz::EnvInfo::observation_dim)
        .def_readonly("action_dim", &cyxwiz::EnvInfo::action_dim)
        .def_readonly("discrete_actions", &cyxwiz::EnvInfo::discrete_actions)
        .def_readonly("num_actions", &cyxwiz::EnvInfo::num_actions)
        .def_readonly("action_low", &cyxwiz::EnvInfo::action_low)
        .def_readonly("action_high", &cyxwiz::EnvInfo::action_high)
        .def_readonly("valid", &cyxwiz::EnvInfo::valid)
        .def_readonly("error_message", &cyxwiz::EnvInfo::error_message);

    py::class_<cyxwiz::ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<size_t, unsigned int>(),
            py::arg("capacity") = 100000, py::arg("seed") = 42,
            "Create a replay buffer with given capacity")
        .def("push", py::overload_cast<const cyxwiz::RLTransition&>(&cyxwiz::ReplayBuffer::Push),
            py::arg("transition"), "Add a transition")
        .def("push", py::overload_cast<const std::vector<float>&, const std::vector<float>&,
                                        float, const std::vector<float>&, bool>(&cyxwiz::ReplayBuffer::Push),
            py::arg("state"), py::arg("action"), py::arg("reward"),
            py::arg("next_state"), py::arg("done"),
            "Add a transition from components")
        .def("sample", &cyxwiz::ReplayBuffer::Sample,
            py::arg("batch_size"), "Sample a random batch")
        .def("size", &cyxwiz::ReplayBuffer::Size, "Current buffer size")
        .def("capacity", &cyxwiz::ReplayBuffer::Capacity, "Maximum capacity")
        .def("can_sample", &cyxwiz::ReplayBuffer::CanSample,
            py::arg("batch_size"), "Check if enough samples available")
        .def("clear", &cyxwiz::ReplayBuffer::Clear, "Clear all transitions")
        .def("__len__", &cyxwiz::ReplayBuffer::Size);

    py::class_<cyxwiz::EpsilonSchedule>(m, "EpsilonSchedule")
        .def(py::init<float, float, int>(),
            py::arg("start") = 1.0f, py::arg("end") = 0.01f,
            py::arg("decay_steps") = 10000,
            "Create linear epsilon decay schedule")
        .def("step", &cyxwiz::EpsilonSchedule::Step, "Decay epsilon by one step")
        .def("reset", &cyxwiz::EpsilonSchedule::Reset, "Reset to initial epsilon")
        .def_property_readonly("epsilon", &cyxwiz::EpsilonSchedule::GetEpsilon)
        .def_property_readonly("current_step", &cyxwiz::EpsilonSchedule::GetStep);


    // =========================================================================
    // RL Training Dashboard Bridge
    // =========================================================================
    // These functions allow Python RL training scripts (SB3) to stream metrics
    // back to the C++ TrainingDashboardPanel and check pause/stop state.

    static std::function<void(const std::string&, float)> s_rl_metric_callback;
    static std::atomic<bool> s_rl_stop_requested{false};
    static std::atomic<bool> s_rl_paused{false};

    m.def("rl_set_metric_callback", [](py::object callback) {
        if (callback.is_none()) {
            s_rl_metric_callback = nullptr;
        } else {
            s_rl_metric_callback = [callback](const std::string& name, float value) {
                py::gil_scoped_acquire acquire;
                callback(name, value);
            };
        }
    }, py::arg("callback"),
       "Set a callback function(name: str, value: float) for RL metric updates");

    m.def("rl_update_metric", [](const std::string& name, float value) {
        if (s_rl_metric_callback) {
            py::gil_scoped_release release;
            s_rl_metric_callback(name, value);
        }
    }, py::arg("name"), py::arg("value"),
       "Update an RL metric on the dashboard (thread-safe)");

    m.def("rl_should_stop", []() -> bool {
        return s_rl_stop_requested.load();
    }, "Check if stop was requested for RL training");

    m.def("rl_is_paused", []() -> bool {
        return s_rl_paused.load();
    }, "Check if RL training is paused");

    m.def("rl_set_stop", [](bool val) {
        s_rl_stop_requested.store(val);
    }, py::arg("value"), "Set RL training stop flag");

    m.def("rl_set_paused", [](bool val) {
        s_rl_paused.store(val);
    }, py::arg("value"), "Set RL training pause flag");

}
