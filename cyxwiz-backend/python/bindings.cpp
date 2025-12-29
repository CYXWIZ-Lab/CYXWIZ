#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "cyxwiz/cyxwiz.h"

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

    // Tensor
    py::class_<cyxwiz::Tensor>(m, "Tensor",
        "Multi-dimensional array with GPU/CPU acceleration support")
        .def(py::init<>(), "Create an empty tensor")
        .def(py::init<const std::vector<size_t>&, cyxwiz::DataType>(),
             py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32,
             "Create a tensor with given shape and data type")

        // Shape and metadata
        .def("shape", &cyxwiz::Tensor::Shape,
             "Get the shape of the tensor")
        .def("num_elements", &cyxwiz::Tensor::NumElements,
             "Get total number of elements")
        .def("num_bytes", &cyxwiz::Tensor::NumBytes,
             "Get total size in bytes")
        .def("get_data_type", &cyxwiz::Tensor::GetDataType,
             "Get the data type")
        .def("num_dimensions", &cyxwiz::Tensor::NumDimensions,
             "Get number of dimensions")

        // Arithmetic operators
        .def("__add__", &cyxwiz::Tensor::operator+,
             py::arg("other"),
             "Element-wise addition")
        .def("__sub__", &cyxwiz::Tensor::operator-,
             py::arg("other"),
             "Element-wise subtraction")
        .def("__mul__", &cyxwiz::Tensor::operator*,
             py::arg("other"),
             "Element-wise multiplication")
        .def("__truediv__", &cyxwiz::Tensor::operator/,
             py::arg("other"),
             "Element-wise division")

        // Device management
        .def("get_device", &cyxwiz::Tensor::GetDevice,
             py::return_value_policy::reference,
             "Get the device this tensor is on")

        // String representation
        .def("__repr__", [](const cyxwiz::Tensor &t) {
            std::string shape_str = "[";
            const auto& shape = t.Shape();
            for (size_t i = 0; i < shape.size(); i++) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(shape[i]);
            }
            shape_str += "]";

            std::string dtype_str;
            switch(t.GetDataType()) {
                case cyxwiz::DataType::Float32: dtype_str = "float32"; break;
                case cyxwiz::DataType::Float64: dtype_str = "float64"; break;
                case cyxwiz::DataType::Int32:   dtype_str = "int32"; break;
                case cyxwiz::DataType::Int64:   dtype_str = "int64"; break;
                case cyxwiz::DataType::UInt8:   dtype_str = "uint8"; break;
            }

            return "<Tensor shape=" + shape_str + " dtype=" + dtype_str + ">";
        })

        // Static factory methods
        .def_static("zeros", &cyxwiz::Tensor::Zeros,
                   py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32,
                   "Create a tensor filled with zeros")
        .def_static("ones", &cyxwiz::Tensor::Ones,
                   py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32,
                   "Create a tensor filled with ones")
        .def_static("random", &cyxwiz::Tensor::Random,
                   py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32,
                   "Create a tensor with random values [0, 1)")

        // NumPy conversion
        .def_static("from_numpy", [](py::array arr) {
            // Get shape
            std::vector<size_t> shape;
            for (py::ssize_t i = 0; i < arr.ndim(); i++) {
                shape.push_back(arr.shape(i));
            }

            // Get data type
            cyxwiz::DataType dtype = numpy_dtype_to_cyxwiz(arr.dtype());

            // Create tensor and copy data
            cyxwiz::Tensor tensor(shape, dtype);

            // Ensure NumPy array is contiguous
            py::array arr_c = py::array::ensure(arr, py::array::c_style);

            // Copy data from NumPy to Tensor
            std::memcpy(tensor.Data(), arr_c.data(), tensor.NumBytes());

            return tensor;
        }, py::arg("array"), "Create a Tensor from a NumPy array")

        .def("to_numpy", [](cyxwiz::Tensor& self) {
            // Get shape
            const auto& shape = self.Shape();
            std::vector<py::ssize_t> np_shape(shape.begin(), shape.end());

            // Determine NumPy dtype
            py::dtype np_dtype;
            switch (self.GetDataType()) {
                case cyxwiz::DataType::Float32:
                    np_dtype = py::dtype::of<float>();
                    break;
                case cyxwiz::DataType::Float64:
                    np_dtype = py::dtype::of<double>();
                    break;
                case cyxwiz::DataType::Int32:
                    np_dtype = py::dtype::of<int32_t>();
                    break;
                case cyxwiz::DataType::Int64:
                    np_dtype = py::dtype::of<int64_t>();
                    break;
                case cyxwiz::DataType::UInt8:
                    np_dtype = py::dtype::of<uint8_t>();
                    break;
                default:
                    throw std::runtime_error("Unsupported data type for NumPy conversion");
            }

            // Create NumPy array and copy data
            py::array result(np_dtype, np_shape);
            std::memcpy(result.mutable_data(), self.Data(), self.NumBytes());

            return result;
        }, "Convert Tensor to NumPy array (Note: data must be on CPU)");

    // OptimizerType enum
    py::enum_<cyxwiz::OptimizerType>(m, "OptimizerType")
        .value("SGD", cyxwiz::OptimizerType::SGD)
        .value("Adam", cyxwiz::OptimizerType::Adam)
        .value("AdamW", cyxwiz::OptimizerType::AdamW)
        .value("RMSprop", cyxwiz::OptimizerType::RMSprop)
        .value("AdaGrad", cyxwiz::OptimizerType::AdaGrad)
        .value("NAdam", cyxwiz::OptimizerType::NAdam)
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

    // Activation base class
    py::class_<cyxwiz::Activation>(m, "Activation")
        .def("forward", &cyxwiz::Activation::Forward,
             py::arg("input"),
             "Forward pass through activation")
        .def("backward", &cyxwiz::Activation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward pass (compute gradients)");

    // ReLU activation
    py::class_<cyxwiz::ReLU, cyxwiz::Activation>(m, "ReLU")
        .def(py::init<>(),
             "Create ReLU activation: f(x) = max(0, x)")
        .def("forward", &cyxwiz::ReLU::Forward,
             py::arg("input"),
             "Forward: f(x) = max(0, x)")
        .def("backward", &cyxwiz::ReLU::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: f'(x) = 1 if x > 0 else 0");

    // Sigmoid activation
    py::class_<cyxwiz::Sigmoid, cyxwiz::Activation>(m, "Sigmoid")
        .def(py::init<>(),
             "Create Sigmoid activation: f(x) = 1 / (1 + exp(-x))")
        .def("forward", &cyxwiz::Sigmoid::Forward,
             py::arg("input"),
             "Forward: f(x) = 1 / (1 + exp(-x))")
        .def("backward", &cyxwiz::Sigmoid::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: f'(x) = f(x) * (1 - f(x))");

    // Tanh activation
    py::class_<cyxwiz::Tanh, cyxwiz::Activation>(m, "Tanh")
        .def(py::init<>(),
             "Create Tanh activation: f(x) = tanh(x)")
        .def("forward", &cyxwiz::Tanh::Forward,
             py::arg("input"),
             "Forward: f(x) = tanh(x)")
        .def("backward", &cyxwiz::Tanh::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: f'(x) = 1 - tanh(x)^2");

    // GELU activation
    py::class_<cyxwiz::GELUActivation, cyxwiz::Activation>(m, "GELU")
        .def(py::init<>(),
             "Create GELU activation: Gaussian Error Linear Unit")
        .def("forward", &cyxwiz::GELUActivation::Forward,
             py::arg("input"),
             "Forward: GELU(x)")
        .def("backward", &cyxwiz::GELUActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients");

    // LeakyReLU activation
    py::class_<cyxwiz::LeakyReLUActivation, cyxwiz::Activation>(m, "LeakyReLU")
        .def(py::init<float>(),
             py::arg("negative_slope") = 0.01f,
             "Create LeakyReLU activation: f(x) = x if x > 0 else negative_slope * x")
        .def("forward", &cyxwiz::LeakyReLUActivation::Forward,
             py::arg("input"),
             "Forward: LeakyReLU(x)")
        .def("backward", &cyxwiz::LeakyReLUActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients")
        .def_property_readonly("alpha", &cyxwiz::LeakyReLUActivation::GetAlpha,
                              "Negative slope value");

    // ELU activation
    py::class_<cyxwiz::ELUActivation, cyxwiz::Activation>(m, "ELU")
        .def(py::init<float>(),
             py::arg("alpha") = 1.0f,
             "Create ELU activation: Exponential Linear Unit")
        .def("forward", &cyxwiz::ELUActivation::Forward,
             py::arg("input"),
             "Forward: ELU(x)")
        .def("backward", &cyxwiz::ELUActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients")
        .def_property_readonly("alpha", &cyxwiz::ELUActivation::GetAlpha,
                              "Alpha value");

    // Swish activation
    py::class_<cyxwiz::SwishActivation, cyxwiz::Activation>(m, "Swish")
        .def(py::init<>(),
             "Create Swish activation: f(x) = x * sigmoid(x)")
        .def("forward", &cyxwiz::SwishActivation::Forward,
             py::arg("input"),
             "Forward: Swish(x)")
        .def("backward", &cyxwiz::SwishActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients");

    // SiLU alias for Swish (PyTorch naming)
    m.attr("SiLU") = m.attr("Swish");

    // Mish activation
    py::class_<cyxwiz::MishActivation, cyxwiz::Activation>(m, "Mish")
        .def(py::init<>(),
             "Create Mish activation: f(x) = x * tanh(softplus(x))")
        .def("forward", &cyxwiz::MishActivation::Forward,
             py::arg("input"),
             "Forward: Mish(x)")
        .def("backward", &cyxwiz::MishActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients");

    // Hardswish activation
    py::class_<cyxwiz::HardswishActivation, cyxwiz::Activation>(m, "Hardswish")
        .def(py::init<>(),
             "Create Hardswish activation: efficient approximation of Swish")
        .def("forward", &cyxwiz::HardswishActivation::Forward,
             py::arg("input"),
             "Forward: Hardswish(x)")
        .def("backward", &cyxwiz::HardswishActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients");

    // Softmax activation
    py::class_<cyxwiz::SoftmaxActivation, cyxwiz::Activation>(m, "Softmax")
        .def(py::init<int>(),
             py::arg("dim") = -1,
             "Create Softmax activation: normalizes to probability distribution")
        .def("forward", &cyxwiz::SoftmaxActivation::Forward,
             py::arg("input"),
             "Forward: Softmax(x)")
        .def("backward", &cyxwiz::SoftmaxActivation::Backward,
             py::arg("grad_output"),
             py::arg("input"),
             "Backward: compute gradients");

    // MSE Loss (concrete implementation)
    py::class_<cyxwiz::MSELoss>(m, "MSELoss")
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
    py::class_<cyxwiz::CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>(),
             "Create CrossEntropy Loss with softmax for classification")
        .def("forward", &cyxwiz::CrossEntropyLoss::Forward,
             py::arg("predictions"),
             py::arg("targets"),
             "Forward: compute cross entropy loss (predictions are logits)")
        .def("backward", &cyxwiz::CrossEntropyLoss::Backward,
             py::arg("predictions"),
             py::arg("targets"),
             "Backward: gradient w.r.t logits");

    // ============================================================================
    // DIRECT OPTIMIZER CLASSES
    // ============================================================================

    // SGD Optimizer
    py::class_<cyxwiz::SGDOptimizer, cyxwiz::Optimizer>(m, "SGD")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("momentum") = 0.0,
             "SGD optimizer with optional momentum")
        .def("step", &cyxwiz::SGDOptimizer::Step,
             py::arg("parameters"),
             py::arg("gradients"),
             "Update parameters using gradients")
        .def("zero_grad", &cyxwiz::SGDOptimizer::ZeroGrad,
             "Clear optimizer state");

    // Adam Optimizer
    py::class_<cyxwiz::AdamOptimizer, cyxwiz::Optimizer>(m, "Adam")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             "Adam optimizer")
        .def("step", &cyxwiz::AdamOptimizer::Step,
             py::arg("parameters"),
             py::arg("gradients"),
             "Update parameters using gradients")
        .def("zero_grad", &cyxwiz::AdamOptimizer::ZeroGrad,
             "Clear optimizer state");

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
             py::arg("parameters"),
             py::arg("gradients"),
             "Update parameters using gradients")
        .def("zero_grad", &cyxwiz::AdamWOptimizer::ZeroGrad,
             "Clear optimizer state");

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

    linalg.def("diag", [](const std::vector<double>& d) {
        auto result = cyxwiz::LinearAlgebra::Diagonal(d);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Create diagonal matrix from vector", py::arg("d"));

    // Decompositions
    linalg.def("svd", [](const std::vector<std::vector<double>>& A, bool full_matrices) {
        auto result = cyxwiz::LinearAlgebra::SVD(A, full_matrices);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.U, result.S, result.Vt);
    }, "Singular Value Decomposition: U, S, Vt = svd(A)", py::arg("A"), py::arg("full_matrices") = false);

    linalg.def("eig", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Eigen(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.eigenvalues, result.eigenvectors);
    }, "Eigenvalue decomposition: eigenvalues, eigenvectors = eig(A)");

    linalg.def("qr", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::QR(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.Q, result.R);
    }, "QR decomposition: Q, R = qr(A)");

    linalg.def("chol", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Cholesky(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.L;
    }, "Cholesky decomposition: L = chol(A) where A = L @ L.T");

    linalg.def("lu", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::LU(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return py::make_tuple(result.L, result.U, result.P);
    }, "LU decomposition: L, U, P = lu(A)");

    // Matrix properties
    linalg.def("det", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Determinant(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.value;
    }, "Compute determinant");

    linalg.def("rank", [](const std::vector<std::vector<double>>& A, double tol) {
        auto result = cyxwiz::LinearAlgebra::Rank(A, tol);
        return static_cast<int>(result.value);
    }, "Compute matrix rank", py::arg("A"), py::arg("tol") = 1e-10);

    linalg.def("trace", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Trace(A);
        return result.value;
    }, "Compute trace (sum of diagonal)");

    linalg.def("norm", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::FrobeniusNorm(A);
        return result.value;
    }, "Frobenius norm");

    linalg.def("cond", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::ConditionNumber(A);
        return result.value;
    }, "Condition number");

    // Matrix operations
    linalg.def("inv", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Inverse(A);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Matrix inverse");

    linalg.def("transpose", [](const std::vector<std::vector<double>>& A) {
        auto result = cyxwiz::LinearAlgebra::Transpose(A);
        return result.matrix;
    }, "Matrix transpose");

    linalg.def("solve", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
        auto result = cyxwiz::LinearAlgebra::Solve(A, b);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Solve Ax = b");

    linalg.def("lstsq", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
        auto result = cyxwiz::LinearAlgebra::LeastSquares(A, b);
        if (!result.success) throw std::runtime_error(result.error_message);
        return result.matrix;
    }, "Least squares solution");

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

    // ============================================================================
    // Activation Functions
    // ============================================================================

    // ActivationType enum
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
        .value("PReLU", cyxwiz::ActivationType::PReLU)
        .export_values();

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


}
