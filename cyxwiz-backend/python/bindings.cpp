#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cyxwiz/cyxwiz.h"

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
        .export_values();

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
}
