#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cyxwiz/cyxwiz.h"

namespace py = pybind11;

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
    py::class_<cyxwiz::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, cyxwiz::DataType>(),
             py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32)
        .def("shape", &cyxwiz::Tensor::Shape)
        .def("num_elements", &cyxwiz::Tensor::NumElements)
        .def("num_bytes", &cyxwiz::Tensor::NumBytes)
        .def("get_data_type", &cyxwiz::Tensor::GetDataType)
        .def("num_dimensions", &cyxwiz::Tensor::NumDimensions)
        .def_static("zeros", &cyxwiz::Tensor::Zeros,
                   py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32)
        .def_static("ones", &cyxwiz::Tensor::Ones,
                   py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32)
        .def_static("random", &cyxwiz::Tensor::Random,
                   py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32);

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
}
