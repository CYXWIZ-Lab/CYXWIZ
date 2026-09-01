#include "bindings_tensor.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cyxwiz/device.h"
#include "cyxwiz/tensor.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

cyxwiz::DataType numpy_dtype_to_cyxwiz(const py::dtype& dt) {
    if (dt.is(py::dtype::of<float>())) {
        return cyxwiz::DataType::Float32;
    }
    if (dt.is(py::dtype::of<double>())) {
        return cyxwiz::DataType::Float64;
    }
    if (dt.is(py::dtype::of<int32_t>())) {
        return cyxwiz::DataType::Int32;
    }
    if (dt.is(py::dtype::of<int64_t>())) {
        return cyxwiz::DataType::Int64;
    }
    if (dt.is(py::dtype::of<uint8_t>())) {
        return cyxwiz::DataType::UInt8;
    }
    throw std::runtime_error("Unsupported NumPy dtype");
}

} // namespace

void BindTensor(py::module_& m) {
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

    // Shape operations
    .def("reshape", &cyxwiz::Tensor::Reshape,
         py::arg("shape"),
         "Return a tensor with a new shape")
    .def("view", &cyxwiz::Tensor::View,
         py::arg("shape"),
         "Alias for reshape semantics")
    .def("squeeze",
         static_cast<cyxwiz::Tensor (cyxwiz::Tensor::*)() const>(
             &cyxwiz::Tensor::Squeeze),
         "Remove all singleton dimensions")
    .def("squeeze",
         static_cast<cyxwiz::Tensor (cyxwiz::Tensor::*)(int) const>(
             &cyxwiz::Tensor::Squeeze),
         py::arg("dim"),
         "Remove a singleton dimension when its size is one")
    .def("unsqueeze", &cyxwiz::Tensor::Unsqueeze,
         py::arg("dim"),
         "Insert a singleton dimension")
    .def("flatten",
         static_cast<cyxwiz::Tensor (cyxwiz::Tensor::*)() const>(&cyxwiz::Tensor::Flatten),
         "Flatten all dimensions")
    .def("flatten",
         static_cast<cyxwiz::Tensor (cyxwiz::Tensor::*)(int, int) const>(&cyxwiz::Tensor::Flatten),
         py::arg("start_dim"), py::arg("end_dim") = -1,
         "Flatten a selected dimension range")
    .def("transpose",
         static_cast<cyxwiz::Tensor (cyxwiz::Tensor::*)() const>(&cyxwiz::Tensor::Transpose),
         "Transpose a 2D tensor")
    .def("transpose",
         static_cast<cyxwiz::Tensor (cyxwiz::Tensor::*)(int, int) const>(&cyxwiz::Tensor::Transpose),
         py::arg("dim0"), py::arg("dim1"),
         "Swap two dimensions")
    .def("permute", &cyxwiz::Tensor::Permute,
         py::arg("dims"),
         "Reorder dimensions")

    // Indexing and slicing
    .def("at", [](const cyxwiz::Tensor& self, py::args args) {
        switch (args.size()) {
            case 1: return self.At(args[0].cast<size_t>());
            case 2: return self.At(args[0].cast<size_t>(), args[1].cast<size_t>());
            case 3: return self.At(args[0].cast<size_t>(), args[1].cast<size_t>(), args[2].cast<size_t>());
            case 4: return self.At(args[0].cast<size_t>(), args[1].cast<size_t>(), args[2].cast<size_t>(), args[3].cast<size_t>());
            default: throw std::runtime_error("Tensor.at expects 1 to 4 indices");
        }
    }, "Read an element as float")
    .def("set", [](cyxwiz::Tensor& self, py::args args) {
        if (args.size() < 2 || args.size() > 5) {
            throw std::runtime_error("Tensor.set expects 1 to 4 indices plus a value");
        }
        const float value = args[args.size() - 1].cast<float>();
        switch (args.size() - 1) {
            case 1: self.Set(args[0].cast<size_t>(), value); break;
            case 2: self.Set(args[0].cast<size_t>(), args[1].cast<size_t>(), value); break;
            case 3: self.Set(args[0].cast<size_t>(), args[1].cast<size_t>(), args[2].cast<size_t>(), value); break;
            case 4: self.Set(args[0].cast<size_t>(), args[1].cast<size_t>(), args[2].cast<size_t>(), args[3].cast<size_t>(), value); break;
            default: throw std::runtime_error("Tensor.set expects 1 to 4 indices plus a value");
        }
    }, "Set an element from a float value")
    .def("slice", &cyxwiz::Tensor::Slice,
         py::arg("dim"), py::arg("start"), py::arg("end") = -1, py::arg("step") = 1,
         "Slice a tensor along one dimension")
    .def("index_select", &cyxwiz::Tensor::IndexSelect,
         py::arg("dim"), py::arg("indices"),
         "Gather indices along one dimension")

    // Concatenation and splitting
    .def_static("cat", &cyxwiz::Tensor::Cat,
                py::arg("tensors"), py::arg("dim") = 0,
                "Concatenate tensors along a dimension")
    .def_static("stack", &cyxwiz::Tensor::Stack,
                py::arg("tensors"), py::arg("dim") = 0,
                "Stack tensors along a new dimension")
    .def("split",
         static_cast<std::vector<cyxwiz::Tensor> (cyxwiz::Tensor::*)(int, int) const>(&cyxwiz::Tensor::Split),
         py::arg("split_size"), py::arg("dim") = 0,
         "Split tensor by fixed size")
    .def("split",
         static_cast<std::vector<cyxwiz::Tensor> (cyxwiz::Tensor::*)(const std::vector<int>&, int) const>(&cyxwiz::Tensor::Split),
         py::arg("sizes"), py::arg("dim") = 0,
         "Split tensor by explicit sizes")
    .def("chunk", &cyxwiz::Tensor::Chunk,
         py::arg("chunks"), py::arg("dim") = 0,
         "Split tensor into chunks")

    // Reductions
    .def("sum", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Sum() : self.Sum(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values")
    .def("mean", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Mean() : self.Mean(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values by mean")
    .def("max", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Max() : self.Max(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values by maximum")
    .def("min", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Min() : self.Min(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values by minimum")
    .def("prod", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Prod() : self.Prod(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values by product")
    .def("var", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Var() : self.Var(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values by population variance")
    .def("std", [](const cyxwiz::Tensor& self, py::object dim, bool keepdim) {
        return dim.is_none() ? self.Std() : self.Std(dim.cast<int>(), keepdim);
    }, py::arg("dim") = py::none(), py::arg("keepdim") = false,
       "Reduce tensor values by population standard deviation")

    // Arithmetic operators
    .def("__add__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self + other.cast<const cyxwiz::Tensor&>()
            : self + other.cast<float>();
    }, py::arg("other"), "Element-wise addition")
    .def("__radd__", [](const cyxwiz::Tensor& self, float scalar) {
        return self + scalar;
    }, py::arg("scalar"), "Scalar addition")
    .def("__sub__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self - other.cast<const cyxwiz::Tensor&>()
            : self - other.cast<float>();
    }, py::arg("other"), "Element-wise subtraction")
    .def("__rsub__", [](const cyxwiz::Tensor& self, float scalar) {
        return (-self) + scalar;
    }, py::arg("scalar"), "Scalar subtraction")
    .def("__mul__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self * other.cast<const cyxwiz::Tensor&>()
            : self * other.cast<float>();
    }, py::arg("other"), "Element-wise multiplication")
    .def("__rmul__", [](const cyxwiz::Tensor& self, float scalar) {
        return self * scalar;
    }, py::arg("scalar"), "Scalar multiplication")
    .def("__truediv__", [](const cyxwiz::Tensor& self, py::object other) {
        if (py::isinstance<cyxwiz::Tensor>(other)) {
            return self / other.cast<const cyxwiz::Tensor&>();
        }
        return self / other.cast<float>();
    }, py::arg("other"), "Element-wise division")
    .def("__neg__", [](const cyxwiz::Tensor& self) {
        return -self;
    },
         "Negate tensor values")
    .def("pow", [](const cyxwiz::Tensor& self, py::object exponent) {
        return py::isinstance<cyxwiz::Tensor>(exponent)
            ? self.Pow(exponent.cast<const cyxwiz::Tensor&>())
            : self.Pow(exponent.cast<float>());
    }, py::arg("exponent"), "Raise tensor values to a power")
    .def("__pow__", [](const cyxwiz::Tensor& self, py::object exponent) {
        return py::isinstance<cyxwiz::Tensor>(exponent)
            ? self.Pow(exponent.cast<const cyxwiz::Tensor&>())
            : self.Pow(exponent.cast<float>());
    }, py::arg("exponent"), "Raise tensor values to a power")
    .def("sqrt", &cyxwiz::Tensor::Sqrt, "Element-wise square root")
    .def("exp", &cyxwiz::Tensor::Exp, "Element-wise exponential")
    .def("log", &cyxwiz::Tensor::Log, "Element-wise natural log")
    .def("abs", &cyxwiz::Tensor::Abs, "Element-wise absolute value")
    .def("sign", &cyxwiz::Tensor::Sign, "Element-wise sign")
    .def("clip", &cyxwiz::Tensor::Clip,
         py::arg("min_val"), py::arg("max_val"),
         "Clip values to a range")

    // Linear algebra
    .def("dot", &cyxwiz::Tensor::Dot,
         py::arg("other"),
         "1D dot product")
    .def("batch_matmul", &cyxwiz::Tensor::BatchMatMul,
         py::arg("other"),
         "3D batched matrix multiplication")

    // Comparisons and logical masks
    .def("__gt__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self > other.cast<const cyxwiz::Tensor&>()
            : self > other.cast<float>();
    }, py::arg("other"), "Greater-than comparison")
    .def("__ge__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self >= other.cast<const cyxwiz::Tensor&>()
            : self >= other.cast<float>();
    }, py::arg("other"), "Greater-or-equal comparison")
    .def("__lt__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self < other.cast<const cyxwiz::Tensor&>()
            : self < other.cast<float>();
    }, py::arg("other"), "Less-than comparison")
    .def("__le__", [](const cyxwiz::Tensor& self, py::object other) {
        return py::isinstance<cyxwiz::Tensor>(other)
            ? self <= other.cast<const cyxwiz::Tensor&>()
            : self <= other.cast<float>();
    }, py::arg("other"), "Less-or-equal comparison")
    .def("__eq__", [](const cyxwiz::Tensor& self, py::object other) {
        if (!py::isinstance<cyxwiz::Tensor>(other) && !py::isinstance<py::float_>(other) && !py::isinstance<py::int_>(other)) {
            return py::cast(false);
        }
        cyxwiz::Tensor result = py::isinstance<cyxwiz::Tensor>(other)
            ? self == other.cast<const cyxwiz::Tensor&>()
            : self == other.cast<float>();
        return py::cast(result);
    }, py::arg("other"), "Equality comparison")
    .def("__ne__", [](const cyxwiz::Tensor& self, py::object other) {
        if (!py::isinstance<cyxwiz::Tensor>(other) && !py::isinstance<py::float_>(other) && !py::isinstance<py::int_>(other)) {
            return py::cast(true);
        }
        cyxwiz::Tensor result = py::isinstance<cyxwiz::Tensor>(other)
            ? self != other.cast<const cyxwiz::Tensor&>()
            : self != other.cast<float>();
        return py::cast(result);
    }, py::arg("other"), "Inequality comparison")
    .def("__and__", &cyxwiz::Tensor::operator&&,
         py::arg("other"),
         "Logical and")
    .def("__or__", &cyxwiz::Tensor::operator||,
         py::arg("other"),
         "Logical or")
    .def("__invert__", &cyxwiz::Tensor::operator!,
         "Logical not")
    .def("logical_not", &cyxwiz::Tensor::operator!,
         "Logical not")

    // Broadcasting
    .def_static("is_broadcastable", &cyxwiz::Tensor::IsBroadcastable,
                py::arg("shape1"), py::arg("shape2"),
                "Check whether shapes can broadcast")
    .def_static("broadcast_shape", &cyxwiz::Tensor::BroadcastShape,
                py::arg("shape1"), py::arg("shape2"),
                "Compute broadcasted shape")
    .def("broadcast_to", &cyxwiz::Tensor::BroadcastTo,
         py::arg("target_shape"),
         "Broadcast tensor to a target shape")
    .def("expand", &cyxwiz::Tensor::Expand,
         py::arg("target_shape"),
         "Materialize tensor expanded to a target shape")

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
    .def_static("range_n", &cyxwiz::Tensor::RangeN,
               py::arg("shape"), py::arg("dtype") = cyxwiz::DataType::Float32,
               "Create a tensor filled with row-major sequential values")

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

}
