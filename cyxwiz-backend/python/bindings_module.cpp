#include "bindings_module.h"

#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "cyxwiz/gradient_check.h"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
#include "pycyxwiz_nodes_source.inc"
}  // namespace

namespace pycyxwiz_bindings {

namespace {

using cyxwiz::Module;
using cyxwiz::Tensor;
using TensorMap = std::map<std::string, Tensor>;

// The trampoline (Part C3): a C++ Module whose virtual methods call the
// Python subclass. Written by hand rather than with PYBIND11_OVERRIDE so a
// Python error is turned into a std::runtime_error, with the traceback as
// text, while the interpreter lock is still held. The lock is taken in every
// call: the Engine's trainer calls modules from threads that do not hold it.
class PythonModule : public Module {
public:
    Tensor Forward(const Tensor& input) override {
        return Call("forward", [&]() {
            py::function forward = Override("forward");
            if (!forward) throw std::runtime_error(ClassName() + " does not define forward(self, x)");
            return ToTensor(forward(input), "forward");
        });
    }

    Tensor Backward(const Tensor& grad_output) override {
        return Call("backward", [&]() {
            py::function backward = Override("backward");
            if (!backward) {
                throw std::runtime_error(ClassName() + " does not define backward(self, grad_output)");
            }
            return ToTensor(backward(grad_output), "backward");
        });
    }

    TensorMap GetParameters() override {
        return Call("get_parameters", [&]() {
            if (py::function f = Override("get_parameters")) return ToTensorMap(f(), "get_parameters");
            return parameters_;
        });
    }

    void SetParameters(const TensorMap& params) override {
        Call("set_parameters", [&]() {
            if (py::function f = Override("set_parameters")) {
                f(params);
                return 0;
            }
            for (const auto& [name, value] : params) {
                if (parameters_.count(name)) parameters_[name] = value;
            }
            return 0;
        });
    }

    TensorMap GetGradients() override {
        return Call("get_gradients", [&]() {
            if (py::function f = Override("get_gradients")) return ToTensorMap(f(), "get_gradients");
            return gradients_;
        });
    }

    bool HasParameters() const override {
        return Call("has_parameters", [&]() {
            if (py::function f = Override("has_parameters")) return f().cast<bool>();
            return !parameters_.empty();
        });
    }

    std::string GetName() const override {
        return Call("get_name", [&]() {
            if (py::function f = Override("get_name")) return f().cast<std::string>();
            return ClassName();
        });
    }

    void SetTraining(bool training) override {
        is_training_ = training;
        Call("set_training", [&]() {
            if (py::function f = Override("set_training")) f(training);
            return 0;
        });
    }

    // Helpers for Python subclasses (add_parameter, parameter, set_gradient).
    void AddParameter(const std::string& name, const Tensor& value) {
        if (name.empty()) throw std::invalid_argument("a parameter needs a name");
        if (parameters_.count(name)) throw std::invalid_argument("parameter '" + name + "' already exists");
        parameters_[name] = value;
        gradients_[name] = Tensor::Zeros(value.Shape(), value.GetDataType());
    }

    const Tensor& Parameter(const std::string& name) const {
        const auto it = parameters_.find(name);
        if (it == parameters_.end()) throw py::key_error("no parameter '" + name + "'; add it with add_parameter");
        return it->second;
    }

    void SetGradient(const std::string& name, const Tensor& gradient) {
        const auto it = parameters_.find(name);
        if (it == parameters_.end()) throw py::key_error("no parameter '" + name + "'; add it with add_parameter");
        if (gradient.Shape() != it->second.Shape()) {
            throw std::invalid_argument("the gradient of '" + name + "' must have the parameter's shape");
        }
        gradients_[name] = gradient;
    }

    const Tensor& Gradient(const std::string& name) const {
        const auto it = gradients_.find(name);
        if (it == gradients_.end()) throw py::key_error("no parameter '" + name + "'");
        return it->second;
    }

    std::vector<std::string> ParameterNames() const {
        std::vector<std::string> names;
        for (const auto& [name, value] : parameters_) names.push_back(name);
        return names;
    }

private:
    // The Python method overriding `name`, or an empty function when the
    // subclass does not define it. Needs the lock.
    py::function Override(const char* name) const {
        return py::get_override(static_cast<const Module*>(this), name);
    }

    // The Python class name, from the bound forward method. Needs the lock.
    std::string ClassName() const {
        if (py::function forward = Override("forward")) {
            if (py::hasattr(forward, "__self__")) {
                return py::str(forward.attr("__self__").attr("__class__").attr("__name__"));
            }
        }
        return "Module";
    }

    template <typename F>
    auto Call(const char* what, F&& body) const -> decltype(body()) {
        py::gil_scoped_acquire gil;
        try {
            return body();
        } catch (py::error_already_set& error) {
            throw std::runtime_error(ClassName() + "." + what + " raised " + error.what());
        }
    }

    Tensor ToTensor(const py::object& value, const char* what) const {
        if (!py::isinstance<Tensor>(value)) {
            throw std::runtime_error(ClassName() + "." + what + " must return a Tensor, not " +
                                     std::string(py::str(py::type::handle_of(value).attr("__name__"))));
        }
        return value.cast<Tensor>();
    }

    TensorMap ToTensorMap(const py::object& value, const char* what) const {
        try {
            return value.cast<TensorMap>();
        } catch (const py::cast_error&) {
            throw std::runtime_error(ClassName() + "." + what + " must return a dict of name -> Tensor");
        }
    }

    TensorMap parameters_;
    TensorMap gradients_;
};

PythonModule& AsPythonModule(Module& module) {
    auto* python_module = dynamic_cast<PythonModule*>(&module);
    if (!python_module) {
        throw py::type_error("parameters can be added only to modules defined in Python "
                             "(subclasses of pycyxwiz.Module)");
    }
    return *python_module;
}

// What a SequentialModel owns for a module that Python also holds. The model
// owns its modules through unique_ptr; Python owns this module's object. This
// keeps the Python object alive while the model lives and forwards every
// call. The lock is taken to release the object: models are destroyed on
// threads that do not hold it.
class PythonHeldModule final : public Module {
public:
    PythonHeldModule(py::object owner, Module* target) : owner_(std::move(owner)), target_(target) {}

    ~PythonHeldModule() override {
        if (Py_IsInitialized()) {
            py::gil_scoped_acquire gil;
            owner_ = py::object();
        } else {
            owner_.release();  // the interpreter is gone; nothing to release
        }
    }

    Tensor Forward(const Tensor& input) override { return target_->Forward(input); }
    Tensor Backward(const Tensor& grad_output) override { return target_->Backward(grad_output); }
    TensorMap GetParameters() override { return target_->GetParameters(); }
    void SetParameters(const TensorMap& params) override { target_->SetParameters(params); }
    TensorMap GetGradients() override { return target_->GetGradients(); }
    bool HasParameters() const override { return target_->HasParameters(); }
    std::string GetName() const override { return target_->GetName(); }
    void SetTraining(bool training) override {
        is_training_ = training;
        target_->SetTraining(training);
    }
    IncrementalDecoding GetIncrementalDecoding() const override { return target_->GetIncrementalDecoding(); }
    Tensor ForwardIncremental(const Tensor& input, size_t position_offset) override {
        return target_->ForwardIncremental(input, position_offset);
    }
    void ResetIncrementalState() override { target_->ResetIncrementalState(); }
    size_t InputCount() const override { return target_->InputCount(); }
    Tensor ForwardInputs(const std::vector<Tensor>& inputs) override { return target_->ForwardInputs(inputs); }
    std::vector<Tensor> BackwardInputs(const Tensor& grad_output) override {
        return target_->BackwardInputs(grad_output);
    }

private:
    py::object owner_;
    Module* target_;
};

}  // namespace

void BindModuleBase(py::module_& m) {
    py::class_<Module, PythonModule>(m, "Module",
        "Base class of every module. Subclass it in Python to write your own: define "
        "forward(self, x) and backward(self, grad_output), call super().__init__() first, "
        "and declare learnable tensors with add_parameter.")
        .def(py::init<>(), "Create a module (for Python subclasses)")
        .def("forward", &Module::Forward, py::arg("input"), "Forward pass through the module")
        .def("backward", &Module::Backward, py::arg("grad_output"), "Backward pass")
        .def("get_parameters", &Module::GetParameters, "Get module parameters")
        .def("set_parameters", &Module::SetParameters, py::arg("params"), "Set module parameters")
        .def("get_gradients", &Module::GetGradients, "Get parameter gradients")
        .def("has_parameters", &Module::HasParameters, "Check if module has trainable parameters")
        .def("get_name", &Module::GetName, "Get module name")
        .def("set_training", &Module::SetTraining, py::arg("training"), "Set training mode")
        .def("is_training", &Module::IsTraining, "Check if in training mode")
        .def("freeze", &Module::Freeze, "Freeze module (disable parameter updates)")
        .def("unfreeze", &Module::Unfreeze, "Unfreeze module (enable parameter updates)")
        .def("is_trainable", &Module::IsTrainable, "Check if module is trainable")
        .def("add_parameter", [](Module& self, const std::string& name, const Tensor& value) {
                AsPythonModule(self).AddParameter(name, value);
            }, py::arg("name"), py::arg("value"),
            "Declare a learnable tensor. The optimizer updates it; read it with parameter(name)")
        .def("parameter", [](Module& self, const std::string& name) {
                return AsPythonModule(self).Parameter(name);
            }, py::arg("name"), "The current value of a learnable tensor")
        .def("set_gradient", [](Module& self, const std::string& name, const Tensor& gradient) {
                AsPythonModule(self).SetGradient(name, gradient);
            }, py::arg("name"), py::arg("gradient"),
            "Store the loss gradient of a parameter; call it in backward")
        .def("gradient", [](Module& self, const std::string& name) {
                return AsPythonModule(self).Gradient(name);
            }, py::arg("name"), "The last gradient stored for a parameter")
        .def("parameter_names", [](Module& self) {
                return AsPythonModule(self).ParameterNames();
            }, "Names of the parameters added with add_parameter");
}

void BindGradientCheck(py::module_& m) {
    py::class_<cyxwiz::GradientCheckResult>(m, "GradientCheckResult",
        "Result of pycyxwiz.gradcheck: passed, the largest relative error, where it is, "
        "and an error text when the module could not be checked")
        .def_readonly("passed", &cyxwiz::GradientCheckResult::passed)
        .def_readonly("tolerance", &cyxwiz::GradientCheckResult::tolerance)
        .def_readonly("max_relative_error", &cyxwiz::GradientCheckResult::max_relative_error)
        .def_readonly("worst", &cyxwiz::GradientCheckResult::worst,
                      "Where the largest error is: 'input' or 'parameter <name>'")
        .def_readonly("error", &cyxwiz::GradientCheckResult::error,
                      "Set when the module could not be checked at all")
        .def("__bool__", [](const cyxwiz::GradientCheckResult& r) { return r.passed; })
        .def("__str__", &cyxwiz::DescribeGradientCheck)
        .def("__repr__", [](const cyxwiz::GradientCheckResult& r) {
            return "<GradientCheckResult " + cyxwiz::DescribeGradientCheck(r) + ">";
        });

    m.def("gradcheck",
        [](py::function make, std::vector<std::vector<size_t>> input_shapes, uint32_t seed,
           double tolerance, double step, float input_range) {
            cyxwiz::GradientCheckOptions options;
            if (!input_shapes.empty()) options.input_shapes = std::move(input_shapes);
            options.seed = seed;
            options.tolerance = tolerance;
            options.step = step;
            options.input_range = input_range;
            // The checker owns the modules it creates; each keeps its Python
            // object alive like a module added to a Sequential.
            const auto factory = [&make](const std::vector<size_t>& shape) -> std::unique_ptr<Module> {
                py::gil_scoped_acquire gil;
                py::object module = make(shape);
                if (!py::isinstance<Module>(module)) {
                    throw std::runtime_error("the factory must return a pycyxwiz.Module, not " +
                                             std::string(py::str(py::type::handle_of(module).attr("__name__"))));
                }
                Module* target = module.cast<Module*>();
                return std::make_unique<PythonHeldModule>(std::move(module), target);
            };
            return cyxwiz::CheckModuleGradients(factory, options);
        },
        py::arg("make"), py::arg("input_shapes") = std::vector<std::vector<size_t>>{},
        py::arg("seed") = 125u, py::arg("tolerance") = 2e-2, py::arg("step") = 1e-2,
        py::arg("input_range") = 2.0f,
        "Check a module's backward against central differences of its forward, for the input "
        "and every parameter. make(input_shape) returns a fresh module; input_shape includes the "
        "batch. Default shapes [2, 3] and [3, 4, 5]. The same checker as the Engine's "
        "Check gradient.");
}

void AddModuleToSequential(cyxwiz::SequentialModel& model, py::object module) {
    if (!py::isinstance<Module>(module)) {
        throw py::type_error("Sequential.add expects a pycyxwiz.Module, not " +
                             std::string(py::str(py::type::handle_of(module).attr("__name__"))));
    }
    Module* target = module.cast<Module*>();
    model.AddModule(std::make_unique<PythonHeldModule>(std::move(module), target));
}

void InstallNodesSubmodule(py::module_& m) {
    std::string source;
    for (const char* const* piece = kPycyxwizNodesSource; *piece != nullptr; ++piece) source += *piece;
    py::module_ nodes = m.def_submodule("nodes", "Describe a Python module as a CyxWiz node");
    py::dict scope = nodes.attr("__dict__");
    scope["_cyx"] = m;
    py::exec(source, scope);
    py::module_::import("sys").attr("modules")["pycyxwiz.nodes"] = nodes;
}

}  // namespace pycyxwiz_bindings
