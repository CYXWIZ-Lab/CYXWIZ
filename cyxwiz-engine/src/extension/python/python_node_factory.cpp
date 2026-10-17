#include "python_node_factory.h"

#include <cyxwiz/sequential.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <utility>

#ifdef CYXWIZ_HAS_PYTHON
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

namespace cyxwiz {

namespace {

std::optional<double> ParseNumber(const std::string& text) {
    if (text.empty()) return std::nullopt;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size() || !std::isfinite(value)) return std::nullopt;
    return value;
}

std::optional<long long> ParseInteger(const std::string& text) {
    if (text.empty()) return std::nullopt;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (end != text.c_str() + text.size()) return std::nullopt;
    return value;
}

// "0.01-" (min), "-5" (max 5), "-1-1", "0-1". pycyxwiz writes "<min>-<max>".
bool ParseRange(const std::string& rule, std::optional<double>& min, std::optional<double>& max) {
    for (size_t i = 0; i < rule.size(); ++i) {
        if (rule[i] != '-' || (i > 0 && (rule[i - 1] == 'e' || rule[i - 1] == 'E'))) continue;
        const std::string left = rule.substr(0, i);
        const std::string right = rule.substr(i + 1);
        const auto lo = ParseNumber(left);
        const auto hi = ParseNumber(right);
        if ((left.empty() || lo) && (right.empty() || hi) && !(left.empty() && right.empty())) {
            min = lo;
            max = hi;
            return true;
        }
    }
    return false;
}

// Python's str() of a number as a Param limit was written: "0.01", "5".
std::string Limit(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

}  // namespace

std::string CheckPythonNodeSettings(const ExtensionNodeDescriptor& descriptor,
                                    const std::map<std::string, std::string>& settings) {
    const auto& declared = descriptor.metadata.parameters;
    for (const auto& [name, value] : settings) {
        const bool known = std::any_of(declared.begin(), declared.end(),
                                       [&](const ParameterDefinition& p) { return p.name == name; });
        if (!known) return descriptor.type_name + " has no setting " + name;
    }
    for (const auto& p : declared) {
        const auto it = settings.find(p.name);
        if (it == settings.end()) continue;  // the default applies
        const std::string& value = it->second;
        std::optional<double> number;
        if (p.type == "float") {
            number = ParseNumber(value);
            if (!number) return p.name + " must be a number, not '" + value + "'";
        } else if (p.type == "int") {
            const auto integer = ParseInteger(value);
            if (!integer) return p.name + " must be an integer, not '" + value + "'";
            number = static_cast<double>(*integer);
        } else if (p.type == "bool") {
            std::string lower;
            for (char c : value) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lower != "true" && lower != "false" && lower != "1" && lower != "0" && lower != "yes" &&
                lower != "no" && !lower.empty()) {
                return p.name + " must be true or false, not '" + value + "'";
            }
        } else if (p.type == "enum") {
            if (std::find(p.enum_values.begin(), p.enum_values.end(), value) == p.enum_values.end()) {
                std::string list;
                for (const auto& choice : p.enum_values) list += (list.empty() ? "" : ", ") + choice;
                return p.name + " must be one of " + list;
            }
        }
        std::optional<double> min;
        std::optional<double> max;
        if (number && !p.validation.empty() && ParseRange(p.validation, min, max)) {
            if (min && *number < *min) return p.name + " must be at least " + Limit(*min);
            if (max && *number > *max) return p.name + " must be at most " + Limit(*max);
        }
    }
    return {};
}

std::string PythonBundleModuleName(const std::string& bundle_id, const std::string& content_hash) {
    std::string name = "cyxwiz_node_";
    for (char c : bundle_id) name += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    const auto colon = content_hash.find(':');
    const std::string digits = colon == std::string::npos ? content_hash : content_hash.substr(colon + 1);
    return name + "_" + digits.substr(0, 16);
}

PythonNodeRuntime& PythonNodeRuntime::Instance() {
    static PythonNodeRuntime runtime;
    return runtime;
}

#ifdef CYXWIZ_HAS_PYTHON

bool PythonNodeRuntime::Available() { return true; }

bool PythonNodeRuntime::Ready(std::string& error) {
    if (Py_IsInitialized()) return true;
    wanted_.store(true);
    error = "Python is not running yet; the Engine starts it for Python nodes (open a project)";
    return false;
}

namespace {

// The message of a Python exception; with its traceback when `trace`.
std::string PythonError(py::error_already_set& error, bool trace) {
    if (trace) return error.what();
    try {
        return py::str(error.value()).cast<std::string>();
    } catch (...) {
        return error.what();
    }
}

// Owns a Python module object for a model (Part C8): forwards every call to
// the C++ module behind it and takes the lock to release the object, since
// models are destroyed on threads that do not hold it.
class PythonNodeModule final : public Module {
public:
    PythonNodeModule(py::object owner, Module* target) : owner_(std::move(owner)), target_(target) {}

    ~PythonNodeModule() override {
        if (Py_IsInitialized()) {
            py::gil_scoped_acquire gil;
            owner_ = py::object();
        } else {
            owner_.release();  // the interpreter is gone
        }
    }

    Tensor Forward(const Tensor& input) override { return target_->Forward(input); }
    Tensor Backward(const Tensor& grad_output) override { return target_->Backward(grad_output); }
    std::map<std::string, Tensor> GetParameters() override { return target_->GetParameters(); }
    void SetParameters(const std::map<std::string, Tensor>& params) override { target_->SetParameters(params); }
    std::map<std::string, Tensor> GetGradients() override { return target_->GetGradients(); }
    bool HasParameters() const override { return target_->HasParameters(); }
    std::string GetName() const override { return target_->GetName(); }
    void SetTraining(bool training) override {
        is_training_ = training;
        target_->SetTraining(training);
    }

private:
    py::object owner_;
    Module* target_;
};

class PythonTensorNodeFactory final : public ITensorNodeFactory {
public:
    PythonTensorNodeFactory(const PythonBundle& bundle, const PythonBundleNode& node)
        : folder_(bundle.folder.string()),
          module_name_(PythonBundleModuleName(bundle.id, bundle.content_hash)),
          entry_class_(node.entry_class),
          same_shape_(node.same_shape),
          descriptor_(node.descriptor) {}

    ExtensionShapeResult InferOutputShape(const std::vector<size_t>& input_shape,
                                          const std::map<std::string, std::string>& parameters) const override {
        if (const std::string refused = CheckPythonNodeSettings(descriptor_, parameters); !refused.empty()) {
            return {false, {}, refused};
        }
        if (same_shape_) return {true, input_shape, {}};
        std::string error;
        if (!PythonNodeRuntime::Instance().Ready(error)) return {false, {}, error};
        py::gil_scoped_acquire gil;
        try {
            py::object nodes = py::module_::import("pycyxwiz.nodes");
            py::object shape = nodes.attr("_engine_output_shape")(folder_, module_name_, entry_class_, parameters,
                                                                  input_shape);
            return {true, shape.cast<std::vector<size_t>>(), {}};
        } catch (py::error_already_set& e) {
            return {false, {}, PythonError(e, false)};
        } catch (const std::exception& e) {
            return {false, {}, e.what()};
        }
    }

    std::unique_ptr<Module> CreateModule(const std::vector<size_t>& input_shape,
                                         const std::map<std::string, std::string>& parameters,
                                         std::string& error) const override {
        if (const std::string refused = CheckPythonNodeSettings(descriptor_, parameters); !refused.empty()) {
            error = refused;
            return nullptr;
        }
        if (!PythonNodeRuntime::Instance().Ready(error)) return nullptr;
        py::gil_scoped_acquire gil;
        try {
            py::object nodes = py::module_::import("pycyxwiz.nodes");
            py::tuple made = nodes.attr("_engine_create")(folder_, module_name_, entry_class_, parameters,
                                                          input_shape);
            auto* target = reinterpret_cast<Module*>(made[1].cast<std::uintptr_t>());
            if (!target) {
                error = "pycyxwiz returned no module";
                return nullptr;
            }
            return std::make_unique<PythonNodeModule>(py::reinterpret_borrow<py::object>(made[0]), target);
        } catch (py::error_already_set& e) {
            error = PythonError(e, true);
            return nullptr;
        } catch (const std::exception& e) {
            error = e.what();
            return nullptr;
        }
    }

private:
    std::string folder_;
    std::string module_name_;
    std::string entry_class_;
    bool same_shape_;
    ExtensionNodeDescriptor descriptor_;
};

}  // namespace

std::shared_ptr<ITensorNodeFactory> MakePythonTensorNodeFactory(const PythonBundle& bundle,
                                                               const PythonBundleNode& node) {
    return std::make_shared<PythonTensorNodeFactory>(bundle, node);
}

#else  // !CYXWIZ_HAS_PYTHON

bool PythonNodeRuntime::Available() { return false; }

bool PythonNodeRuntime::Ready(std::string& error) {
    error = "Python is not available in this build of CyxWiz";
    return false;
}

namespace {

class UnavailablePythonFactory final : public ITensorNodeFactory {
public:
    ExtensionShapeResult InferOutputShape(const std::vector<size_t>&,
                                          const std::map<std::string, std::string>&) const override {
        return {false, {}, "Python is not available in this build of CyxWiz"};
    }
    std::unique_ptr<Module> CreateModule(const std::vector<size_t>&, const std::map<std::string, std::string>&,
                                         std::string& error) const override {
        error = "Python is not available in this build of CyxWiz";
        return nullptr;
    }
};

}  // namespace

std::shared_ptr<ITensorNodeFactory> MakePythonTensorNodeFactory(const PythonBundle&, const PythonBundleNode&) {
    return std::make_shared<UnavailablePythonFactory>();
}

#endif  // CYXWIZ_HAS_PYTHON

}  // namespace cyxwiz
