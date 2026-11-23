#include "python_node_calls.h"

#include "python_bundle_ui.h"
#include "../core/python_bundle_presentation.h"
#include "../extension/python/python_node_provider.h"

#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_PYTHON
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

namespace cyxwiz::python_calls {

std::future<Result> CallNodes(const char* function, std::string script, std::string folder, std::string name) {
    return std::async(std::launch::async, [function, script, folder, name]() -> Result {
#ifdef CYXWIZ_HAS_PYTHON
        if (!Py_IsInitialized()) return {false, "Python is not running"};
        py::gil_scoped_acquire gil;
        try {
            py::object nodes = py::module_::import("pycyxwiz.nodes");
            py::object out = folder.empty() ? nodes.attr(function)(script)
                                            : nodes.attr(function)(script, folder, name);
            return {true, py::str(out).cast<std::string>()};
        } catch (py::error_already_set& e) {
            return {false, e.what()};
        } catch (const std::exception& e) {
            return {false, e.what()};
        }
#else
        (void)function;
        return {false, "Python is not available in this build of CyxWiz"};
#endif
    });
}

bool FinishCreatedBundle(const std::filesystem::path& folder, std::string* error) {
    PythonBundle bundle;
    std::string read_error;
    if (!ReadPythonBundle(folder, bundle, read_error)) {
        if (error) *error = "The bundle was written but cannot be read: " + read_error;
        return false;
    }
    // The user wrote this code: approved as created (TOFIX125 decision D6).
    std::string approve_error;
    if (!PythonNodeApprovals::User().Approve(MakePythonNodeApproval(bundle, "created", TodayIsoDate()),
                                             &approve_error)) {
        spdlog::warn("Created {} but could not record its approval: {}", folder.string(), approve_error);
    }
    if (auto* provider = python_ui::Provider()) provider->Rescan();
    return true;
}

}  // namespace cyxwiz::python_calls
