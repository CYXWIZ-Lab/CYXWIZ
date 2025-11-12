#include "python_engine.h"
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

namespace scripting {

PythonEngine::PythonEngine() : initialized_(false) {
    Initialize();
}

PythonEngine::~PythonEngine() {
    Shutdown();
}

bool PythonEngine::Initialize() {
    try {
        py::initialize_interpreter();
        spdlog::info("Python interpreter initialized");
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize Python: {}", e.what());
        return false;
    }
}

void PythonEngine::Shutdown() {
    if (initialized_) {
        py::finalize_interpreter();
        initialized_ = false;
    }
}

bool PythonEngine::ExecuteScript(const std::string& script) {
    if (!initialized_) return false;

    try {
        py::exec(script);
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::error("Python error: {}", e.what());
        return false;
    }
}

bool PythonEngine::ExecuteFile(const std::string& filepath) {
    if (!initialized_) return false;

    try {
        py::eval_file(filepath);
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::error("Python error: {}", e.what());
        return false;
    }
}

} // namespace scripting
