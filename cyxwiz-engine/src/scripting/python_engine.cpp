#include "python_engine.h"
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

namespace scripting {

PythonEngine::PythonEngine() : initialized_(false), main_thread_state_(nullptr) {
    Initialize();
}

PythonEngine::~PythonEngine() {
    Shutdown();
}

bool PythonEngine::Initialize() {
    try {
        // Check if Python is already initialized (by another PythonEngine instance)
        if (Py_IsInitialized()) {
            spdlog::info("Python interpreter already initialized (reusing existing)");
            initialized_ = true;
            initialized_by_us_ = false;  // We didn't initialize it, so don't finalize
            return true;
        }

        py::initialize_interpreter();
        spdlog::info("Python interpreter initialized");
        initialized_ = true;
        initialized_by_us_ = true;  // We initialized it, so we'll finalize it

        // Release the GIL so background threads can use Python
        // This is REQUIRED for multi-threaded Python execution
        ReleaseGIL();
        spdlog::info("GIL released for multi-threaded use");

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize Python: {}", e.what());
        return false;
    }
}

void PythonEngine::Shutdown() {
    if (initialized_ && initialized_by_us_) {
        // Only finalize if we were the ones who initialized Python
        // Reacquire the GIL before finalizing
        AcquireGIL();
        py::finalize_interpreter();
        initialized_ = false;
        main_thread_state_ = nullptr;
        spdlog::info("Python interpreter finalized");
    } else if (initialized_) {
        // We're using a shared interpreter, just mark as not initialized
        initialized_ = false;
    }
}

void PythonEngine::ReleaseGIL() {
    if (main_thread_state_ == nullptr) {
        // Save the current thread state and release the GIL
        main_thread_state_ = PyEval_SaveThread();
    }
}

void PythonEngine::AcquireGIL() {
    if (main_thread_state_ != nullptr) {
        // Restore the thread state and reacquire the GIL
        PyEval_RestoreThread(main_thread_state_);
        main_thread_state_ = nullptr;
    }
}

bool PythonEngine::ExecuteScript(const std::string& script) {
    if (!initialized_) return false;

    try {
        // Must acquire GIL since we released it after initialization
        py::gil_scoped_acquire acquire;
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
        // Must acquire GIL since we released it after initialization
        py::gil_scoped_acquire acquire;
        py::eval_file(filepath);
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::error("Python error: {}", e.what());
        return false;
    }
}

} // namespace scripting
