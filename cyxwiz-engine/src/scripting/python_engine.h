#pragma once

#include <string>

#ifdef CYXWIZ_HAS_PYTHON

#include <Python.h>

namespace scripting {

class PythonEngine {
public:
    PythonEngine();
    ~PythonEngine();

    bool Initialize(std::string* error_out = nullptr);
    void Shutdown();
    bool ReloadForProject(std::string* error_out = nullptr);
    bool IsInitialized() const { return initialized_; }
    bool HasInterpreterMismatch(std::string* reason_out = nullptr) const;

    bool ExecuteScript(const std::string& script);
    bool ExecuteFile(const std::string& filepath);

    // GIL management for multi-threaded use
    // After initialization, call ReleaseGIL() to allow background threads to use Python
    void ReleaseGIL();
    void AcquireGIL();

private:
    struct PythonSelection {
        std::string interpreter_path;  // Project venv or system Python
        std::string source;            // "project" or "system"
    };

    bool initialized_ = false;
    bool initialized_by_us_ = false;  // True if we called py::initialize_interpreter()
    PyThreadState* main_thread_state_ = nullptr;  // Saved when releasing GIL
    std::string effective_interpreter_path_;
    std::string active_interpreter_path_;
    std::string active_source_;

    // Configure PYTHONHOME before initialization (venv or system Python)
    bool ConfigurePythonHome(std::string* error_out);

    // Configure custom Python packages path from EngineConfig
    void ConfigureCustomPythonPath();

    PythonSelection ResolvePythonConfig() const;
    bool ValidateSelection(const PythonSelection& selection, std::string* error_out) const;
    void ApplySelection(const PythonSelection& selection);

};

} // namespace scripting

#else // !CYXWIZ_HAS_PYTHON

// Stub implementation when Python is disabled
namespace scripting {

class PythonEngine {
public:
    PythonEngine() = default;
    ~PythonEngine() = default;

    bool Initialize(std::string* = nullptr) { return false; }
    void Shutdown() {}
    bool ReloadForProject(std::string* = nullptr) { return false; }
    bool IsInitialized() const { return false; }
    bool HasInterpreterMismatch(std::string* = nullptr) const { return false; }

    bool ExecuteScript(const std::string&) { return false; }
    bool ExecuteFile(const std::string&) { return false; }

    void ReleaseGIL() {}
    void AcquireGIL() {}
};

} // namespace scripting

#endif // CYXWIZ_HAS_PYTHON
