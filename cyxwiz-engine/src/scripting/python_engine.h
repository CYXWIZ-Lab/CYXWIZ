#pragma once

#include <string>

namespace scripting {

// What opening a project did to a running interpreter.
enum class PythonProjectSwitch {
    Unchanged,     // already the project's environment, or Python not started
    Switched,      // moved to the project's environment in place
    RestartNeeded, // the project's environment needs a new Engine process
    Busy           // Python is running something; try again later
};

} // namespace scripting

#ifdef CYXWIZ_HAS_PYTHON

#include <Python.h>

namespace scripting {

class PythonEngine {
public:
    PythonEngine();
    ~PythonEngine();

    bool Initialize(std::string* error_out = nullptr);
    void Shutdown();
    // Moves a running interpreter to the active project's environment (or
    // the Engine's Python with no project). `detail_out`: the new
    // interpreter, or why a restart is needed.
    PythonProjectSwitch SwitchToProjectEnvironment(std::string* detail_out = nullptr);
    bool IsInitialized() const { return initialized_; }
    bool HasInterpreterMismatch(std::string* reason_out = nullptr) const;

    // Interpreter the running session was started with ("project"/"system").
    const std::string& ActiveInterpreterPath() const { return active_interpreter_path_; }
    const std::string& ActiveSource() const { return active_source_; }
    // Interpreter the next start would use (project venv, configured,
    // bundled or detected). Does not log.
    std::string PreviewInterpreterPath() const;

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
    std::string active_scripts_path_;  // the project scripts folder on sys.path

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
    PythonProjectSwitch SwitchToProjectEnvironment(std::string* = nullptr) { return PythonProjectSwitch::Unchanged; }
    bool IsInitialized() const { return false; }
    bool HasInterpreterMismatch(std::string* = nullptr) const { return false; }
    const std::string& ActiveInterpreterPath() const { static const std::string e; return e; }
    const std::string& ActiveSource() const { static const std::string e; return e; }
    std::string PreviewInterpreterPath() const { return {}; }

    bool ExecuteScript(const std::string&) { return false; }
    bool ExecuteFile(const std::string&) { return false; }

    void ReleaseGIL() {}
    void AcquireGIL() {}
};

} // namespace scripting

#endif // CYXWIZ_HAS_PYTHON
