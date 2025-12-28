#pragma once

#include <string>

#ifdef CYXWIZ_HAS_PYTHON

#include <Python.h>

namespace scripting {

class PythonEngine {
public:
    PythonEngine();
    ~PythonEngine();

    bool Initialize();
    void Shutdown();

    bool ExecuteScript(const std::string& script);
    bool ExecuteFile(const std::string& filepath);

    // GIL management for multi-threaded use
    // After initialization, call ReleaseGIL() to allow background threads to use Python
    void ReleaseGIL();
    void AcquireGIL();

private:
    bool initialized_ = false;
    bool initialized_by_us_ = false;  // True if we called py::initialize_interpreter()
    PyThreadState* main_thread_state_ = nullptr;  // Saved when releasing GIL
};

} // namespace scripting

#else // !CYXWIZ_HAS_PYTHON

// Stub implementation when Python is disabled
namespace scripting {

class PythonEngine {
public:
    PythonEngine() = default;
    ~PythonEngine() = default;

    bool Initialize() { return false; }
    void Shutdown() {}

    bool ExecuteScript(const std::string&) { return false; }
    bool ExecuteFile(const std::string&) { return false; }

    void ReleaseGIL() {}
    void AcquireGIL() {}

private:
    bool initialized_ = false;
};

} // namespace scripting

#endif // CYXWIZ_HAS_PYTHON
