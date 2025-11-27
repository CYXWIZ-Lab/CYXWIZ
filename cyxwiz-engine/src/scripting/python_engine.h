#pragma once
#include <string>
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
    bool initialized_;
    PyThreadState* main_thread_state_ = nullptr;  // Saved when releasing GIL
};

} // namespace scripting
