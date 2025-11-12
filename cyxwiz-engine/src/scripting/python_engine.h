#pragma once
#include <string>

namespace scripting {

class PythonEngine {
public:
    PythonEngine();
    ~PythonEngine();

    bool Initialize();
    void Shutdown();

    bool ExecuteScript(const std::string& script);
    bool ExecuteFile(const std::string& filepath);

private:
    bool initialized_;
};

} // namespace scripting
