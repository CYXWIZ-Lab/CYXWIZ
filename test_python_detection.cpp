// Standalone test for Python detection
#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;

std::string FindPython() {
    // Try common Python commands
    std::vector<std::string> candidates = {
        "py -3.12",
        "py -3",
        "python",
        "python3",
        "python3.12"
    };

    for (const auto& cmd : candidates) {
        std::string test_cmd = cmd + " --version 2>&1";
        FILE* pipe = _popen(test_cmd.c_str(), "r");
        if (pipe) {
            char buffer[256];
            std::string result;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            _pclose(pipe);

            // Check if output contains "Python"
            if (result.find("Python") != std::string::npos) {
                std::cout << "✅ Found: " << cmd << " -> " << result;
                return cmd;
            }
        }
    }

    std::cout << "❌ No Python found in PATH" << std::endl;
    return "";
}

int main() {
    std::cout << "=== Quick Python Detection Test ===" << std::endl;
    std::cout << std::endl;

    std::string python_cmd = FindPython();

    if (python_cmd.empty()) {
        std::cout << std::endl;
        std::cout << "Please install Python 3.12 or newer" << std::endl;
        return 1;
    }

    // Get detailed info
    std::string info_cmd = python_cmd + " -c \"import sys; print(f'Version: {sys.version}'); print(f'Executable: {sys.executable}'); print(f'Prefix: {sys.prefix}')\"";

    std::cout << std::endl;
    std::cout << "Python Info:" << std::endl;
    system(info_cmd.c_str());

    // Check venv module
    std::cout << std::endl;
    std::string venv_cmd = python_cmd + " -m venv --help >nul 2>&1";
    int venv_result = system(venv_cmd.c_str());

    std::cout << "venv module: " << (venv_result == 0 ? "✅ Available" : "❌ Missing") << std::endl;

    // Check pip
    std::string pip_cmd = python_cmd + " -m pip --version >nul 2>&1";
    int pip_result = system(pip_cmd.c_str());

    std::cout << "pip: " << (pip_result == 0 ? "✅ Available" : "❌ Missing") << std::endl;

    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;

    return 0;
}
