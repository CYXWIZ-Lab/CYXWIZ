#include "../src/core/python_detector.h"
#include <iostream>

using namespace cyxwiz::core;

int main() {
    std::cout << "=== CyxWiz Python Detector Test ===" << std::endl;
    std::cout << std::endl;

    // Test 1: Find all Python installations
    std::cout << "[TEST 1] Finding all Python installations..." << std::endl;
    auto all_pythons = PythonDetector::FindAllPythonInstallations();

    if (all_pythons.empty()) {
        std::cout << "  ❌ No Python installations found!" << std::endl;
    } else {
        std::cout << "  ✅ Found " << all_pythons.size() << " Python installation(s):" << std::endl;
        for (const auto& python : all_pythons) {
            std::cout << "    - Python " << python.version << std::endl;
            std::cout << "      Path: " << python.executable_path << std::endl;
            std::cout << "      Home: " << python.home << std::endl;
            std::cout << "      venv: " << (python.has_venv_module ? "✅" : "❌") << std::endl;
            std::cout << "      pip: " << (python.has_pip ? "✅" : "❌") << std::endl;

            bool meets_req = PythonDetector::MeetsRequirements(python);
            std::cout << "      Requirements: " << (meets_req ? "✅ PASS" : "❌ FAIL");
            if (!meets_req) {
                std::cout << " - " << PythonDetector::GetRequirementError(python);
            }
            std::cout << std::endl << std::endl;
        }
    }

    // Test 2: Find best Python
    std::cout << "[TEST 2] Finding best Python for CyxWiz..." << std::endl;
    auto best_python = PythonDetector::FindBestPython();

    if (!best_python.has_value()) {
        std::cout << "  ❌ No suitable Python 3.12+ found!" << std::endl;
        std::cout << std::endl;
        std::cout << "Please install Python 3.12 or newer:" << std::endl;
        std::cout << "  https://www.python.org/downloads/" << std::endl;
        return 1;
    }

    std::cout << "  ✅ Best Python: " << best_python->version << std::endl;
    std::cout << "     Path: " << best_python->executable_path << std::endl;
    std::cout << std::endl;

    // Test 3: Check specific Python
    std::cout << "[TEST 3] Checking system 'python' command..." << std::endl;
    auto system_python = PythonDetector::CheckPython("python");

    if (system_python.has_value()) {
        std::cout << "  ✅ Found: Python " << system_python->version << std::endl;
        std::cout << "     " << system_python->executable_path << std::endl;
    } else {
        std::cout << "  ℹ️  'python' command not found (this is OK)" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}
