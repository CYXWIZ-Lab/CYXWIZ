#include "../src/core/python_detector.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using cyxwiz::core::PythonDetector;

PythonDetector::PythonInstallation MakePython(int major, int minor, bool has_venv = true) {
    PythonDetector::PythonInstallation python;
    python.major = major;
    python.minor = minor;
    python.micro = 0;
    python.version = std::to_string(major) + "." + std::to_string(minor) + ".0";
    python.has_venv_module = has_venv;
    return python;
}

bool Expect(bool condition, const char* message) {
    if (condition) {
        return true;
    }
    std::cerr << "FAILED: " << message << '\n';
    return false;
}

} // namespace

int main() {
    bool passed = true;

    passed &= Expect(!PythonDetector::MeetsRequirements(MakePython(3, 11)),
                     "Python 3.11 must be rejected");
    passed &= Expect(PythonDetector::MeetsRequirements(MakePython(3, 12)),
                     "Python 3.12 must be accepted");
    passed &= Expect(PythonDetector::MeetsRequirements(MakePython(3, 13)),
                     "Python 3.13 must be accepted");
    passed &= Expect(!PythonDetector::MeetsRequirements(MakePython(3, 14)),
                     "Python 3.14 must be rejected until it is explicitly supported");
    passed &= Expect(!PythonDetector::MeetsRequirements(MakePython(3, 12, false)),
                     "Python without venv must be rejected");
    passed &= Expect(PythonDetector::GetRequirementError(MakePython(3, 14)).find("unsupported") != std::string::npos,
                     "Python 3.14 must report an unsupported-version error");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
