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

    // An Engine that embeds Python 3.12 names only 3.12, never the 3.12-3.13
    // range (the HP walk showed "supported versions: 3.12-3.13" for 3.14).
    const std::string embedded_314 = PythonDetector::GetRequirementError(MakePython(3, 14), 12);
    passed &= Expect(embedded_314 == "This Engine embeds Python 3.12 and needs a matching 3.12; found 3.14.0",
                     "an embedded-3.12 Engine must ask for 3.12 when it finds 3.14");
    passed &= Expect(PythonDetector::GetRequirementError(MakePython(3, 13), 12).find("matching 3.12") != std::string::npos,
                     "an embedded-3.12 Engine must ask for 3.12 when it finds 3.13");
    passed &= Expect(PythonDetector::GetRequirementError(MakePython(3, 12), 12).empty(),
                     "Python 3.12 must suit an embedded-3.12 Engine");
    passed &= Expect(PythonDetector::GetRequirementError(MakePython(3, 12, false), 12).find("venv") != std::string::npos,
                     "a matching Python without venv must say so");
    passed &= Expect(PythonDetector::SupportedVersionsText() == "3.12-3.13",
                     "without an embedded Python the supported range is 3.12-3.13");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
