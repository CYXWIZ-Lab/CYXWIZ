// Python setup view (TOFIX129 A2-3): chip and dialog wording for scanning,
// ready, choose and missing states.
#include "../src/core/python_setup_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::pythonsetup;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    ScanResult none;
    PythonView v = BuildPythonView(none, true, "3.12", "");
    Check(v.chip == "Checking Python..." && v.level == 1 && !v.ready, "scanning");

    ScanResult s;
    s.scanned = true;
    s.seconds = 2.43;
    s.found = {{"3.13.2", "C:/Python313/python.exe", true, true, false, "This Engine embeds Python 3.12; found 3.13.2"},
               {"3.12.8", "C:/Python312/python.exe", true, true, true, ""},
               {"3.11.9", "C:/Python311/python.exe", true, true, false, "Python version too old"},
               {"3.12.4", "C:/Users/me/Python312/python.exe", true, true, true, ""}};

    v = BuildPythonView(s, false, "3.12", "C:/Python312/python.exe");
    Check(v.ready && v.level == 0 && v.chip == "Python 3.12.8 ready", "ready chip");
    Check(v.usable.size() == 2 && v.unusable.size() == 2, "usable and unusable split");
    Check(v.usable[v.selected].path == "C:/Python312/python.exe", "selected is the one in use");
    Check(v.unusable[0].reason.find("3.12") != std::string::npos, "unusable keeps its reason");
    Check(v.scan_line == "Scanned in 2.4 s.", "scan line");

    v = BuildPythonView(s, false, "3.12", "");
    Check(!v.ready && v.level == 2 && v.chip == "Choose a Python", "usable but none in use");

    ScanResult missing;
    missing.scanned = true;
    missing.found = {s.found[0], s.found[2]};
    v = BuildPythonView(missing, false, "3.12", "");
    Check(!v.ready && v.chip == "Python not found" && v.usable.empty() && v.unusable.size() == 2, "missing");
    Check(v.headline.find("3.12") != std::string::npos && !v.install_hint.empty(), "missing says what to install");

    ScanResult custom;
    custom.scanned = true;
    custom.configured_path_set = true;
    custom.configured_path = "D:/tools/python/python.exe";
    custom.configured_ok = true;
    v = BuildPythonView(custom, false, "3.12", "D:/tools/python/python.exe");
    Check(v.ready && v.usable.size() == 1 && v.usable[0].path == custom.configured_path, "saved custom path counts as ready");

    ScanResult broken;
    broken.scanned = true;
    broken.configured_path_set = true;
    broken.configured_path = "D:/gone/python.exe";
    v = BuildPythonView(broken, false, "3.12", "");
    Check(v.headline.find("no longer works") != std::string::npos, "broken saved path is reported");

    std::cout << "python setup presentation: scanning, ready, choose, missing, custom path. OK\n";
    return 0;
}
