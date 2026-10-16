// Python for scripting: what the start page chip and the Python dialog
// show (TOFIX129 A2-3). Pure data in, data out; the scan itself runs on a
// worker in the application and fills ScanResult.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::pythonsetup {

struct Candidate {
    std::string version;   // "3.12.8"
    std::string path;      // python.exe
    bool venv = false;
    bool pip = false;
    bool usable = false;   // meets this Engine's requirements
    std::string reason;    // why not, when not usable
    std::string home;      // sys.prefix
};

struct ScanResult {
    bool scanned = false;
    bool configured_path_set = false;   // a path was saved in the Engine settings
    std::string configured_path;
    bool configured_ok = false;         // the saved path is usable
    std::vector<Candidate> found;       // every interpreter the scan found, best first
    double seconds = 0.0;
    bool from_cache = false;            // the last scan's result, nothing changed since
};

struct PythonView {
    // Chip on the start page.
    std::string chip;        // "Checking Python...", "Python 3.12.8 ready", "Python needs attention"
    int level = 1;           // 0 ready, 1 checking, 2 needs attention
    // Dialog.
    std::string headline;
    std::string explanation;
    std::vector<Candidate> usable;
    std::vector<Candidate> unusable;
    std::string scan_line;   // "Scanned in 2.4 s."
    std::string install_hint;
    bool ready = false;      // a usable interpreter is in use
    int selected = 0;        // index into usable of the one in use
};

// `required` is the version text this build needs ("3.12" or "3.12 or 3.13");
// `in_use` is the path the Engine uses now (empty when none).
PythonView BuildPythonView(const ScanResult& scan, bool scanning, const std::string& required, const std::string& in_use);

}  // namespace cyxwiz::pythonsetup
