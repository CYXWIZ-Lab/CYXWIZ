#include "python_setup_presentation.h"

#include <cstdio>

namespace cyxwiz::pythonsetup {

PythonView BuildPythonView(const ScanResult& scan, bool scanning, const std::string& required, const std::string& in_use) {
    PythonView v;
    v.explanation = "This Engine embeds Python " + required +
                    ". Scripts, the Python console and project environments need a matching interpreter with the venv "
                    "module. Everything else in the Engine works without it.";
#if defined(_WIN32)
    v.install_hint = "Install Python " + required + " from python.org/downloads and keep \"pip\" ticked in the installer. "
                     "Then press Scan again, or point the Engine at the python.exe you installed.";
#elif defined(__APPLE__)
    v.install_hint = "Install Python " + required + " from python.org/downloads or with \"brew install python@" + required +
                     "\". Then press Scan again, or point the Engine at the interpreter you installed.";
#else
    v.install_hint = "Install Python " + required + " and its venv module, for example \"sudo apt install python" + required +
                     " python" + required + "-venv\". Then press Scan again, or point the Engine at the interpreter you installed.";
#endif
    if (scanning || !scan.scanned) {
        v.chip = "Checking Python...";
        v.level = 1;
        v.headline = "Looking for Python " + required + "...";
        return v;
    }
    for (const auto& c : scan.found) (c.usable ? v.usable : v.unusable).push_back(c);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Scanned in %.1f s.", scan.seconds);
    v.scan_line = buf;

    for (size_t i = 0; i < v.usable.size(); ++i) {
        if (!in_use.empty() && v.usable[i].path == in_use) {
            v.ready = true;
            v.selected = static_cast<int>(i);
        }
    }
    if (!v.ready && scan.configured_ok && !scan.configured_path.empty()) {
        // The saved interpreter is usable but was not in the scan list (custom location).
        v.usable.insert(v.usable.begin(), Candidate{"", scan.configured_path, true, true, true, "", ""});
        v.ready = in_use.empty() || in_use == scan.configured_path;
        v.selected = 0;
    }

    if (v.ready) {
        const Candidate& c = v.usable[v.selected];
        v.chip = c.version.empty() ? "Python ready" : "Python " + c.version + " ready";
        v.level = 0;
        v.headline = v.usable.size() > 1 ? "Python " + required + " is ready. Another usable interpreter was found."
                                         : "Python " + required + " is ready.";
        return v;
    }
    v.level = 2;
    if (!v.usable.empty()) {
        v.chip = "Choose a Python";
        v.headline = v.usable.size() == 1 ? "One usable Python was found. Use it?" : "Several usable Pythons were found. Choose one.";
        v.selected = 0;
    } else {
        v.chip = "Python not found";
        v.headline = "No Python " + required + " with the venv module was found";
        if (scan.configured_path_set && !scan.configured_ok)
            v.headline += "; the interpreter saved in the settings no longer works";
    }
    return v;
}

}  // namespace cyxwiz::pythonsetup
