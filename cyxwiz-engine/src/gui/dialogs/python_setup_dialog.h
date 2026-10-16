#pragma once

// Python for scripting (TOFIX129 A2-3). Replaces the Python Setup Wizard,
// which scanned on the first frame and blocked the Engine. The scan now runs
// on a worker while the start page is up; the start page chip shows its state
// and opens this dialog. The dialog opens by itself only when no usable
// interpreter was found. Wording comes from core/python_setup_presentation.

#include "../../core/python_setup_presentation.h"

#include <future>
#include <string>

namespace cyxwiz {

class PythonSetupDialog {
public:
    enum class Outcome { None, Ready, ContinueWithout };

    // Starts a scan on a worker; ignored while one is running. use_cache:
    // take the last scan's result when nothing changed since (a start);
    // false scans every interpreter again ("Scan again").
    void StartScan(bool use_cache = true);
    // Call every frame: collects a finished scan. Returns true on the frame
    // the scan finished.
    bool Poll();

    bool Scanning() const { return scanning_; }
    // The scan finished and the Engine has a usable interpreter, or the user
    // chose to continue without scripting.
    bool Settled() const { return !scanning_ && scan_.scanned && (View().ready || without_scripting_); }
    pythonsetup::PythonView View() const;

    void Open() { request_open_ = true; }
    Outcome Render();

private:
    void Browse();

    std::future<pythonsetup::ScanResult> worker_;
    bool scanning_ = false;
    pythonsetup::ScanResult scan_;
    std::string in_use_;
    int choice_ = 0;
    bool request_open_ = false;
    bool without_scripting_ = false;
    std::string browse_note_;
    bool browse_ok_ = false;
};

}  // namespace cyxwiz
