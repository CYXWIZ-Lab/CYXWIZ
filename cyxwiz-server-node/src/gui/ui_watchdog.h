// ui_watchdog.h - names what blocks the UI thread (TOFIX139 hang).
// The frame loop reports each frame; a watchdog thread notices when no frame
// has been drawn for a few seconds, captures the UI thread's stack and logs
// it (module!symbol+offset when the PDB sits next to the executable). That
// turns "stopped interacting with Windows" into a line in the log.
#pragma once

#include <chrono>

namespace cyxwiz::servernode::gui {

class UiWatchdog {
public:
    // Call from the UI thread, once, before the frame loop.
    void Start(std::chrono::milliseconds stall_after = std::chrono::milliseconds(3000));
    // Call from the UI thread every frame.
    void Frame();
    // Call from the UI thread when the loop ends.
    void Stop();

    ~UiWatchdog();
};

}  // namespace cyxwiz::servernode::gui
