#pragma once

// Small platform actions shared by screens (TOFIX129): open a web page,
// show a file or folder in the system file manager.

#include <string>

namespace cyxwiz::ui {

// Opens the URL in the default browser. Returns false when it could not start.
bool OpenUrl(const std::string& url);

// Shows the file selected in its folder (Explorer, Finder) or opens the
// folder. Returns false when it could not start.
bool ShowInFileManager(const std::string& path);

}  // namespace cyxwiz::ui
