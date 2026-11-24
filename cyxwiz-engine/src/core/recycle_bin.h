#pragma once

// A file the user deletes in the Engine goes where the system keeps deleted
// files, so it can be brought back: the Recycle Bin on Windows, the Trash on
// macOS and Linux (the freedesktop.org trash folder).

#include <filesystem>
#include <string>

namespace cyxwiz {

// "Recycle Bin" on Windows, "Trash" elsewhere: the word the user knows.
const char* RecycleBinName();

// Moves the file there. False, with the reason in `error` (when given), when
// it could not; the file is then where it was.
bool MoveToRecycleBin(const std::filesystem::path& path, std::string* error = nullptr);

}  // namespace cyxwiz
