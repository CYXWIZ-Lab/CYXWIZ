#pragma once

// Runs a short helper program (python -c ..., python -m venv ...) and waits
// for it. No shell and, on Windows, no console window: a GUI app that went
// through system() or _popen() flashed a cmd.exe window for each call.

#include <string>
#include <vector>

namespace cyxwiz::core {

// args[0] is the program: a path, or a name looked up on PATH ("py",
// "python"). The other entries are passed as they are, one argument each.
// stdout goes to `output` when it is given and is discarded otherwise;
// stderr is always discarded (a warning there would spoil parsed output).
// Returns the program's exit code, or -1 when it could not start.
int RunHidden(const std::vector<std::string>& args, std::string* output = nullptr);

}  // namespace cyxwiz::core
