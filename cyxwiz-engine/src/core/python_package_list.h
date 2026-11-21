// Packages to install in each new project environment (Preferences >
// Python, TOFIX129 B9-9): the text typed there, and the pip command. No ImGui.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::core {

// Names split on spaces and commas, empties and repeats dropped, order kept.
// A word starting with '-' is a pip option, not a package: it goes to
// `ignored` (when given) and never reaches pip.
std::vector<std::string> ParsePackageList(const std::string& text, std::vector<std::string>* ignored = nullptr);
// Space separated, as the field shows it
std::string JoinPackageList(const std::vector<std::string>& packages);
// python -m pip install <packages>, run with the environment's interpreter
std::vector<std::string> PipInstallCommand(const std::string& python, const std::vector<std::string>& packages);

}  // namespace cyxwiz::core
