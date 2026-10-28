#pragma once

// Where a Python environment lives on disk, and moving a running interpreter
// from one environment to another (a project's venv) without a restart.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace scripting {

// <venv>/Scripts (Windows) or <venv>/bin.
bool IsVenvBinDir(const std::filesystem::path& path);
// The venv an interpreter belongs to (the folder with pyvenv.cfg), or empty.
std::filesystem::path ResolveVenvRoot(const std::filesystem::path& interpreter_path);
// The venv root, or the installation folder of a base interpreter.
std::filesystem::path ResolvePythonHomeFromInterpreter(const std::filesystem::path& interpreter_path);
std::filesystem::path FindPosixSitePackages(const std::filesystem::path& base_root);
// The site-packages folder of an interpreter (venv or base), or "" when it
// does not exist.
std::string DeriveSitePackagesPath(const std::string& interpreter_path);
// A value of <venv>/pyvenv.cfg ("home", "version"), trimmed.
std::optional<std::string> ReadPyvenvValue(const std::filesystem::path& venv_root, const std::string& key);
// The base installation a venv runs on ("home", made absolute).
std::optional<std::filesystem::path> ReadVenvBaseHome(const std::filesystem::path& venv_root);

std::filesystem::path NormalizePath(const std::filesystem::path& path);
bool IsPathUnderRoot(const std::filesystem::path& path, const std::filesystem::path& root);
// Case-insensitive on Windows.
bool IsSamePath(const std::filesystem::path& a, const std::filesystem::path& b);
std::string TrimWhitespace(const std::string& value);

struct PythonEnvironment {
    std::filesystem::path interpreter;
    std::filesystem::path venv_root;      // empty: a base installation
    std::filesystem::path home;           // the base installation it runs on
    std::filesystem::path site_packages;  // empty when missing
    std::string version;                  // pyvenv.cfg "version"; empty for a base
};
PythonEnvironment DescribePythonEnvironment(const std::filesystem::path& interpreter);

// Why the interpreter running `from` cannot move to `to` in place (a restart
// is needed), or "" when it can: both must run on the same base installation,
// and a venv's version must be the running one (major.minor).
std::string InPlaceSwitchBlocker(const PythonEnvironment& from, const PythonEnvironment& to,
                                 int running_major, int running_minor);

// Moves the running interpreter from `from` to `to`: the modules loaded from
// the old environment and the old scripts folder are forgotten, sys.path
// swaps both for the new ones (with the new site-packages' .pth files), and
// sys.prefix names the new environment. A compiled module of the old
// environment cannot be unloaded: then nothing changes and its names are
// returned (a restart is needed). Needs the GIL.
std::vector<std::string> SwitchEnvironmentInPlace(const PythonEnvironment& from, const PythonEnvironment& to,
                                                  const std::string& old_scripts, const std::string& new_scripts);

} // namespace scripting
