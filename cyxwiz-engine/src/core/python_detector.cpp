#include "python_detector.h"
#include "child_process.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace fs = std::filesystem;

namespace cyxwiz {
namespace core {

namespace {

// "py -3.12" (the Python launcher and its version) or one interpreter path.
std::vector<std::string> InterpreterArgs(const std::string& path) {
    if (path.rfind("py -", 0) != 0) return {path};
    std::vector<std::string> args;
    std::istringstream words(path);
    for (std::string word; words >> word;) args.push_back(word);
    return args;
}

} // namespace

std::vector<std::string> PythonDetector::GetCandidatePaths() {
    std::vector<std::string> candidates;

#ifdef _WIN32
    // 1. Python Launcher (py.exe) - most reliable on Windows
    candidates.push_back("py -3.13");
    candidates.push_back("py -3.12");
    candidates.push_back("py -3");

    // 2. Direct commands in PATH
    candidates.push_back("python");
    candidates.push_back("python3");
    candidates.push_back("python3.13");
    candidates.push_back("python3.12");

    // 3. Common installation directories
    std::vector<std::string> common_paths = {
        "C:\\Python313\\python.exe",
        "C:\\Python312\\python.exe",
        "C:\\Program Files\\Python313\\python.exe",
        "C:\\Program Files\\Python312\\python.exe",
        "C:\\Program Files (x86)\\Python313\\python.exe",
        "C:\\Program Files (x86)\\Python312\\python.exe",
    };

    // 4. Microsoft Store installations (AppData\Local\Microsoft\WindowsApps)
    const char* localappdata = std::getenv("LOCALAPPDATA");
    if (localappdata) {
        fs::path ms_store_path = fs::path(localappdata) / "Microsoft" / "WindowsApps";
        if (fs::exists(ms_store_path)) {
            candidates.push_back((ms_store_path / "python.exe").string());
            candidates.push_back((ms_store_path / "python3.exe").string());
            candidates.push_back((ms_store_path / "python3.13.exe").string());
            candidates.push_back((ms_store_path / "python3.12.exe").string());
        }
    }

    // 5. User profile Python installations
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        fs::path user_python = fs::path(userprofile) / "AppData" / "Local" / "Programs" / "Python";
        if (fs::exists(user_python)) {
            for (const auto& entry : fs::directory_iterator(user_python)) {
                if (entry.is_directory()) {
                    fs::path python_exe = entry.path() / "python.exe";
                    if (fs::exists(python_exe)) {
                        candidates.push_back(python_exe.string());
                    }
                }
            }
        }
    }

    candidates.insert(candidates.end(), common_paths.begin(), common_paths.end());

#else  // Linux/macOS
    // 1. Commands in PATH
    candidates.push_back("python3.13");
    candidates.push_back("python3.12");
    candidates.push_back("python3");
    candidates.push_back("python");

    // 2. Common system locations
    std::vector<std::string> common_paths = {
        "/usr/bin/python3.13",
        "/usr/bin/python3.12",
        "/usr/bin/python3",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3",
        "/opt/python3.13/bin/python3",
        "/opt/python3.12/bin/python3",
    };

#ifdef __APPLE__
    // macOS Homebrew locations
    common_paths.push_back("/opt/homebrew/bin/python3.13");
    common_paths.push_back("/opt/homebrew/bin/python3.12");
    common_paths.push_back("/opt/homebrew/bin/python3");
    common_paths.push_back("/usr/local/bin/python3.13");
    common_paths.push_back("/usr/local/bin/python3.12");

    // Check Homebrew Cellar
    if (fs::exists("/opt/homebrew/Cellar/python@3.12")) {
        for (const auto& entry : fs::recursive_directory_iterator("/opt/homebrew/Cellar/python@3.12")) {
            if (entry.path().filename() == "python3" && entry.is_regular_file()) {
                candidates.push_back(entry.path().string());
            }
        }
    }
#endif

    candidates.insert(candidates.end(), common_paths.begin(), common_paths.end());
#endif

    return candidates;
}

std::optional<PythonDetector::PythonInstallation> PythonDetector::CheckPython(const std::string& path) {
    PythonInstallation result;

    // The interpreter's facts as JSON.
    std::vector<std::string> args = InterpreterArgs(path);
    args.insert(args.end(), {"-c", "import sys, json; print(json.dumps({'version': '.'.join(map(str, "
                                   "sys.version_info[:3])), 'executable': sys.executable, 'prefix': sys.prefix}))"});
    std::string output;
    const int exit_code = RunHidden(args, &output);

    if (exit_code != 0 || output.empty()) {
        return std::nullopt;
    }

    // Parse JSON output
    try {
        nlohmann::json j = nlohmann::json::parse(output);

        result.version = j["version"].get<std::string>();
        result.executable_path = j["executable"].get<std::string>();
        result.home = j["prefix"].get<std::string>();

        // Parse version numbers
        std::regex version_regex(R"((\d+)\.(\d+)\.(\d+))");
        std::smatch match;
        if (std::regex_search(result.version, match, version_regex)) {
            result.major = std::stoi(match[1].str());
            result.minor = std::stoi(match[2].str());
            result.micro = std::stoi(match[3].str());
        } else {
            spdlog::warn("Failed to parse Python version: {}", result.version);
            return std::nullopt;
        }

    } catch (const std::exception& e) {
        spdlog::warn("Failed to parse Python info from '{}': {}", path, e.what());
        return std::nullopt;
    }

    result.has_venv_module = RunHidden({result.executable_path, "-m", "venv", "--help"}) == 0;
    result.has_pip = RunHidden({result.executable_path, "-m", "pip", "--version"}) == 0;

    return result;
}

bool PythonDetector::MeetsRequirements(const PythonInstallation& python) {
    // Python bindings and supported packages are validated for 3.12 and 3.13.
    if (python.major != 3 || python.minor < 12 || python.minor > 13) {
        return false;
    }
#ifdef CYXWIZ_PYTHON_LINKED_MINOR
    // The Engine embeds one libpython; a different minor version is not
    // ABI-compatible with it.
    if (python.minor != CYXWIZ_PYTHON_LINKED_MINOR) {
        return false;
    }
#endif

    // Require venv module
    if (!python.has_venv_module) {
        return false;
    }

    return true;
}

namespace {

int EmbeddedMinor() {
#ifdef CYXWIZ_PYTHON_LINKED_MINOR
    return CYXWIZ_PYTHON_LINKED_MINOR;
#else
    return 0;
#endif
}

} // namespace

std::string PythonDetector::SupportedVersionsText() {
    const int embedded = EmbeddedMinor();
    return embedded != 0 ? "3." + std::to_string(embedded) : "3.12-3.13";
}

std::string PythonDetector::GetRequirementError(const PythonInstallation& python) {
    return GetRequirementError(python, EmbeddedMinor());
}

std::string PythonDetector::GetRequirementError(const PythonInstallation& python, int embedded_minor) {
    // An embedded libpython accepts only its own minor version, so the
    // general 3.12-3.13 range would name versions this Engine cannot use.
    if (embedded_minor != 0 && (python.major != 3 || python.minor != embedded_minor)) {
        const std::string embedded = "3." + std::to_string(embedded_minor);
        return "This Engine embeds Python " + embedded + " and needs a matching " + embedded + "; found " +
               python.version;
    }

    if (python.major < 3 || (python.major == 3 && python.minor < 12)) {
        return "Python version too old (supported versions: 3.12-3.13; found " + python.version + ")";
    }

    if (python.major > 3 || (python.major == 3 && python.minor > 13)) {
        return "Python version unsupported (supported versions: 3.12-3.13; found " + python.version + ")";
    }

    if (!python.has_venv_module) {
        return "Python missing venv module (required for creating virtual environments)";
    }

    return "";
}

std::vector<PythonDetector::PythonInstallation> PythonDetector::FindAllPythonInstallations() {
    std::vector<PythonInstallation> installations;
    auto candidates = GetCandidatePaths();

    spdlog::info("Scanning for Python installations ({} candidates)...", candidates.size());

    for (const auto& candidate : candidates) {
        auto python = CheckPython(candidate);
        if (python.has_value()) {
            // Check for duplicates (same executable path)
            bool duplicate = false;
            for (const auto& existing : installations) {
                if (existing.executable_path == python->executable_path) {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                installations.push_back(*python);
                spdlog::info("Found Python: {} ({})", python->executable_path, python->version);
            }
        }
    }

    spdlog::info("Found {} Python installation(s)", installations.size());

    return installations;
}

std::optional<PythonDetector::PythonInstallation> PythonDetector::FindBestPython() {
    auto installations = FindAllPythonInstallations();

    // Filter to those meeting requirements
    std::vector<PythonInstallation> valid;
    for (const auto& python : installations) {
        if (MeetsRequirements(python)) {
            valid.push_back(python);
        } else {
            std::string error = GetRequirementError(python);
            spdlog::info("Python {} does not meet requirements: {}", python.executable_path, error);
        }
    }

    if (valid.empty()) {
        spdlog::warn("No supported Python {} installations found", SupportedVersionsText());
        return std::nullopt;
    }

    // All valid interpreters are supported; prefer the newest patch version.
    std::sort(valid.begin(), valid.end(), [](const PythonInstallation& a, const PythonInstallation& b) {
        if (a.major != b.major) return a.major > b.major;
        if (a.minor != b.minor) return a.minor > b.minor;
        return a.micro > b.micro;
    });

    spdlog::info("Best Python: {} ({})", valid[0].executable_path, valid[0].version);
    return valid[0];
}

} // namespace core
} // namespace cyxwiz
