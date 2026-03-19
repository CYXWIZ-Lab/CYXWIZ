#include "python_engine.h"

#ifdef CYXWIZ_HAS_PYTHON

#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include "../core/engine_config.h"
#include "../core/project_manager.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <optional>
#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <stdlib.h>  // _putenv_s
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#else
#include <stdlib.h>  // setenv
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <limits.h>
#include <libgen.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif
#endif

namespace py = pybind11;

namespace scripting {

namespace {

bool IsVenvBinDir(const std::filesystem::path& path) {
    auto name = path.filename().string();
#ifdef _WIN32
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name == "scripts";
#else
    return name == "bin";
#endif
}

std::filesystem::path ResolveVenvRoot(const std::filesystem::path& interpreter_path) {
    auto parent = interpreter_path.parent_path();
    if (!IsVenvBinDir(parent)) {
        return {};
    }
    auto root = parent.parent_path();
    if (root.empty()) {
        return {};
    }
    if (std::filesystem::exists(root / "pyvenv.cfg")) {
        return root;
    }
    return {};
}

std::string ReadVenvBasePython(const std::filesystem::path& venv_root) {
    // Read the 'home' field from pyvenv.cfg to get the base Python installation
    std::filesystem::path cfg_file = venv_root / "pyvenv.cfg";
    if (!std::filesystem::exists(cfg_file)) {
        return "";
    }

    std::ifstream file(cfg_file);
    if (!file.is_open()) {
        return "";
    }

    std::string line;
    while (std::getline(file, line)) {
        // Look for "home = <path>" line
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "home") {
            return value;
        }
    }

    return "";
}

std::filesystem::path ResolvePythonHomeFromInterpreter(const std::filesystem::path& interpreter_path) {
    auto venv_root = ResolveVenvRoot(interpreter_path);
    if (!venv_root.empty()) {
        return venv_root;
    }
    return interpreter_path.parent_path();
}

std::filesystem::path FindPosixSitePackages(const std::filesystem::path& base_root) {
#ifndef _WIN32
    std::filesystem::path lib_dir = base_root / "lib";
    if (!std::filesystem::exists(lib_dir)) {
        return {};
    }

    for (const auto& entry : std::filesystem::directory_iterator(lib_dir)) {
        if (!entry.is_directory()) {
            continue;
        }
        auto name = entry.path().filename().string();
        if (name.rfind("python", 0) != 0) {
            continue;
        }
        std::filesystem::path candidate = entry.path() / "site-packages";
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
#endif
    return {};
}

std::string DeriveSitePackagesFromHome(const std::filesystem::path& python_home) {
#ifdef _WIN32
    std::filesystem::path site_packages = python_home / "Lib" / "site-packages";
    if (std::filesystem::exists(site_packages)) {
        return site_packages.string();
    }
#else
    std::filesystem::path site_packages = FindPosixSitePackages(python_home);
    if (site_packages.empty()) {
        site_packages = python_home / "lib" / "python3" / "site-packages";
    }
    if (std::filesystem::exists(site_packages)) {
        return site_packages.string();
    }
#endif
    return "";
}

std::filesystem::path NormalizePath(const std::filesystem::path& path) {
    try {
        return std::filesystem::absolute(path).lexically_normal();
    } catch (...) {
        return path.lexically_normal();
    }
}

bool IsPathUnderRoot(const std::filesystem::path& path, const std::filesystem::path& root) {
    if (root.empty()) {
        return false;
    }
    auto norm_path = NormalizePath(path);
    auto norm_root = NormalizePath(root);

#ifdef _WIN32
    std::string path_str = norm_path.string();
    std::string root_str = norm_root.string();
    std::transform(path_str.begin(), path_str.end(), path_str.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(root_str.begin(), root_str.end(), root_str.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (root_str.back() != '\\' && root_str.back() != '/') {
        root_str += "\\";
    }
    return path_str.rfind(root_str, 0) == 0;
#else
    auto pit = norm_path.begin();
    auto rit = norm_root.begin();
    for (; rit != norm_root.end(); ++rit, ++pit) {
        if (pit == norm_path.end() || *pit != *rit) {
            return false;
        }
    }
    return true;
#endif
}

bool IsSamePath(const std::filesystem::path& a, const std::filesystem::path& b) {
    if (a.empty() || b.empty()) {
        return false;
    }
    auto na = NormalizePath(a);
    auto nb = NormalizePath(b);
#ifdef _WIN32
    std::string sa = na.string();
    std::string sb = nb.string();
    std::transform(sa.begin(), sa.end(), sa.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(sb.begin(), sb.end(), sb.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return sa == sb;
#else
    return na == nb;
#endif
}

std::string TrimWhitespace(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::optional<std::filesystem::path> ReadVenvBaseHome(const std::filesystem::path& venv_root) {
    auto cfg_path = venv_root / "pyvenv.cfg";
    if (!std::filesystem::exists(cfg_path)) {
        return std::nullopt;
    }
    std::ifstream file(cfg_path.string());
    if (!file.is_open()) {
        return std::nullopt;
    }
    std::string line;
    while (std::getline(file, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = TrimWhitespace(line.substr(0, pos));
        if (key != "home") {
            continue;
        }
        std::string value = TrimWhitespace(line.substr(pos + 1));
        if (value.empty()) {
            return std::nullopt;
        }
        std::filesystem::path home_path(value);
        if (home_path.is_relative()) {
            home_path = venv_root / home_path;
        }
        return home_path;
    }
    return std::nullopt;
}

std::filesystem::path GetExecutableDir() {
#ifdef _WIN32
    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        return {};
    }
    return std::filesystem::path(path).parent_path();
#elif defined(__APPLE__)
    char exec_path[PATH_MAX];
    uint32_t size = sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &size) != 0) {
        return {};
    }
    return std::filesystem::path(exec_path).parent_path();
#elif defined(__linux__)
    char exec_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
    if (len <= 0) {
        return {};
    }
    exec_path[len] = '\0';
    return std::filesystem::path(exec_path).parent_path();
#else
    return {};
#endif
}

bool HasPyCyxWizModule(const std::filesystem::path& dir) {
    if (dir.empty() || !std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return false;
    }
#ifdef _WIN32
    std::filesystem::path candidate = dir / "pycyxwiz.pyd";
    return std::filesystem::exists(candidate);
#else
    // Python extension modules are typically .so on Linux/macOS
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto name = entry.path().filename().string();
        if (name.rfind("pycyxwiz.", 0) == 0 && entry.path().extension() == ".so") {
            return true;
        }
    }
    return false;
#endif
}

uint16_t ReadLe16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t ReadLe32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

bool ZipContainsFile(const std::filesystem::path& zip_path, const std::string& filename) {
    std::ifstream file(zip_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamoff size = file.tellg();
    if (size < 22) {
        return false;
    }

    const size_t max_back = 0xFFFF + 22;
    size_t read_size = static_cast<size_t>(std::min<std::streamoff>(size, static_cast<std::streamoff>(max_back)));
    file.seekg(size - static_cast<std::streamoff>(read_size));

    std::vector<uint8_t> buffer(read_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
        return false;
    }

    const uint32_t eocd_sig = 0x06054b50;
    size_t eocd_pos = std::string::npos;
    for (size_t i = buffer.size() - 4; i-- > 0;) {
        if (ReadLe32(buffer.data() + i) == eocd_sig) {
            eocd_pos = i;
            break;
        }
    }
    if (eocd_pos == std::string::npos || eocd_pos + 22 > buffer.size()) {
        return false;
    }

    const uint8_t* eocd = buffer.data() + eocd_pos;
    uint32_t cd_size = ReadLe32(eocd + 12);
    uint32_t cd_offset = ReadLe32(eocd + 16);
    if (cd_size == 0) {
        return false;
    }

    file.seekg(cd_offset, std::ios::beg);
    std::vector<uint8_t> cd(cd_size);
    if (!file.read(reinterpret_cast<char*>(cd.data()), static_cast<std::streamsize>(cd.size()))) {
        return false;
    }

    const uint32_t cd_sig = 0x02014b50;
    size_t pos = 0;
    auto matches = [&filename](const std::string& name) {
        if (name == filename) {
            return true;
        }
        if (name.size() > filename.size() + 1 &&
            name.compare(name.size() - filename.size(), filename.size(), filename) == 0 &&
            name[name.size() - filename.size() - 1] == '/') {
            return true;
        }
        return false;
    };

    while (pos + 46 <= cd.size()) {
        if (ReadLe32(cd.data() + pos) != cd_sig) {
            break;
        }
        uint16_t name_len = ReadLe16(cd.data() + pos + 28);
        uint16_t extra_len = ReadLe16(cd.data() + pos + 30);
        uint16_t comment_len = ReadLe16(cd.data() + pos + 32);
        size_t name_pos = pos + 46;
        if (name_pos + name_len > cd.size()) {
            break;
        }
        std::string name(reinterpret_cast<const char*>(cd.data() + name_pos), name_len);
        if (matches(name)) {
            return true;
        }
        pos = name_pos + name_len + extra_len + comment_len;
    }

    return false;
}

bool HasStdLibInHome(const std::filesystem::path& python_home, std::string* detail_out) {
    if (python_home.empty()) {
        return false;
    }

    std::filesystem::path stdlib_path = python_home / "Lib" / "threading.py";
    if (std::filesystem::exists(stdlib_path)) {
        return true;
    }

    // Check stdlib zip (embeddable dist)
    if (std::filesystem::exists(python_home) && std::filesystem::is_directory(python_home)) {
        for (const auto& entry : std::filesystem::directory_iterator(python_home)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            auto name = entry.path().filename().string();
            if (name.rfind("python", 0) != 0 || entry.path().extension() != ".zip") {
                continue;
            }
            if (ZipContainsFile(entry.path(), "threading.py")) {
                return true;
            }
        }
    }

    if (detail_out) {
        *detail_out = "missing stdlib (expected '" + (python_home / "Lib" / "threading.py").string() +
                      "' or threading.py inside python*.zip under " + python_home.string() + ")";
    }
    return false;
}

std::string ReadProjectInterpreterOverride() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        return "";
    }

    std::filesystem::path env_path = std::filesystem::path(pm.GetProjectRoot()) / "python_env.json";
    if (!std::filesystem::exists(env_path)) {
        return "";
    }

    try {
        std::ifstream file(env_path);
        if (!file.is_open()) {
            spdlog::warn("Failed to open python_env.json: {}", env_path.string());
            return "";
        }
        nlohmann::json j;
        file >> j;

        if (!j.contains("python") || !j["python"].is_object()) {
            return "";
        }
        const auto& python = j["python"];
        if (!python.contains("interpreter_path") || !python["interpreter_path"].is_string()) {
            return "";
        }

        std::string interpreter = python["interpreter_path"].get<std::string>();
        if (interpreter.empty()) {
            return "";
        }

        std::filesystem::path interp_path(interpreter);
        if (interp_path.is_relative()) {
            interp_path = std::filesystem::path(pm.GetProjectRoot()) / interp_path;
        }
        return interp_path.string();
    } catch (const std::exception& e) {
        spdlog::warn("Failed to parse python_env.json: {}", e.what());
        return "";
    }
}

std::string DeriveSitePackagesPath(const std::string& interpreter_path) {
    if (interpreter_path.empty()) {
        return "";
    }
    std::filesystem::path interp(interpreter_path);
    std::filesystem::path parent = interp.parent_path();
#ifdef _WIN32
    std::filesystem::path site_packages;
    auto venv_root = ResolveVenvRoot(interp);
    if (!venv_root.empty()) {
        site_packages = venv_root / "Lib" / "site-packages";
    } else {
        site_packages = parent / "Lib" / "site-packages";
    }
#else
    std::filesystem::path base_root = ResolveVenvRoot(interp);
    if (base_root.empty()) {
        base_root = parent.parent_path();
    }
    std::filesystem::path site_packages = FindPosixSitePackages(base_root);
    if (site_packages.empty()) {
        site_packages = base_root / "lib" / "python3" / "site-packages";
    }
#endif
    if (std::filesystem::exists(site_packages)) {
        return site_packages.string();
    }
    return "";
}

} // namespace

PythonEngine::PythonEngine() : initialized_(false), main_thread_state_(nullptr) {
}

PythonEngine::~PythonEngine() {
    Shutdown();
}

bool PythonEngine::Initialize(std::string* error_out) {
    try {
        if (initialized_) {
            return true;
        }

        // Hard error if Python was initialized before our config was applied
        if (Py_IsInitialized()) {
            const char* msg = "Python already initialized before engine configuration; restart required";
            spdlog::error(msg);
            if (error_out) {
                *error_out = msg;
            }
            return false;
        }

        PythonSelection selection = ResolvePythonConfig();
        std::string validation_error;
        if (!ValidateSelection(selection, &validation_error)) {
            spdlog::error("Python selection invalid: {}", validation_error);
            if (error_out) {
                *error_out = validation_error;
            }
            return false;
        }

        // Log initialization details
        if (selection.source == "project") {
            spdlog::info("→ Initializing Python with project environment...");
        } else {
            spdlog::info("→ Initializing Python interpreter...");
        }

        ApplySelection(selection);

        // Configure PYTHONHOME BEFORE initializing Python
        // This determines which Python installation is used
        std::string env_error;
        if (!ConfigurePythonHome(&env_error)) {
            spdlog::error("Failed to configure Python environment: {}", env_error);
            if (error_out) {
                *error_out = env_error;
            }
            return false;
        }

        py::initialize_interpreter();
        initialized_ = true;
        initialized_by_us_ = true;  // We initialized it, so we'll finalize it

        // Record the active selection (runtime is now locked)
        active_interpreter_path_ = effective_interpreter_path_;
        active_source_ = selection.source;

        // Configure additional Python paths (site-packages, etc.)
        ConfigureCustomPythonPath();

        // Release the GIL so background threads can use Python
        // This is REQUIRED for multi-threaded Python execution
        ReleaseGIL();

        // Final success message
        if (selection.source == "project") {
            spdlog::info("✓ Python environment ready (project venv)");
        } else {
            spdlog::info("✓ Python interpreter ready (system Python)");
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize Python: {}", e.what());
        if (error_out) {
            *error_out = e.what();
        }
        return false;
    }
}

bool PythonEngine::ConfigurePythonHome(std::string* error_out) {
    // Configure Python environment before Py_Initialize()
    // This MUST be called BEFORE py::initialize_interpreter()

    // Clear any existing PYTHONHOME/PYTHONPATH to avoid leakage
    // Also disable user site-packages to avoid conflicts between Python versions
#ifdef _WIN32
    _putenv_s("PYTHONHOME", "");
    _putenv_s("PYTHONPATH", "");
    _putenv_s("PYTHONNOUSERSITE", "1");  // Disable user site-packages
#else
    setenv("PYTHONHOME", "", 1);
    setenv("PYTHONPATH", "", 1);
    setenv("PYTHONNOUSERSITE", "1", 1);  // Disable user site-packages
#endif

    // If no interpreter configured, use system Python
    if (effective_interpreter_path_.empty()) {
        spdlog::info("Using system Python (no PYTHONHOME set)");
        return true;
    }

    // Check interpreter exists
    if (!std::filesystem::exists(effective_interpreter_path_)) {
        std::string msg = "Interpreter does not exist: " + effective_interpreter_path_;
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }

    // Determine if this is a venv interpreter
    std::filesystem::path interp_path(effective_interpreter_path_);
    auto venv_root = ResolveVenvRoot(interp_path);

    if (!venv_root.empty()) {
        // This is a venv - let pyvenv.cfg drive sys.path
        // Setting PYTHONHOME can break stdlib resolution in venvs
        spdlog::info("  Environment: Virtual environment (venv)");
        spdlog::info("  Location: {}", venv_root.string());
        return true;
    }

    // This is a system Python - set PYTHONHOME
    std::string python_home = ResolvePythonHomeFromInterpreter(interp_path).string();

#ifdef _WIN32
    if (_putenv_s("PYTHONHOME", python_home.c_str()) != 0) {
        spdlog::error("Failed to set PYTHONHOME environment variable");
        if (error_out) {
            *error_out = "Failed to set PYTHONHOME environment variable";
        }
        return false;
    }
#else
    if (setenv("PYTHONHOME", python_home.c_str(), 1) != 0) {
        spdlog::error("Failed to set PYTHONHOME environment variable");
        if (error_out) {
            *error_out = "Failed to set PYTHONHOME environment variable";
        }
        return false;
    }
#endif

    spdlog::info("Using system Python: {}", python_home);
    return true;
}

void PythonEngine::ConfigureCustomPythonPath() {
    // Configure additional paths after Python is initialized
    // This adds custom site-packages to sys.path if configured

    if (!Py_IsInitialized()) {
        return;
    }

    std::string python_home;
    if (!effective_interpreter_path_.empty()) {
        python_home = ResolvePythonHomeFromInterpreter(std::filesystem::path(effective_interpreter_path_)).string();
    }

    std::string packages_dir;
    if (!effective_interpreter_path_.empty()) {
        packages_dir = DeriveSitePackagesPath(effective_interpreter_path_);
        if (packages_dir.empty()) {
            spdlog::warn("Python interpreter site-packages not found");
        }
    }

    std::string scripts_path;
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (pm.HasActiveProject()) {
        scripts_path = pm.GetScriptsPath();
    }

    std::string engine_module_dir;
    auto exe_dir = GetExecutableDir();
    if (!exe_dir.empty() && HasPyCyxWizModule(exe_dir)) {
        engine_module_dir = exe_dir.string();
    }

    try {
        py::gil_scoped_acquire acquire;
        py::object sys = py::module_::import("sys");
        py::list sys_path = sys.attr("path").cast<py::list>();
        py::object version_info = sys.attr("version_info");
        int major = py::int_(version_info.attr("major"));
        int minor = py::int_(version_info.attr("minor"));

        // Get base_prefix - for venvs, read from pyvenv.cfg instead of sys.base_prefix
        // (pybind11 embedded interpreter sets sys.base_prefix incorrectly)
        std::string base_prefix;
        std::filesystem::path venv_root = !python_home.empty() ? ResolveVenvRoot(std::filesystem::path(effective_interpreter_path_)) : std::filesystem::path{};
        if (!venv_root.empty()) {
            // This is a venv - read base Python from pyvenv.cfg
            base_prefix = ReadVenvBasePython(venv_root);
            if (base_prefix.empty()) {
                spdlog::warn("Failed to read base Python from pyvenv.cfg, falling back to sys.base_prefix");
                base_prefix = py::str(sys.attr("base_prefix")).cast<std::string>();
            }
        } else {
            // Not a venv - use sys.base_prefix
            base_prefix = py::str(sys.attr("base_prefix")).cast<std::string>();
        }


        std::vector<std::string> new_paths;
        auto add_unique = [&new_paths](const std::string& path) {
            if (path.empty()) {
                return;
            }
            if (std::find(new_paths.begin(), new_paths.end(), path) == new_paths.end()) {
                new_paths.push_back(path);
            }
        };

        if (!packages_dir.empty()) {
            add_unique(packages_dir);
        }
        if (!scripts_path.empty()) {
            add_unique(scripts_path);
        }
        if (!engine_module_dir.empty()) {
            add_unique(engine_module_dir);
        }

        std::vector<std::filesystem::path> allowed_roots;
        if (!python_home.empty()) {
            // Using a specific Python home (venv or custom interpreter)
            allowed_roots.push_back(std::filesystem::path(python_home));
            // For venvs, also add the base_prefix to find standard library
            if (!base_prefix.empty() && base_prefix != python_home) {
                allowed_roots.push_back(std::filesystem::path(base_prefix));
            }
        } else if (!base_prefix.empty()) {
            // No custom Python home - use system Python's base_prefix
            allowed_roots.push_back(std::filesystem::path(base_prefix));
        }

        // Ensure stdlib paths are present for each allowed root
        for (const auto& root : allowed_roots) {
#ifdef _WIN32
            std::filesystem::path stdlib = root / "Lib";
#else
            std::filesystem::path stdlib = root / "lib" / ("python" + std::to_string(major) + "." + std::to_string(minor));
#endif
            if (std::filesystem::exists(stdlib)) {
                add_unique(stdlib.string());
            }
        }

        if (!allowed_roots.empty()) {
            for (auto item : sys_path) {
                std::string entry = py::str(item).cast<std::string>();
                if (entry.empty()) {
                    continue;
                }
                std::filesystem::path entry_path(entry);
                bool keep = false;
                for (const auto& root : allowed_roots) {
                    if (IsPathUnderRoot(entry_path, root)) {
                        keep = true;
                        break;
                    }
                }
                if (keep) {
                    add_unique(entry);
                }
            }
        } else {
            for (auto item : sys_path) {
                std::string entry = py::str(item).cast<std::string>();
                add_unique(entry);
            }
        }

        py::list updated;
        for (const auto& path : new_paths) {
            updated.append(path);
        }
        sys.attr("path") = updated;

        // Log configured paths in a user-friendly way
        if (!packages_dir.empty()) {
            spdlog::info("  Site-packages: {}", packages_dir);
        }
        if (!scripts_path.empty()) {
            spdlog::info("  Project scripts: {}", scripts_path);
        }
        if (!engine_module_dir.empty()) {
            spdlog::info("  Engine modules: {}", engine_module_dir);
        }
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to configure Python path: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::error("Error configuring Python path: {}", e.what());
    }
}

PythonEngine::PythonSelection PythonEngine::ResolvePythonConfig() const {
    auto& config = cyxwiz::core::EngineConfig::Instance();
    PythonSelection selection;

    // Priority 1: Check for project venv
    std::string project_venv = ReadProjectInterpreterOverride();
    if (!project_venv.empty()) {
        if (std::filesystem::exists(project_venv)) {
            selection.interpreter_path = project_venv;
            selection.source = "project";

            // Get project name for better logging
            auto& pm = cyxwiz::ProjectManager::Instance();
            std::string project_name = pm.HasActiveProject() ? pm.GetProjectName() : "unknown";

            spdlog::info("✓ Using project virtual environment");
            spdlog::info("  Project: {}", project_name);
            spdlog::info("  Python: {}", project_venv);
            return selection;
        } else {
            spdlog::warn("Project interpreter '{}' does not exist, falling back to system Python", project_venv);
        }
    }

    // Priority 2: Use system Python from EngineConfig
    selection.interpreter_path = config.GetSystemPythonPath();
    selection.source = "system";

    if (selection.interpreter_path.empty()) {
        spdlog::warn("⚠ No Python configured - Python features will be unavailable");
    } else {
        spdlog::info("✓ Using system Python interpreter: {}", selection.interpreter_path);
    }

    return selection;
}

bool PythonEngine::ValidateSelection(const PythonSelection& selection, std::string* error_out) const {
    // No Python configured
    if (selection.interpreter_path.empty()) {
        std::string msg = "No Python interpreter configured. Please configure system Python in settings.";
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }

    // Check if interpreter exists
    if (!std::filesystem::exists(selection.interpreter_path)) {
        std::string msg = "Python interpreter not found: " + selection.interpreter_path;
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }

    // Validate interpreter has standard library
    std::filesystem::path interp_path(selection.interpreter_path);
    std::filesystem::path venv_root = ResolveVenvRoot(interp_path);

    if (!venv_root.empty()) {
        // This is a venv - check the base Python has stdlib
        auto base_home = ReadVenvBaseHome(venv_root);
        if (!base_home.has_value()) {
            std::string msg = "Venv pyvenv.cfg missing 'home' entry; cannot validate standard library.";
            if (error_out) {
                *error_out = msg;
            }
            return false;
        }
        std::string stdlib_error;
        if (!HasStdLibInHome(*base_home, &stdlib_error)) {
            std::string msg = "Venv base Python is missing standard library: " + stdlib_error +
                ". Install a full Python distribution or update the interpreter path.";
            if (error_out) {
                *error_out = msg;
            }
            return false;
        }
    } else {
        // This is a system Python - check it has stdlib
        std::string stdlib_error;
        std::filesystem::path home = ResolvePythonHomeFromInterpreter(interp_path);
        if (!HasStdLibInHome(home, &stdlib_error)) {
            std::string msg = "Python interpreter is missing standard library: " + stdlib_error +
                ". Install a full Python distribution or update the interpreter path.";
            if (error_out) {
                *error_out = msg;
            }
            return false;
        }
    }

    return true;
}

void PythonEngine::ApplySelection(const PythonSelection& selection) {
    effective_interpreter_path_ = selection.interpreter_path;
}

bool PythonEngine::HasInterpreterMismatch(std::string* reason_out) const {
    if (!initialized_) {
        return false;
    }

    PythonSelection desired = ResolvePythonConfig();

    auto normalize = [](const std::string& path) {
        std::filesystem::path p(path);
        p = NormalizePath(p);
        std::string s = p.string();
#ifdef _WIN32
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
        return s;
    };

    std::string desired_path = normalize(desired.interpreter_path);
    std::string active_path = normalize(active_interpreter_path_);

    if (desired_path == active_path) {
        return false;
    }

    if (reason_out) {
        std::string desired_desc = desired.interpreter_path.empty() ? "system" : desired.interpreter_path;
        std::string active_desc = active_interpreter_path_.empty() ? "system" : active_interpreter_path_;
        *reason_out = "desired='" + desired_desc + "' active='" + active_desc + "'";
    }
    return true;
}

bool PythonEngine::ReloadForProject(std::string* error_out) {
    if (!initialized_) {
        // Lazy initialization: do not initialize here. Initialization happens on first Python use.
        return true;
    }

    std::string mismatch;
    if (HasInterpreterMismatch(&mismatch)) {
        std::string msg = "Python interpreter mismatch; restart required (" + mismatch + ")";
        spdlog::warn("{}", msg);
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }
    return true;
}

void PythonEngine::Shutdown() {
    if (initialized_ && initialized_by_us_) {
        // Skip Python finalization during shutdown.
        // py::finalize_interpreter() can cause crashes if there are still
        // pybind11 objects (lambdas, callbacks) that need to be cleaned up.
        // The OS will clean up Python when the process exits.
        //
        // Note: This is safe because we're only called during process shutdown.
        // If we needed to reinitialize Python, we would need to finalize properly.
        spdlog::info("Python interpreter shutdown (not finalized - OS will cleanup)");
        initialized_ = false;
        main_thread_state_ = nullptr;
    } else if (initialized_) {
        // We're using a shared interpreter, just mark as not initialized
        initialized_ = false;
    }
}

void PythonEngine::ReleaseGIL() {
    if (main_thread_state_ == nullptr) {
        // Save the current thread state and release the GIL
        main_thread_state_ = PyEval_SaveThread();
    }
}

void PythonEngine::AcquireGIL() {
    if (main_thread_state_ != nullptr) {
        // Restore the thread state and reacquire the GIL
        PyEval_RestoreThread(main_thread_state_);
        main_thread_state_ = nullptr;
    }
}

bool PythonEngine::ExecuteScript(const std::string& script) {
    if (!initialized_) return false;

    try {
        // Must acquire GIL since we released it after initialization
        py::gil_scoped_acquire acquire;
        py::exec(script);
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::error("Python error: {}", e.what());
        return false;
    }
}

bool PythonEngine::ExecuteFile(const std::string& filepath) {
    if (!initialized_) return false;

    try {
        // Must acquire GIL since we released it after initialization
        py::gil_scoped_acquire acquire;
        py::eval_file(filepath);
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::error("Python error: {}", e.what());
        return false;
    }
}

} // namespace scripting

#endif // CYXWIZ_HAS_PYTHON
