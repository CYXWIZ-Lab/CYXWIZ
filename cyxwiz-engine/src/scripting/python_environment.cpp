#include "python_environment.h"

#include <algorithm>
#include <cctype>
#include <fstream>

#ifdef CYXWIZ_HAS_PYTHON
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

namespace scripting {

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


std::filesystem::path ResolvePythonHomeFromInterpreter(const std::filesystem::path& interpreter_path) {
    auto venv_root = ResolveVenvRoot(interpreter_path);
    if (!venv_root.empty()) {
        return venv_root;
    }
    auto parent = interpreter_path.parent_path();
#ifndef _WIN32
    // POSIX layouts keep the interpreter in <prefix>/bin.
    if (parent.filename() == "bin") {
        return parent.parent_path();
    }
#endif
    return parent;
}

std::filesystem::path FindPosixSitePackages([[maybe_unused]] const std::filesystem::path& base_root) {
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

std::optional<std::string> ReadPyvenvValue(const std::filesystem::path& venv_root, const std::string& key) {
    std::ifstream file(venv_root / "pyvenv.cfg");
    if (!file.is_open()) {
        return std::nullopt;
    }
    std::string line;
    while (std::getline(file, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos || TrimWhitespace(line.substr(0, pos)) != key) {
            continue;
        }
        std::string value = TrimWhitespace(line.substr(pos + 1));
        if (value.empty()) {
            return std::nullopt;
        }
        return value;
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> ReadVenvBaseHome(const std::filesystem::path& venv_root) {
    auto value = ReadPyvenvValue(venv_root, "home");
    if (!value) {
        return std::nullopt;
    }
    std::filesystem::path home_path(*value);
    if (home_path.is_relative()) {
        home_path = venv_root / home_path;
    }
    return home_path;
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

PythonEnvironment DescribePythonEnvironment(const std::filesystem::path& interpreter) {
    PythonEnvironment env;
    if (interpreter.empty()) {
        return env;
    }
    env.interpreter = interpreter;
    env.venv_root = ResolveVenvRoot(interpreter);
    if (!env.venv_root.empty()) {
        env.home = ReadVenvBaseHome(env.venv_root).value_or(std::filesystem::path{});
        env.version = ReadPyvenvValue(env.venv_root, "version").value_or("");
    } else {
        env.home = ResolvePythonHomeFromInterpreter(interpreter);
    }
    env.site_packages = DeriveSitePackagesPath(interpreter.string());
    return env;
}

std::string InPlaceSwitchBlocker(const PythonEnvironment& from, const PythonEnvironment& to,
                                 int running_major, int running_minor) {
    if (from.home.empty() || to.home.empty()) {
        return "the Python installation of the running or the new environment is unknown";
    }
    if (!IsSamePath(from.home, to.home)) {
        return "the project's environment runs on " + to.home.string() + ", the Engine's Python on " +
               from.home.string();
    }
    if (!to.version.empty()) {
        const std::string running = std::to_string(running_major) + "." + std::to_string(running_minor);
        const bool same = to.version == running || to.version.rfind(running + ".", 0) == 0;
        if (!same) {
            return "the project's environment is Python " + to.version + ", the Engine runs Python " + running;
        }
    }
    return {};
}

#ifdef CYXWIZ_HAS_PYTHON

namespace {

// Runs in the interpreter (see SwitchEnvironmentInPlace).
constexpr const char* kSwitchCode = R"PY(
import importlib, os, site, sys

def _norm(path):
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError):
        return None

def _under(path, roots):
    p = _norm(path) if path else None
    return p is not None and any(p == r or p.startswith(r + os.sep) for r in roots)

def switch(old_dirs, new_prefix, new_site, old_scripts, new_scripts):
    roots = [r for r in (_norm(d) for d in list(old_dirs) + [old_scripts] if d) if r]
    loaded, compiled = [], []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None) or ""
        places = [file] + list(getattr(module, "__path__", None) or [])
        if not any(_under(p, roots) for p in places):
            continue
        loaded.append(name)
        if file.lower().endswith((".pyd", ".so")):
            compiled.append(name)
    if compiled:
        return sorted(compiled)
    for name in loaded:
        del sys.modules[name]
    gone = set(loaded)
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ not in gone]
    kept = [p for p in sys.path if not _under(p, roots)]
    front = [p for p in (new_site, new_scripts) if p and p not in kept]
    sys.path[:] = front + kept
    if new_site:
        site.addsitedir(new_site)
    sys.prefix = sys.exec_prefix = new_prefix
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()
    return []
)PY";

} // namespace

std::vector<std::string> SwitchEnvironmentInPlace(const PythonEnvironment& from, const PythonEnvironment& to,
                                                  const std::string& old_scripts, const std::string& new_scripts) {
    // What belongs to the old environment: the whole venv, or a base
    // installation's site-packages only (its standard library stays).
    std::vector<std::string> old_dirs;
    if (!from.venv_root.empty()) {
        old_dirs.push_back(from.venv_root.string());
    } else if (!from.site_packages.empty()) {
        old_dirs.push_back(from.site_packages.string());
    }
    const std::filesystem::path new_prefix = to.venv_root.empty() ? to.home : to.venv_root;

    py::dict scope;
    py::exec(kSwitchCode, scope);
    py::object result = scope["switch"](old_dirs, new_prefix.string(), to.site_packages.string(), old_scripts,
                                        new_scripts);
    return result.cast<std::vector<std::string>>();
}

#endif // CYXWIZ_HAS_PYTHON

} // namespace scripting
