// An installed Engine's Python, and every Python it starts, keeps compiled
// bytecode in the user's cache folder (PYTHONPYCACHEPREFIX, set by main.cpp),
// never as __pycache__ beside the imported .py files: the bundled Python in
// the base is read-only and inventoried.
#include "../src/core/user_folders.h"

#include <pybind11/embed.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifndef CYXWIZ_TEST_PYTHONHOME
#error "CYXWIZ_TEST_PYTHONHOME must name the Python the build uses"
#endif

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++failures;
}

bool HasBytecodeOf(const fs::path& folder, const std::string& module) {
    std::error_code ec;
    for (fs::recursive_directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (it->path().extension() == ".pyc" && name.rfind(module + ".", 0) == 0) return true;
    }
    return false;
}

}  // namespace

int main() {
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const fs::path work = fs::temp_directory_path() / ("cyxwiz-pycache-" + stamp);
    const fs::path modules = work / "modules";
    const fs::path child_modules = work / "child_modules";
    fs::create_directories(modules);
    fs::create_directories(child_modules);
    std::ofstream(modules / "cyxwiz_pycache_probe.py") << "VALUE = 'probe'\n";
    std::ofstream(child_modules / "cyxwiz_child_probe.py") << "VALUE = 'child'\n";
    std::ofstream(work / "child.py") << "import sys\nsys.path.insert(0, sys.argv[1])\nimport cyxwiz_child_probe\n";

#ifdef _WIN32
    _putenv_s("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME);
    _putenv_s("PYTHONDONTWRITEBYTECODE", "");
    const fs::path python = fs::path(CYXWIZ_TEST_PYTHONHOME) / "python.exe";
#else
    setenv("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME, 1);
    unsetenv("PYTHONDONTWRITEBYTECODE");
    const fs::path python = fs::path(CYXWIZ_TEST_PYTHONHOME) / "python3";
#endif
    {
        cyxwiz::ScopedUserFoldersOverrideForTesting folders(work / "profile");
        const fs::path prefix = cyxwiz::PythonBytecodeDir();
        Check(!prefix.empty() && prefix.string().rfind((work / "profile").string(), 0) == 0,
              "the bytecode folder is in the user's cache folder: " + prefix.string());
        Check(cyxwiz::SetPythonBytecodeEnvironment(prefix), "the bytecode folder is set in the environment");

        // The embedded interpreter, started as the Engine starts it.
        py::initialize_interpreter();
        {
            py::module_ sys = py::module_::import("sys");
            const py::object configured = sys.attr("pycache_prefix");
            Check(!configured.is_none() && fs::path(configured.cast<std::string>()) == prefix,
                  "sys.pycache_prefix is the bytecode folder");
            Check(sys.attr("dont_write_bytecode").cast<bool>() == false, "bytecode is still written (faster starts)");
            sys.attr("path").attr("insert")(0, modules.string());
            const std::string value = py::module_::import("cyxwiz_pycache_probe").attr("VALUE").cast<std::string>();
            Check(value == "probe", "the module imports");
        }
        py::finalize_interpreter();
        Check(!fs::exists(modules / "__pycache__"), "no __pycache__ beside the module the Engine imported");
        Check(HasBytecodeOf(prefix, "cyxwiz_pycache_probe"), "its bytecode is in the bytecode folder");

        // A Python the Engine starts (the Python check, venv, pip) inherits it.
        std::string command = "\"" + python.string() + "\" \"" + (work / "child.py").string() + "\" \"" +
                              child_modules.string() + "\"";
#ifdef _WIN32
        command = "\"" + command + "\"";  // cmd /c drops the outer quotes
#endif
        Check(std::system(command.c_str()) == 0, "a child Python runs: " + python.string());
        Check(!fs::exists(child_modules / "__pycache__"), "no __pycache__ beside the module a child Python imported");
        Check(HasBytecodeOf(prefix, "cyxwiz_child_probe"), "the child's bytecode is in the bytecode folder");
    }

    std::error_code ec;
    fs::remove_all(work, ec);
    if (failures) {
        std::cout << failures << " Python bytecode check(s) failed\n";
        return 1;
    }
    std::cout << "Python bytecode checks passed\n";
    return 0;
}
