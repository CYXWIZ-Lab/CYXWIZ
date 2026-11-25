// The Engine's Python keeps compiled bytecode in the user's cache folder
// (sys.pycache_prefix), never as __pycache__ beside the imported .py files:
// an installed base is read-only and inventoried.
#include "../src/core/user_folders.h"
#include "../src/scripting/python_environment.h"

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

bool HasBytecode(const fs::path& folder) {
    std::error_code ec;
    for (fs::recursive_directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() == ".pyc") return true;
    }
    return false;
}

}  // namespace

int main() {
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const fs::path work = fs::temp_directory_path() / ("cyxwiz-pycache-" + stamp);
    const fs::path modules = work / "modules";
    fs::create_directories(modules);
    std::ofstream(modules / "cyxwiz_pycache_probe.py") << "VALUE = 'probe'\n";

#ifdef _WIN32
    _putenv_s("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME);
    _putenv_s("PYTHONDONTWRITEBYTECODE", "");
#else
    setenv("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME, 1);
    unsetenv("PYTHONDONTWRITEBYTECODE");
#endif
    {
        cyxwiz::ScopedUserFoldersOverrideForTesting folders(work / "profile");
        const fs::path prefix = scripting::PythonBytecodeFolder();
        Check(!prefix.empty() && prefix.string().rfind((work / "profile").string(), 0) == 0,
              "the bytecode folder is in the user's cache folder: " + prefix.string());

        scripting::StartEmbeddedInterpreter();
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

        Check(!fs::exists(modules / "__pycache__"), "no __pycache__ beside the imported module");
        Check(HasBytecode(prefix), "the module's bytecode is in the bytecode folder");
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
