// A running interpreter moves to another project's environment (TOFIX125
// walk): two venvs on the Python the build uses, each with its own
// site-packages. Without Python: what each environment is and when a move
// needs a restart. With Python: the move forgets the old environment's
// modules, puts the new site-packages (and its .pth paths) on sys.path and
// names the new prefix; a compiled module of the old environment stops it.
#include "../src/scripting/python_environment.h"

#include <pybind11/embed.h>

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

void WriteFile(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

std::string RunningVersion() {
    return std::to_string(PY_MAJOR_VERSION) + "." + std::to_string(PY_MINOR_VERSION) + ".0";
}

// <root>/Scripts/python.exe (bin/python elsewhere) and pyvenv.cfg.
fs::path MakeVenv(const fs::path& root, const fs::path& home, const std::string& version) {
    WriteFile(root / "pyvenv.cfg", "home = " + home.string() + "\ninclude-system-site-packages = false\nversion = " +
                                       version + "\n");
#ifdef _WIN32
    const fs::path interpreter = root / "Scripts" / "python.exe";
    fs::create_directories(root / "Lib" / "site-packages");
#else
    const fs::path interpreter = root / "bin" / "python";
    fs::create_directories(root / "lib" / ("python" + version.substr(0, version.rfind('.'))) / "site-packages");
#endif
    WriteFile(interpreter, "");
    return interpreter;
}

std::string ImportValue(const char* module) {
    return py::module_::import(module).attr("VALUE").cast<std::string>();
}

bool OnSysPath(const fs::path& dir) {
    for (auto item : py::module_::import("sys").attr("path")) {
        if (scripting::IsSamePath(fs::path(item.cast<std::string>()), dir)) return true;
    }
    return false;
}

bool Loaded(const char* module) {
    return py::module_::import("sys").attr("modules").contains(module);
}

} // namespace

int main() {
    const fs::path home = fs::path(CYXWIZ_TEST_PYTHONHOME);
    const fs::path scratch = fs::temp_directory_path() / "cyxwiz_test_python_environment_switch";
    std::error_code ec;
    fs::remove_all(scratch, ec);

    const fs::path old_interpreter = MakeVenv(scratch / "old", home, RunningVersion());
    const fs::path new_interpreter = MakeVenv(scratch / "new", home, RunningVersion());
    const auto old_env = scripting::DescribePythonEnvironment(old_interpreter);
    const auto new_env = scripting::DescribePythonEnvironment(new_interpreter);

    std::cout << "what an environment is\n";
    Check(scripting::IsSamePath(old_env.venv_root, scratch / "old"), "the venv root is the folder with pyvenv.cfg");
    Check(scripting::IsSamePath(old_env.home, home), "the venv runs on its pyvenv.cfg home");
    Check(old_env.version == RunningVersion(), "the venv's version comes from pyvenv.cfg");
    Check(!old_env.site_packages.empty(), "the venv's site-packages is found");
#ifdef _WIN32
    const auto base = scripting::DescribePythonEnvironment(home / "python.exe");
    Check(base.venv_root.empty() && scripting::IsSamePath(base.home, home), "a base interpreter is its own home");
#endif

    std::cout << "when a move needs a restart\n";
    Check(scripting::InPlaceSwitchBlocker(old_env, new_env, PY_MAJOR_VERSION, PY_MINOR_VERSION).empty(),
          "two venvs of the same Python: in place");
    const auto other_home =
        scripting::DescribePythonEnvironment(MakeVenv(scratch / "other", scratch / "elsewhere", RunningVersion()));
    const std::string different_home =
        scripting::InPlaceSwitchBlocker(old_env, other_home, PY_MAJOR_VERSION, PY_MINOR_VERSION);
    Check(different_home.find("runs on") != std::string::npos, "another Python installation: restart (" +
                                                                   different_home + ")");
    const auto older = scripting::DescribePythonEnvironment(MakeVenv(scratch / "older", home, "3.1.0"));
    const std::string different_version =
        scripting::InPlaceSwitchBlocker(old_env, older, PY_MAJOR_VERSION, PY_MINOR_VERSION);
    Check(different_version.find("Python 3.1.0") != std::string::npos, "another Python version: restart (" +
                                                                           different_version + ")");
    Check(!scripting::InPlaceSwitchBlocker({}, new_env, PY_MAJOR_VERSION, PY_MINOR_VERSION).empty(),
          "an unknown running environment: restart");

    // The old environment has two modules; the new one its own `shared`, and
    // a .pth file that adds a folder.
    WriteFile(old_env.site_packages / "oldmod.py", "VALUE = 'old'\n");
    WriteFile(old_env.site_packages / "shared.py", "VALUE = 'old'\n");
    WriteFile(new_env.site_packages / "shared.py", "VALUE = 'new'\n");
    WriteFile(scratch / "extra" / "extramod.py", "VALUE = 'extra'\n");
    WriteFile(new_env.site_packages / "extra.pth", (scratch / "extra").string() + "\n");
    WriteFile(scratch / "old_scripts" / "oldscript.py", "VALUE = 'old script'\n");

    std::cout << "the move, with Python running\n";
#ifdef _WIN32
    _putenv_s("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME);
#else
    setenv("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME, 1);
#endif
    py::scoped_interpreter interpreter;
    {
        py::list path = py::module_::import("sys").attr("path");
        path.insert(0, old_env.site_packages.string());
        path.insert(1, (scratch / "old_scripts").string());
    }
    Check(ImportValue("shared") == "old" && ImportValue("oldmod") == "old", "the old environment's modules load");
    Check(ImportValue("oldscript") == "old script", "the old scripts folder loads");

    auto compiled = scripting::SwitchEnvironmentInPlace(old_env, new_env, (scratch / "old_scripts").string(), "");
    Check(compiled.empty(), "pure Python modules do not stop the move");
    Check(!Loaded("shared") && !Loaded("oldmod") && !Loaded("oldscript"),
          "the old environment's and scripts folder's modules are forgotten");
    Check(!OnSysPath(old_env.site_packages) && !OnSysPath(scratch / "old_scripts"),
          "the old site-packages and scripts folder leave sys.path");
    Check(OnSysPath(new_env.site_packages), "the new site-packages is on sys.path");
    Check(ImportValue("shared") == "new", "an import finds the new environment's module");
    Check(ImportValue("extramod") == "extra", "the new environment's .pth paths are on sys.path");
    Check(scripting::IsSamePath(fs::path(py::module_::import("sys").attr("prefix").cast<std::string>()),
                                new_env.venv_root),
          "sys.prefix names the new environment");
    Check(py::module_::import("json").attr("dumps")(1).cast<std::string>() == "1",
          "the standard library still imports");

    std::cout << "a compiled module of the running environment stops the move\n";
    {
        py::module_ types = py::module_::import("types");
        py::object fake = types.attr("ModuleType")("fake_compiled");
        fake.attr("__file__") = (new_env.site_packages / "fake_compiled.cp312-win_amd64.pyd").string();
        py::module_::import("sys").attr("modules")["fake_compiled"] = fake;
    }
    compiled = scripting::SwitchEnvironmentInPlace(new_env, old_env, "", "");
    Check(compiled.size() == 1 && compiled[0] == "fake_compiled", "the compiled module is named");
    Check(Loaded("shared") && OnSysPath(new_env.site_packages) &&
              scripting::IsSamePath(fs::path(py::module_::import("sys").attr("prefix").cast<std::string>()),
                                    new_env.venv_root),
          "nothing changed: modules, sys.path and sys.prefix stay");

    fs::remove_all(scratch, ec);
    std::cout << (failures ? "FAILED: " + std::to_string(failures) : std::string("all passed")) << "\n";
    return failures ? 1 : 0;
}
