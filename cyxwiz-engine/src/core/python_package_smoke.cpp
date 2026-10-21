#include "python_package_smoke.h"

#ifdef CYXWIZ_HAS_PYTHON
#include <Python.h>
#endif

namespace cyxwiz::core {

#ifdef CYXWIZ_HAS_PYTHON
namespace {

std::string FetchString(PyObject* globals, const char* name) {
    PyObject* value = PyDict_GetItemString(globals, name);  // borrowed
    if (value == nullptr || !PyUnicode_Check(value)) {
        return {};
    }
    const char* text = PyUnicode_AsUTF8(value);
    return text ? std::string(text) : std::string{};
}

}  // namespace

std::string RunBundledPythonSmoke(const std::filesystem::path& engine_dir,
                                  std::string* error) {
    const auto home = engine_dir / "python";
    std::error_code ec;
    if (!std::filesystem::is_directory(home, ec)) {
        if (error) *error = "bundled_python_missing";
        return {};
    }

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    PyStatus status = PyConfig_SetString(&config, &config.home, home.wstring().c_str());
    if (!PyStatus_Exception(status)) {
        status = Py_InitializeFromConfig(&config);
    }
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = std::string("python_initialize_failed detail='") +
                     (status.err_msg ? status.err_msg : "unknown") + "'";
        }
        return {};
    }

    PyObject* main_module = PyImport_AddModule("__main__");  // borrowed
    PyObject* globals = main_module ? PyModule_GetDict(main_module) : nullptr;
    std::string version;
    std::string failure;
    if (globals == nullptr) {
        failure = "python_main_unavailable";
    } else {
        // ssl/sqlite3/ctypes load native extensions from the bundled tree;
        // venv is what project environments are created with.
        static const char* kScript =
            "import os, sys, ssl, sqlite3, ctypes, venv\n"
            "_home = os.path.normcase(os.path.realpath(cyxwiz_home))\n"
            "_prefix = os.path.normcase(os.path.realpath(sys.prefix))\n"
            "cyxwiz_prefix_ok = 'yes' if _prefix == _home else sys.prefix\n"
            "cyxwiz_version = '%d.%d.%d' % sys.version_info[:3]\n";
        PyObject* home_value = PyUnicode_FromWideChar(home.wstring().c_str(), -1);
        if (home_value) {
            PyDict_SetItemString(globals, "cyxwiz_home", home_value);
            Py_DECREF(home_value);
        }
        PyObject* result = PyRun_String(kScript, Py_file_input, globals, globals);
        if (result == nullptr) {
            PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
            PyErr_Fetch(&type, &value, &traceback);
            PyObject* text = value ? PyObject_Str(value) : nullptr;
            const char* message = text ? PyUnicode_AsUTF8(text) : nullptr;
            failure = std::string("python_import_failed detail='") +
                      (message ? message : "unknown") + "'";
            Py_XDECREF(text);
            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(traceback);
        } else {
            Py_DECREF(result);
            const std::string prefix_ok = FetchString(globals, "cyxwiz_prefix_ok");
            if (prefix_ok != "yes") {
                failure = "python_not_bundled prefix='" + prefix_ok + "'";
            } else {
                version = FetchString(globals, "cyxwiz_version");
                if (version.empty()) failure = "python_version_unavailable";
            }
        }
    }
    Py_FinalizeEx();
    if (!failure.empty()) {
        if (error) *error = failure;
        return {};
    }
    return version;
}
#else
std::string RunBundledPythonSmoke(const std::filesystem::path&, std::string*) {
    return "disabled";
}
#endif

}  // namespace cyxwiz::core
