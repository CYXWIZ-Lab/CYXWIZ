#include "python_engine.h"

#ifdef CYXWIZ_HAS_PYTHON

#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include "../core/engine_config.h"

#ifdef _WIN32
#include <stdlib.h>  // _putenv_s
#else
#include <stdlib.h>  // setenv
#endif

namespace py = pybind11;

namespace scripting {

PythonEngine::PythonEngine() : initialized_(false), main_thread_state_(nullptr) {
    Initialize();
}

PythonEngine::~PythonEngine() {
    Shutdown();
}

bool PythonEngine::Initialize() {
    try {
        // Check if Python is already initialized (by another PythonEngine instance)
        if (Py_IsInitialized()) {
            spdlog::info("Python interpreter already initialized (reusing existing)");
            initialized_ = true;
            initialized_by_us_ = false;  // We didn't initialize it, so don't finalize
            return true;
        }

        // Configure PYTHONHOME BEFORE initializing Python
        // This determines which Python installation is used
        ConfigurePythonHome();

        py::initialize_interpreter();
        spdlog::info("Python interpreter initialized");
        initialized_ = true;
        initialized_by_us_ = true;  // We initialized it, so we'll finalize it

        // Configure additional Python paths (site-packages, etc.)
        ConfigureCustomPythonPath();

        // Release the GIL so background threads can use Python
        // This is REQUIRED for multi-threaded Python execution
        ReleaseGIL();
        spdlog::info("GIL released for multi-threaded use");

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize Python: {}", e.what());
        return false;
    }
}

void PythonEngine::ConfigurePythonHome() {
    // Configure Python environment before Py_Initialize()
    // This MUST be called BEFORE py::initialize_interpreter()
    auto& config = cyxwiz::core::EngineConfig::Instance();

    std::string python_home = config.GetEffectivePythonHome();

    if (python_home.empty()) {
        // Use system Python - no PYTHONHOME needed
        spdlog::info("Using system Python (no PYTHONHOME set)");
        return;
    }

    // Check if the Python home directory exists
    if (!std::filesystem::exists(python_home)) {
        spdlog::warn("Python home '{}' does not exist, falling back to system Python", python_home);
        return;
    }

    if (config.UseBundledPython()) {
        // For bundled Python (embeddable distribution):
        // - DON'T set PYTHONHOME (it uses ._pth file for path configuration)
        // - Add the Python directory to PATH so DLLs are found
#ifdef _WIN32
        std::string path = std::getenv("PATH") ? std::getenv("PATH") : "";
        std::string new_path = python_home + ";" + path;
        if (_putenv_s("PATH", new_path.c_str()) != 0) {
            spdlog::error("Failed to update PATH for bundled Python");
        }
#else
        std::string ld_path = std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "";
        std::string new_ld_path = python_home + ":" + ld_path;
        if (setenv("LD_LIBRARY_PATH", new_ld_path.c_str(), 1) != 0) {
            spdlog::error("Failed to update LD_LIBRARY_PATH for bundled Python");
        }
#endif
        spdlog::info("Using bundled Python: {}", python_home);
    } else {
        // For custom Python: Set PYTHONHOME to use the specified installation
#ifdef _WIN32
        if (_putenv_s("PYTHONHOME", python_home.c_str()) != 0) {
            spdlog::error("Failed to set PYTHONHOME environment variable");
            return;
        }
#else
        if (setenv("PYTHONHOME", python_home.c_str(), 1) != 0) {
            spdlog::error("Failed to set PYTHONHOME environment variable");
            return;
        }
#endif
        spdlog::info("Using custom Python: {}", python_home);
    }
}

void PythonEngine::ConfigureCustomPythonPath() {
    // Configure additional paths after Python is initialized
    // This adds custom site-packages to sys.path if configured
    auto& config = cyxwiz::core::EngineConfig::Instance();

    // If using bundled Python, add its site-packages
    if (config.UseBundledPython()) {
        std::string bundled_home = config.GetBundledPythonHome();
        if (!bundled_home.empty()) {
            std::filesystem::path site_packages = std::filesystem::path(bundled_home) / "Lib" / "site-packages";
            if (std::filesystem::exists(site_packages)) {
                try {
                    py::gil_scoped_acquire acquire;
                    py::object sys = py::module_::import("sys");
                    py::list sys_path = sys.attr("path").cast<py::list>();
                    sys_path.insert(0, site_packages.string());
                    spdlog::info("Added bundled site-packages: {}", site_packages.string());
                } catch (const std::exception& e) {
                    spdlog::warn("Failed to add bundled site-packages: {}", e.what());
                }
            }
        }
        return;
    }

    // Custom Python path handling (when not using bundled)
    if (!config.HasCustomPythonPath()) {
        spdlog::info("Using system Python packages (no custom path configured)");
        return;
    }

    std::string packages_dir = config.GetPythonPackagesDir();
    if (packages_dir.empty()) {
        spdlog::warn("Custom Python path set but site-packages not found");
        return;
    }

    try {
        py::gil_scoped_acquire acquire;
        py::object sys = py::module_::import("sys");
        py::list sys_path = sys.attr("path").cast<py::list>();

        // Insert custom site-packages at the beginning of sys.path
        // This makes it have priority over system packages
        sys_path.insert(0, packages_dir);

        spdlog::info("Custom Python packages path configured: {}", packages_dir);
        spdlog::debug("  sys.path[0] = {}", py::str(sys_path[0]).cast<std::string>());
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to configure custom Python path: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::error("Error configuring Python path: {}", e.what());
    }
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
