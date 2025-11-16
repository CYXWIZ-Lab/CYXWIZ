#pragma once

#include <string>
#include <vector>
#include <memory>

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {
class CommandWindowPanel;
}

namespace scripting {

/**
 * StartupScriptManager - Manages and executes startup scripts
 *
 * Automatically runs .cyx Python scripts on application startup.
 * Scripts are defined in a configuration file (startup_scripts.txt).
 */
class StartupScriptManager {
public:
    explicit StartupScriptManager(std::shared_ptr<ScriptingEngine> engine);
    ~StartupScriptManager() = default;

    // Configuration file management
    bool LoadConfig(const std::string& config_file = "startup_scripts.txt");
    bool SaveConfig(const std::string& config_file = "startup_scripts.txt");

    // Script execution
    bool ExecuteAll(cyxwiz::CommandWindowPanel* output_window = nullptr);
    bool ExecuteScript(const std::string& filepath, cyxwiz::CommandWindowPanel* output_window = nullptr);

    // Script list management
    void AddScript(const std::string& filepath);
    void RemoveScript(const std::string& filepath);
    void ClearScripts();

    // Query
    const std::vector<std::string>& GetScriptList() const { return script_paths_; }
    bool IsEnabled() const { return enabled_; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }

    // Configuration
    void SetTimeout(int seconds) { timeout_seconds_ = seconds; }
    int GetTimeout() const { return timeout_seconds_; }

    void SetContinueOnError(bool continue_on_error) { continue_on_error_ = continue_on_error; }
    bool GetContinueOnError() const { return continue_on_error_; }

private:
    std::shared_ptr<ScriptingEngine> scripting_engine_;
    std::vector<std::string> script_paths_;

    bool enabled_;
    int timeout_seconds_;
    bool continue_on_error_;

    // Helper methods
    bool FileExists(const std::string& filepath) const;
    std::string ReadConfigFile(const std::string& filepath) const;
    void ParseConfigContent(const std::string& content);
};

} // namespace scripting
