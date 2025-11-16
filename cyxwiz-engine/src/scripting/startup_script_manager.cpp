#include "startup_script_manager.h"
#include "scripting_engine.h"
#include "../gui/panels/command_window.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

namespace scripting {

StartupScriptManager::StartupScriptManager(std::shared_ptr<ScriptingEngine> engine)
    : scripting_engine_(engine)
    , enabled_(true)
    , timeout_seconds_(60)
    , continue_on_error_(true)
{
    spdlog::info("StartupScriptManager created");
}

bool StartupScriptManager::LoadConfig(const std::string& config_file) {
    spdlog::info("Loading startup scripts config: {}", config_file);

    if (!FileExists(config_file)) {
        spdlog::info("Startup scripts config not found (this is normal for first run)");
        return false;
    }

    std::string content = ReadConfigFile(config_file);
    if (content.empty()) {
        spdlog::warn("Startup scripts config is empty");
        return false;
    }

    ParseConfigContent(content);

    spdlog::info("Loaded {} startup scripts from config", script_paths_.size());
    return true;
}

bool StartupScriptManager::SaveConfig(const std::string& config_file) {
    std::ofstream file(config_file);
    if (!file.is_open()) {
        spdlog::error("Failed to open config file for writing: {}", config_file);
        return false;
    }

    // Write header
    file << "# CyxWiz Startup Scripts Configuration\n";
    file << "# Lines starting with # are comments\n";
    file << "# One script path per line (absolute or relative)\n";
    file << "#\n";
    file << "# Example:\n";
    file << "#   scripts/startup/init_imports.cyx\n";
    file << "#   C:/Users/me/my_startup.cyx\n";
    file << "\n";

    // Write script paths
    for (const auto& path : script_paths_) {
        file << path << "\n";
    }

    spdlog::info("Saved {} startup scripts to config: {}", script_paths_.size(), config_file);
    return true;
}

bool StartupScriptManager::ExecuteAll(cyxwiz::CommandWindowPanel* output_window) {
    if (!enabled_) {
        spdlog::info("Startup scripts disabled, skipping execution");
        return false;
    }

    if (script_paths_.empty()) {
        spdlog::info("No startup scripts configured");
        return true;
    }

    spdlog::info("Executing {} startup scripts...", script_paths_.size());

    if (output_window) {
        output_window->DisplayScriptOutput(
            "Startup Scripts",
            "=== Running startup scripts ===",
            false
        );
    }

    auto start_time = std::chrono::steady_clock::now();
    int success_count = 0;
    int error_count = 0;

    for (const auto& script_path : script_paths_) {
        bool result = ExecuteScript(script_path, output_window);
        if (result) {
            success_count++;
        } else {
            error_count++;
            if (!continue_on_error_) {
                spdlog::warn("Stopping startup script execution due to error");
                break;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::ostringstream summary;
    summary << "=== Startup scripts complete ("
            << success_count << " successful, "
            << error_count << " errors, "
            << duration.count() / 1000.0 << "s) ===";

    if (output_window) {
        output_window->DisplayScriptOutput("Startup Scripts", summary.str(), false);
    }

    spdlog::info(summary.str());

    return error_count == 0;
}

bool StartupScriptManager::ExecuteScript(const std::string& filepath, cyxwiz::CommandWindowPanel* output_window) {
    if (!FileExists(filepath)) {
        std::string error_msg = "Script not found: " + filepath;
        spdlog::error(error_msg);

        if (output_window) {
            output_window->DisplayScriptOutput(filepath, error_msg, true);
        }

        return false;
    }

    spdlog::info("Executing startup script: {}", filepath);

    if (output_window) {
        output_window->DisplayScriptOutput(filepath, "Executing: " + filepath, false);
    }

    if (!scripting_engine_) {
        std::string error_msg = "ScriptingEngine not initialized";
        spdlog::error(error_msg);

        if (output_window) {
            output_window->DisplayScriptOutput(filepath, error_msg, true);
        }

        return false;
    }

    // Read script file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::string error_msg = "Failed to open script: " + filepath;
        spdlog::error(error_msg);

        if (output_window) {
            output_window->DisplayScriptOutput(filepath, error_msg, true);
        }

        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string script_content = buffer.str();

    // Execute script
    auto result = scripting_engine_->ExecuteScript(script_content);

    // Display output
    if (output_window) {
        if (!result.output.empty()) {
            output_window->DisplayScriptOutput(filepath, result.output, false);
        }

        if (!result.success) {
            output_window->DisplayScriptOutput(filepath, "Error: " + result.error_message, true);
        }
    }

    if (!result.success) {
        spdlog::error("Startup script failed: {} - {}", filepath, result.error_message);
    } else {
        spdlog::info("Startup script completed: {}", filepath);
    }

    return result.success;
}

void StartupScriptManager::AddScript(const std::string& filepath) {
    // Check if already in list
    for (const auto& path : script_paths_) {
        if (path == filepath) {
            spdlog::warn("Script already in startup list: {}", filepath);
            return;
        }
    }

    script_paths_.push_back(filepath);
    spdlog::info("Added startup script: {}", filepath);
}

void StartupScriptManager::RemoveScript(const std::string& filepath) {
    auto it = std::find(script_paths_.begin(), script_paths_.end(), filepath);
    if (it != script_paths_.end()) {
        script_paths_.erase(it);
        spdlog::info("Removed startup script: {}", filepath);
    } else {
        spdlog::warn("Script not found in startup list: {}", filepath);
    }
}

void StartupScriptManager::ClearScripts() {
    script_paths_.clear();
    spdlog::info("Cleared all startup scripts");
}

bool StartupScriptManager::FileExists(const std::string& filepath) const {
    return std::filesystem::exists(filepath);
}

std::string StartupScriptManager::ReadConfigFile(const std::string& filepath) const {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void StartupScriptManager::ParseConfigContent(const std::string& content) {
    script_paths_.clear();

    std::istringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Add script path
        script_paths_.push_back(line);
        spdlog::debug("Added startup script from config: {}", line);
    }
}

} // namespace scripting
