#include "console.h"
#include <imgui.h>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include "../core/project_manager.h"
#include "../core/async_task_manager.h"
#include <thread>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdio>
#include <memory>
#include <array>
#endif

namespace {
    std::mutex g_console_mutex;
}  // namespace

namespace gui {

Console::Console()
    : scroll_to_bottom_(false)
    , show_window_(true)
    , auto_scroll_(true)
    , selected_tab_(0)
    , show_copy_notification_(false)
    , copy_notification_time_(0.0f)
{
    memset(input_buf_, 0, sizeof(input_buf_));
    // Note: Cannot call AddInfo() here as ImGui::GetTime() requires an active ImGui frame
    // Initial messages will be added in the first Render() call
}

Console::~Console() = default;

void Console::Render() {
    if (!show_window_) return;

    // Add initial messages on first render when ImGui context is active
    static bool first_render = true;
    if (first_render) {
        AddInfo("CyxWiz Console initialized");
        AddInfo("Type 'help' for available commands");
        AddInfo("Ready");
        first_render = false;
    }

    if (ImGui::Begin("Console", &show_window_)) {
        // Toolbar
        if (ImGui::Button("Clear")) {
            Clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Copy All")) {
            CopyAllLogs();
            ShowCopyNotification();
        }
        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &auto_scroll_);
        ImGui::SameLine();

        // Status display: "Copied!" in green for 2 seconds, then "Ready"
        if (show_copy_notification_) {
            float elapsed = static_cast<float>(ImGui::GetTime()) - copy_notification_time_;
            if (elapsed < 2.0f) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f)); // Green
                ImGui::Text("Copied!");
                ImGui::PopStyleColor();
            } else {
                show_copy_notification_ = false;
                ImGui::TextDisabled("Ready");
            }
        } else {
            ImGui::TextDisabled("Ready");
        }

        ImGui::Separator();

        // Tabs
        if (ImGui::BeginTabBar("ConsoleTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("All")) {
                RenderAllTab();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Info")) {
                RenderLogTab("Info", LogLevel::Info);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Warnings")) {
                RenderLogTab("Warnings", LogLevel::Warning);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Errors")) {
                RenderLogTab("Errors", LogLevel::Error);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Success")) {
                RenderLogTab("Success", LogLevel::Success);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::Separator();

        // Command input at bottom
        bool reclaim_focus = false;
        ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue;
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputTextWithHint("##input", "Enter command...", input_buf_, IM_ARRAYSIZE(input_buf_), input_text_flags)) {
            char* s = input_buf_;
            if (s[0]) {
                ExecCommand(s);
            }
            strcpy(s, "");
            reclaim_focus = true;
        }
        ImGui::PopItemWidth();

        ImGui::SetItemDefaultFocus();
        if (reclaim_focus) {
            ImGui::SetKeyboardFocusHere(-1);
        }
    }
    ImGui::End();
}

void Console::RenderAllTab() {
    const float footer_height = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImGui::BeginChild("AllLogsRegion", ImVec2(0, -footer_height), false, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

    for (size_t i = 0; i < items_.size(); i++) {
        const auto& entry = items_[i];
        ImVec4 color = GetLevelColor(entry.level);
        ImGui::PushStyleColor(ImGuiCol_Text, color);

        // Use Selectable for right-click support
        char buf[512];
        snprintf(buf, sizeof(buf), "[%.2fs] %s %s",
            entry.timestamp,
            GetLevelPrefix(entry.level),
            entry.message.c_str());

        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(buf, false, ImGuiSelectableFlags_AllowDoubleClick)) {
            if (ImGui::IsMouseDoubleClicked(0)) {
                ImGui::SetClipboardText(entry.message.c_str());
                ShowCopyNotification();
            }
        }

        // Right-click context menu
        if (ImGui::BeginPopupContextItem("LogContextMenu")) {
            if (ImGui::MenuItem("Copy Message")) {
                ImGui::SetClipboardText(entry.message.c_str());
                ShowCopyNotification();
            }
            if (ImGui::MenuItem("Copy Full Line")) {
                ImGui::SetClipboardText(buf);
                ShowCopyNotification();
            }
            ImGui::EndPopup();
        }
        ImGui::PopID();

        ImGui::PopStyleColor();
    }

    if (auto_scroll_ && (scroll_to_bottom_ || ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.0f);
    }

    scroll_to_bottom_ = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();
}

void Console::RenderLogTab(const char* name, LogLevel filter) {
    const float footer_height = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImGui::BeginChild((std::string(name) + "Region").c_str(), ImVec2(0, -footer_height), false, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

    int count = 0;
    int id_counter = 0;
    for (size_t i = 0; i < items_.size(); i++) {
        const auto& entry = items_[i];
        if (entry.level == filter) {
            ImVec4 color = GetLevelColor(entry.level);
            ImGui::PushStyleColor(ImGuiCol_Text, color);

            char buf[512];
            snprintf(buf, sizeof(buf), "[%.2fs] %s",
                entry.timestamp,
                entry.message.c_str());

            ImGui::PushID(id_counter++);
            if (ImGui::Selectable(buf, false, ImGuiSelectableFlags_AllowDoubleClick)) {
                if (ImGui::IsMouseDoubleClicked(0)) {
                    ImGui::SetClipboardText(entry.message.c_str());
                    ShowCopyNotification();
                }
            }

            if (ImGui::BeginPopupContextItem("LogContextMenu")) {
                if (ImGui::MenuItem("Copy Message")) {
                    ImGui::SetClipboardText(entry.message.c_str());
                    ShowCopyNotification();
                }
                if (ImGui::MenuItem("Copy Full Line")) {
                    ImGui::SetClipboardText(buf);
                    ShowCopyNotification();
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();

            ImGui::PopStyleColor();
            count++;
        }
    }

    if (count == 0) {
        ImGui::TextDisabled("No %s messages", name);
    }

    if (auto_scroll_ && (scroll_to_bottom_ || ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.0f);
    }

    scroll_to_bottom_ = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();
}

void Console::AddLog(const std::string& message, LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    LogEntry entry;
    entry.message = message;
    entry.level = level;
    entry.timestamp = ImGui::GetTime();
    items_.push_back(entry);
    scroll_to_bottom_ = true;

    // Keep history bounded (e.g., 1000 entries)
    if (items_.size() > 1000) {
        items_.erase(items_.begin());
    }
}

void Console::AddInfo(const std::string& message) {
    AddLog(message, LogLevel::Info);
}

void Console::AddWarning(const std::string& message) {
    AddLog(message, LogLevel::Warning);
}

void Console::AddError(const std::string& message) {
    AddLog(message, LogLevel::Error);
}

void Console::AddSuccess(const std::string& message) {
    AddLog(message, LogLevel::Success);
}

void Console::Clear() {
    items_.clear();
    AddInfo("Console cleared");
}

bool Console::IsPipCommand(const std::string& command) const {
    // Check if command starts with pip
    if (command.find("pip") == 0 || command.find("pip3") == 0) {
        return true;
    }
    return false;
}

void Console::ExecutePipCommand(const std::string& pip_args) {
    auto& pm = cyxwiz::ProjectManager::Instance();

    if (!pm.HasActiveProject()) {
        AddError("No active project - pip commands require an open project");
        return;
    }

    // Get the project's venv pip path
    std::filesystem::path project_root(pm.GetProjectRoot());
    std::filesystem::path venv_pip;

#ifdef _WIN32
    venv_pip = project_root / "python" / "Scripts" / "pip.exe";
#else
    venv_pip = project_root / "python" / "bin" / "pip";
#endif

    if (!std::filesystem::exists(venv_pip)) {
        AddError("Project virtual environment not found");
        AddInfo("Please wait for venv creation to complete or create it manually");
        return;
    }

    // Build the full command
    std::string full_command = "\"" + venv_pip.string() + "\" " + pip_args;

    AddInfo("Executing: " + full_command);
    AddInfo("Running in background (UI remains responsive)...");
    spdlog::info("Console executing pip command asynchronously: {}", full_command);

    // Run command asynchronously using AsyncTaskManager
    auto& task_mgr = cyxwiz::AsyncTaskManager::Instance();

    // Capture 'this' pointer for thread-safe logging
    Console* console_ptr = this;

    task_mgr.RunAsync(
        "pip " + pip_args,
        [console_ptr, full_command](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.1f, "Starting pip command...");

#ifdef _WIN32
            // Windows: Use CreateProcess with pipes
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = NULL;

            HANDLE hStdoutRead, hStdoutWrite;
            if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) {
                console_ptr->AddError("Failed to create pipe for command output");
                task.MarkFailed("Failed to create pipe");
                return;
            }

            SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOA si;
            PROCESS_INFORMATION pi;
            ZeroMemory(&si, sizeof(si));
            si.cb = sizeof(si);
            si.hStdError = hStdoutWrite;
            si.hStdOutput = hStdoutWrite;
            si.dwFlags |= STARTF_USESTDHANDLES;
            ZeroMemory(&pi, sizeof(pi));

            std::string cmd_copy = full_command;  // CreateProcessA modifies the string
            if (!CreateProcessA(NULL, const_cast<char*>(cmd_copy.c_str()), NULL, NULL, TRUE,
                                CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
                console_ptr->AddError("Failed to execute pip command");
                CloseHandle(hStdoutRead);
                CloseHandle(hStdoutWrite);
                task.MarkFailed("Failed to create process");
                return;
            }

            CloseHandle(hStdoutWrite);

            task.ReportProgress(0.3f, "Reading pip output...");

            // Read output in real-time
            char buffer[4096];
            DWORD bytes_read;
            std::string line_buffer;

            while (ReadFile(hStdoutRead, buffer, sizeof(buffer) - 1, &bytes_read, NULL) && bytes_read > 0) {
                buffer[bytes_read] = '\0';
                line_buffer += buffer;

                // Process complete lines
                size_t pos;
                while ((pos = line_buffer.find('\n')) != std::string::npos) {
                    std::string line = line_buffer.substr(0, pos);
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    if (!line.empty()) {
                        console_ptr->AddInfo(line);
                    }
                    line_buffer = line_buffer.substr(pos + 1);
                }

                // Check for cancellation
                if (task.IsCancelRequested()) {
                    TerminateProcess(pi.hProcess, 1);
                    console_ptr->AddWarning("Command cancelled by user");
                    break;
                }
            }

            // Print remaining buffer
            if (!line_buffer.empty()) {
                console_ptr->AddInfo(line_buffer);
            }

            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exit_code;
            GetExitCodeProcess(pi.hProcess, &exit_code);

            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hStdoutRead);

            task.ReportProgress(1.0f, "Command finished");

            if (exit_code == 0) {
                console_ptr->AddSuccess("Command completed successfully");
            } else {
                console_ptr->AddError("Command failed with exit code: " + std::to_string(exit_code));
                task.MarkFailed("Exit code: " + std::to_string(exit_code));
            }
#else
            // Linux/macOS: Use popen
            FILE* pipe = popen(full_command.c_str(), "r");
            if (!pipe) {
                console_ptr->AddError("Failed to execute pip command");
                task.MarkFailed("Failed to open pipe");
                return;
            }

            task.ReportProgress(0.3f, "Reading pip output...");

            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string line(buffer);
                // Remove trailing newline
                if (!line.empty() && line.back() == '\n') {
                    line.pop_back();
                }
                if (!line.empty()) {
                    console_ptr->AddInfo(line);
                }

                // Check for cancellation
                if (task.IsCancelRequested()) {
                    pclose(pipe);
                    console_ptr->AddWarning("Command cancelled by user");
                    return;
                }
            }

            task.ReportProgress(1.0f, "Command finished");

            int exit_code = pclose(pipe);
            if (exit_code == 0) {
                console_ptr->AddSuccess("Command completed successfully");
            } else {
                console_ptr->AddError("Command failed with exit code: " + std::to_string(exit_code));
                task.MarkFailed("Exit code: " + std::to_string(exit_code));
            }
#endif
        },
        nullptr,  // progress callback
        [console_ptr](bool success, const std::string& error) {
            if (!success && !error.empty()) {
                spdlog::error("Pip command task failed: {}", error);
            }
        }
    );
}

void Console::ExecCommand(const char* command) {
    AddLog(std::string("> ") + command, LogLevel::Info);

    std::string cmd_str(command);

    // Check for pip commands
    if (IsPipCommand(cmd_str)) {
        // Extract pip arguments (remove "pip" or "pip3" prefix)
        size_t pip_end = cmd_str.find("pip");
        if (pip_end != std::string::npos) {
            pip_end += 3; // Length of "pip"
            if (cmd_str[pip_end] == '3') {
                pip_end++; // Skip '3' in pip3
            }
            // Skip whitespace
            while (pip_end < cmd_str.length() && std::isspace(cmd_str[pip_end])) {
                pip_end++;
            }
            std::string pip_args = cmd_str.substr(pip_end);
            ExecutePipCommand(pip_args);
        }
        return;
    }

    // Built-in console commands
    if (strcmp(command, "clear") == 0) {
        Clear();
    } else if (strcmp(command, "help") == 0) {
        AddInfo("=== CyxWiz Console Help ===");
        AddInfo("");
        AddInfo("Built-in Commands:");
        AddInfo("  clear          - Clear all console output");
        AddInfo("  help           - Show this help message");
        AddInfo("  test           - Test different log level displays");
        AddInfo("");
        AddInfo("Package Management (pip):");
        AddInfo("  pip install <package>      - Install a package (e.g., pip install numpy)");
        AddInfo("  pip uninstall <package>    - Uninstall a package");
        AddInfo("  pip list                   - List installed packages");
        AddInfo("  pip show <package>         - Show package information");
        AddInfo("  pip freeze                 - Output installed packages in requirements format");
        AddInfo("");
        AddInfo("Examples:");
        AddInfo("  pip install numpy pandas matplotlib");
        AddInfo("  pip install torch --index-url https://download.pytorch.org/whl/cpu");
        AddInfo("  pip list");
        AddInfo("");
        AddInfo("Notes:");
        AddInfo("  - All pip commands run in the project's virtual environment");
        AddInfo("  - Requires an active project to be open");
        AddInfo("  - Large installs (like PyTorch) may take a few minutes");
    } else if (strcmp(command, "test") == 0) {
        AddInfo("This is an info message");
        AddWarning("This is a warning message");
        AddError("This is an error message");
        AddSuccess("This is a success message");
    } else {
        AddError(std::string("Unknown command: ") + command);
        AddInfo("Type 'help' for available commands");
    }
}

const char* Console::GetLevelPrefix(LogLevel level) const {
    switch (level) {
        case LogLevel::Info:    return "[INFO]";
        case LogLevel::Warning: return "[WARN]";
        case LogLevel::Error:   return "[ERROR]";
        case LogLevel::Success: return "[OK]";
        case LogLevel::Debug:   return "[DEBUG]";
        default:                return "[???]";
    }
}

ImVec4 Console::GetLevelColor(LogLevel level) const {
    switch (level) {
        case LogLevel::Info:    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f); // Gray
        case LogLevel::Warning: return ImVec4(1.0f, 0.8f, 0.0f, 1.0f); // Yellow
        case LogLevel::Error:   return ImVec4(1.0f, 0.3f, 0.3f, 1.0f); // Red
        case LogLevel::Success: return ImVec4(0.3f, 1.0f, 0.3f, 1.0f); // Green
        case LogLevel::Debug:   return ImVec4(0.6f, 0.6f, 1.0f, 1.0f); // Blue
        default:                return ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // White
    }
}

void Console::CopyAllLogs() {
    std::ostringstream ss;
    for (const auto& entry : items_) {
        ss << "[" << std::fixed;
        ss.precision(2);
        ss << entry.timestamp << "s] "
           << GetLevelPrefix(entry.level) << " "
           << entry.message << "\n";
    }
    ImGui::SetClipboardText(ss.str().c_str());
}

void Console::ShowCopyNotification() {
    show_copy_notification_ = true;
    copy_notification_time_ = static_cast<float>(ImGui::GetTime());
}

} // namespace gui
