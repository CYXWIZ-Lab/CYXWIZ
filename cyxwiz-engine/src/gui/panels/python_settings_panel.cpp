#include "python_settings_panel.h"

#include "../../core/engine_config.h"
#include "../../core/file_dialogs.h"
#include "../../core/python_detector.h"
#include "../../core/python_package_list.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <imgui_stdlib.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace gui {

namespace ui = cyxwiz::ui;
using cyxwiz::core::EngineConfig;
using cyxwiz::core::PythonDetector;

namespace {
void Heading(const char* text) {
    ImGui::Spacing();
    ImGui::TextUnformatted(text);
    ImGui::Separator();
    ImGui::Spacing();
}

void Note(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", text);
    ImGui::PopTextWrapPos();
}

template <typename T>
bool Ready(const std::future<T>& f) {
    return f.valid() && f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

std::string Trimmed(std::string s) {
    s.erase(0, s.find_first_not_of(" \t\""));
    s.erase(s.find_last_not_of(" \t\"") + 1);
    return s;
}
}  // namespace

void PythonSettingsPanel::Load() {
    auto& config = EngineConfig::Instance();
    saved_path_ = config.GetSystemPythonPath();
    path_ = saved_path_;
    packages_ = cyxwiz::core::JoinPackageList(config.GetDefaultVenvPackages());
    startup_script_ = config.GetPythonStart().startup_script;
    if (!path_.empty()) StartCheck(path_);
}

void PythonSettingsPanel::StartCheck(const std::string& path) {
    if (path.empty() || checks_.count(path) || (checking_.valid() && checking_path_ == path)) return;
    if (checking_.valid()) checking_.wait();  // one check at a time; Poll stores it
    Poll();
    checking_path_ = path;
    checking_ = std::async(std::launch::async, [path]() {
        Check c;
        const auto python = PythonDetector::ValidatePythonInstallation(path);
        if (!python)
            c.error = "Not a Python interpreter (or it cannot be run)";
        else if (!PythonDetector::MeetsRequirements(*python))
            c.error = PythonDetector::GetRequirementError(*python);
        else
            c.version = python->version;
        return c;
    });
}

void PythonSettingsPanel::Poll() {
    if (Ready(checking_)) checks_[checking_path_] = checking_.get();
    if (Ready(finding_)) {
        found_ = finding_.get();
        found_shown_ = true;
        for (const auto& f : found_) checks_[f.path] = Check{f.version, ""};
        spdlog::info("{} of them supported ({})", found_.size(), PythonDetector::SupportedVersionsText());
    }
}

void PythonSettingsPanel::Render(const std::function<std::string()>& runtime_details) {
    if (!loaded_) {
        Load();
        loaded_ = true;
    }
    Poll();
    RenderInterpreter(runtime_details);
    RenderNewProjects();
    RenderSessionStart();
    RenderConsole();
}

void PythonSettingsPanel::RenderInterpreter(const std::function<std::string()>& runtime_details) {
    const auto& tokens = ui::CurrentTokens();
    Heading("Interpreter");
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Python");
    ImGui::SameLine();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float buttons = ui::ButtonWidth("Browse...", ui::ButtonSize::Small) +
                          ui::ButtonWidth("Find Python", ui::ButtonSize::Small) + 2.0f * spacing;
    ImGui::SetNextItemWidth(std::max(200.0f, ImGui::GetContentRegionAvail().x - buttons));
    ImGui::InputTextWithHint("##python_path", "path to python.exe", &path_);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        path_ = Trimmed(path_);
        StartCheck(path_);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Browse...##python")) {
        if (auto chosen = cyxwiz::FileDialogs::OpenFile(
                "Choose a Python interpreter", {{"Python", "exe"}, {"All files", "*"}},
                path_.empty() ? nullptr : path_.c_str())) {
            path_ = *chosen;
            StartCheck(path_);
        }
    }
    ImGui::SameLine();
    const bool finding = finding_.valid();
    if (ui::SecondaryButton("Find Python", !finding, "Looking for Python on this PC...")) {
        spdlog::info("Looking for Python installations...");
        finding_ = std::async(std::launch::async, []() {
            std::vector<Found> found;
            for (const auto& p : PythonDetector::FindAllPythonInstallations())
                if (PythonDetector::MeetsRequirements(p)) found.push_back({p.executable_path, p.version});
            return found;
        });
    }

    // What the field holds
    const auto it = checks_.find(path_);
    if (path_.empty()) {
        Note("None set, and no Python ships with this Engine: Python features are off until one is chosen.");
    } else if (checking_.valid() && checking_path_ == path_) {
        ImGui::TextDisabled("Checking %s...", path_.c_str());
    } else if (it == checks_.end()) {
        Note("Press Enter or leave the field to check it.");
    } else if (!it->second.error.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(tokens.error, ICON_FA_XMARK " %s", it->second.error.c_str());
        ImGui::PopTextWrapPos();
    } else {
        ImGui::TextColored(tokens.success, ICON_FA_CHECK " Python %s", it->second.version.c_str());
        if (path_ == saved_path_) {
            ImGui::SameLine();
            ImGui::TextDisabled("- project environments are made from it; Python runs in it when a project has none -");
            ImGui::SameLine();
            if (ui::LinkButton("Runtime details")) {
                details_text_ = runtime_details ? runtime_details() : std::string("Python runtime details are not available.");
                ImGui::OpenPopup("Python runtime details");
            }
        }
    }
    if (path_ != saved_path_) {
        const bool usable = it != checks_.end() && it->second.error.empty();
        if (ui::PrimaryButton("Use this Python", usable,
                              it == checks_.end() ? "Check it first (Enter in the field)" : "This one cannot be used",
                              ui::ButtonSize::Small)) {
            auto& config = EngineConfig::Instance();
            config.SetSystemPythonPath(path_);
            config.Save();
            saved_path_ = path_;
            changed_this_run_ = true;
            spdlog::info("Python interpreter set: {} (Python {})", path_, it->second.version);
        }
        ImGui::SameLine();
        if (ui::LinkButton("Keep the current one")) path_ = saved_path_;
    }
    if (changed_this_run_)
        Note("Python that is already running keeps its interpreter until the Engine restarts; new project "
             "environments are made from this one.");

    if (found_shown_) {
        ImGui::Spacing();
        if (found_.empty()) {
            ImGui::TextDisabled("No Python %s found on this PC.", PythonDetector::SupportedVersionsText().c_str());
        } else {
            ImGui::TextDisabled("Found on this PC (pick one, then Use this Python):");
            for (size_t i = 0; i < found_.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                const std::string label = "Python " + found_[i].version + "   " + found_[i].path;
                if (ImGui::Selectable(label.c_str(), path_ == found_[i].path)) path_ = found_[i].path;
                ImGui::PopID();
            }
        }
    }

    if (ImGui::BeginPopupModal("Python runtime details", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const ImVec2 size(std::min(720.0f, ImGui::GetMainViewport()->WorkSize.x * 0.6f),
                          std::min(360.0f, ImGui::GetMainViewport()->WorkSize.y * 0.5f));
        ImGui::BeginChild("##python_details", size, ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::TextUnformatted(details_text_.c_str());
        ImGui::EndChild();
        if (ui::SecondaryButton("Copy")) ImGui::SetClipboardText(details_text_.c_str());
        ImGui::SameLine();
        if (ui::SecondaryButton("Close")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void PythonSettingsPanel::RenderNewProjects() {
    auto& config = EngineConfig::Instance();
    Heading("New projects");
    bool venv = config.GetAutoCreateVenv();
    if (ImGui::Checkbox("Create a virtual environment for each new project", &venv)) {
        config.SetAutoCreateVenv(venv);
        config.Save();
    }
    Note("In the project's python folder, made from the interpreter above. A project without one runs in that "
         "interpreter.");
    ImGui::BeginDisabled(!venv);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Packages to install");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##packages", "numpy pandas (space separated; empty: none)", &packages_);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        packages_ignored_.clear();
        const auto packages = cyxwiz::core::ParsePackageList(packages_, &packages_ignored_);
        config.SetDefaultVenvPackages(packages);
        config.Save();
        packages_ = cyxwiz::core::JoinPackageList(packages);
    }
    ImGui::EndDisabled();
    if (!packages_ignored_.empty())
        ImGui::TextColored(ui::CurrentTokens().warning, "Not passed to pip: %s (pip options are not taken here)",
                           cyxwiz::core::JoinPackageList(packages_ignored_).c_str());
}

void PythonSettingsPanel::RenderSessionStart() {
    auto& config = EngineConfig::Instance();
    Heading("When Python starts");
    Note("Each Python session: before its first command or script, and again after Restart. A change applies to "
         "the next session.");
    auto start = config.GetPythonStart();
    bool changed = false;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Startup script");
    ImGui::SameLine();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    float buttons = ui::ButtonWidth("Browse...", ui::ButtonSize::Small) + spacing;
    if (!startup_script_.empty()) buttons += ImGui::CalcTextSize("Clear").x + 2.0f * spacing;
    ImGui::SetNextItemWidth(std::max(200.0f, ImGui::GetContentRegionAvail().x - buttons));
    ImGui::InputTextWithHint("##startup_script", "none", &startup_script_);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        startup_script_ = Trimmed(startup_script_);
        changed = true;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Browse...##startup")) {
        if (auto chosen = cyxwiz::FileDialogs::OpenFile("Choose a startup script",
                                                        {{"Python scripts", "py,cyx"}, {"All files", "*"}},
                                                        startup_script_.empty() ? nullptr : startup_script_.c_str())) {
            startup_script_ = *chosen;
            changed = true;
        }
    }
    if (!startup_script_.empty()) {
        ImGui::SameLine();
        if (ui::LinkButton("Clear##startup")) {
            startup_script_.clear();
            changed = true;
        }
        std::error_code ec;
        if (!std::filesystem::is_regular_file(std::filesystem::path(startup_script_), ec))
            ImGui::TextColored(ui::CurrentTokens().warning, "The file is not there: sessions start without it.");
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Imports");
    ImGui::SameLine();
    changed |= ImGui::Checkbox("import numpy as np", &start.import_numpy);
    ImGui::SameLine();
    changed |= ImGui::Checkbox("import pycyxwiz", &start.import_pycyxwiz);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("With its short names: cyx, svd, eig, zeros, ones and the other MATLAB-style functions.");
    if (changed) {
        start.startup_script = startup_script_;
        config.SetPythonStart(start);
        config.Save();
    }
}

void PythonSettingsPanel::RenderConsole() {
    auto& config = EngineConfig::Instance();
    Heading("Python console");
    auto start = config.GetPythonStart();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Commands kept");
    ImGui::SameLine();
    if (ui::NumberField("##commands_kept", &start.console_commands_kept,
                        cyxwiz::core::PythonStartConfig::kMinCommandsKept,
                        cyxwiz::core::PythonStartConfig::kMaxCommandsKept, 100, 80.0f)) {
        config.SetPythonStart(start);
        config.Save();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("each with its output; 100 to 10,000");
}

}  // namespace gui
