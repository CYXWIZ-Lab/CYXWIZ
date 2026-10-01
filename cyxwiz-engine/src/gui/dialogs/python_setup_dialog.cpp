#include "python_setup_dialog.h"

#include "../../core/engine_config.h"
#include "../../core/file_dialogs.h"
#include "../../core/python_detector.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <chrono>

namespace cyxwiz {

namespace {
constexpr const char* kTitle = "Python for scripting###python_setup";
constexpr const char* kDownloads = "https://www.python.org/downloads/";

std::string Required() {
#ifdef CYXWIZ_PYTHON_LINKED_MINOR
    return "3." + std::to_string(CYXWIZ_PYTHON_LINKED_MINOR);
#else
    return "3.12 or 3.13";
#endif
}

pythonsetup::Candidate ToCandidate(const core::PythonDetector::PythonInstallation& p) {
    pythonsetup::Candidate c;
    c.version = p.version;
    c.path = p.executable_path;
    c.venv = p.has_venv_module;
    c.pip = p.has_pip;
    c.usable = core::PythonDetector::MeetsRequirements(p);
    if (!c.usable) c.reason = core::PythonDetector::GetRequirementError(p);
    c.home = p.home;
    return c;
}

// Runs on the worker: no EngineConfig writes, no ImGui.
pythonsetup::ScanResult RunScan(std::string configured) {
    const auto start = std::chrono::steady_clock::now();
    pythonsetup::ScanResult r;
    r.configured_path = configured;
    r.configured_path_set = !configured.empty();
    bool configured_listed = false;
    if (r.configured_path_set) {
        if (auto info = core::PythonDetector::ValidatePythonInstallation(configured)) {
            r.configured_ok = core::PythonDetector::MeetsRequirements(*info);
            r.found.push_back(ToCandidate(*info));
            configured_listed = true;
        }
    }
    for (const auto& p : core::PythonDetector::FindAllPythonInstallations()) {
        if (configured_listed && p.executable_path == configured) continue;
        r.found.push_back(ToCandidate(p));
    }
    r.scanned = true;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}
}  // namespace

void PythonSetupDialog::StartScan() {
    if (scanning_) return;
    scanning_ = true;
    browse_note_.clear();
    spdlog::info("Scanning for Python installations in the background...");
    worker_ = std::async(std::launch::async, RunScan, core::EngineConfig::Instance().GetSystemPythonPath());
}

bool PythonSetupDialog::Poll() {
    if (!scanning_ || worker_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    scanning_ = false;
    scan_ = worker_.get();
    auto& config = core::EngineConfig::Instance();
    in_use_.clear();
    if (scan_.configured_ok) {
        in_use_ = scan_.configured_path;
    } else {
        for (const auto& c : scan_.found) {
            if (!c.usable) continue;
            // Use the best one for this session without saving it, as before.
            config.SetDetectedPythonPath(c.path);
            in_use_ = c.path;
            break;
        }
    }
    const auto view = View();
    choice_ = view.selected;
    if (view.ready) {
        spdlog::info("Python {} in use: {} (scan {:.1f} s)", view.usable[view.selected].version, in_use_, scan_.seconds);
    } else {
        spdlog::warn("{} ({} interpreters found, none usable)", view.headline, scan_.found.size());
        request_open_ = true;
    }
    return true;
}

pythonsetup::PythonView PythonSetupDialog::View() const {
    return pythonsetup::BuildPythonView(scan_, scanning_, Required(), in_use_);
}

void PythonSetupDialog::Browse() {
#ifdef _WIN32
    const FileDialogs::FilterList filters = {{"Python interpreter", "exe"}};
#else
    const FileDialogs::FilterList filters = {};
#endif
    const auto picked = FileDialogs::OpenFile("Choose a Python interpreter", filters);
    if (!picked) return;
    const auto info = core::PythonDetector::ValidatePythonInstallation(*picked);
    if (!info) {
        browse_ok_ = false;
        browse_note_ = "That file did not run as a Python interpreter: " + *picked;
        return;
    }
    pythonsetup::Candidate c = ToCandidate(*info);
    for (auto it = scan_.found.begin(); it != scan_.found.end();) it = it->path == c.path ? scan_.found.erase(it) : it + 1;
    scan_.found.insert(scan_.found.begin(), c);
    if (c.usable) {
        browse_ok_ = true;
        browse_note_ = "Python " + c.version + " added. Press Use this Python to keep it.";
        choice_ = 0;
    } else {
        browse_ok_ = false;
        browse_note_ = "Python " + c.version + " cannot be used: " + c.reason;
    }
}

PythonSetupDialog::Outcome PythonSetupDialog::Render() {
    using namespace ui;
    if (request_open_) {
        ImGui::OpenPopup(kTitle);
        request_open_ = false;
    }
    DialogOptions options;
    options.size = ImVec2(700.0f, 440.0f);
    options.min_size = ImVec2(520.0f, 380.0f);
    if (!BeginDialog(kTitle, options)) return Outcome::None;
    const Tokens& t = CurrentTokens();
    const pythonsetup::PythonView view = View();

    {
        FontScope bold(Font::Bold);
        StatusText(view.level == 0 ? Status::Verified : (view.level == 1 ? Status::Verifying : Status::NeedsDriverUpdate),
                   view.headline.c_str());
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", view.explanation.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    if (!view.usable.empty()) {
        SectionHeader("Usable");
        if (choice_ >= static_cast<int>(view.usable.size())) choice_ = 0;
        for (int i = 0; i < static_cast<int>(view.usable.size()); ++i) {
            const auto& c = view.usable[i];
            ImGui::PushID(c.path.c_str());
            const std::string label = (c.version.empty() ? std::string("Python") : "Python " + c.version) +
                                      (c.path == in_use_ ? "  (in use)" : "");
            if (ImGui::RadioButton(label.c_str(), choice_ == i)) choice_ = i;
            ImGui::Indent(ImGui::GetFrameHeight() + t.space_sm);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", c.path.c_str());
            std::string features = std::string("venv: ") + (c.venv ? "yes" : "no") + ", pip: " + (c.pip ? "yes" : "no");
            if (!c.home.empty()) features += ", home: " + c.home;
            ImGui::TextColored(t.text_dim, "%s", features.c_str());
            ImGui::PopTextWrapPos();
            ImGui::Unindent(ImGui::GetFrameHeight() + t.space_sm);
            ImGui::PopID();
        }
        ImGui::Spacing();
    }

    if (!view.unusable.empty()) {
        const std::string header = "Found but not usable (" + std::to_string(view.unusable.size()) + ")###unusable";
        ImGui::SetNextItemOpen(view.usable.empty(), ImGuiCond_Appearing);
        if (ImGui::TreeNodeEx(header.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth)) {
            for (const auto& c : view.unusable) {
                ImGui::PushID(c.path.c_str());
                ImGui::TextColored(t.text, "Python %s", c.version.c_str());
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_dim, "%s", c.path.c_str());
                StatusText(Status::Failed, c.reason.c_str());
                ImGui::PopTextWrapPos();
                ImGui::Spacing();
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        ImGui::Spacing();
    }

    if (!view.ready && view.usable.empty() && !scanning_) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(view.install_hint.c_str());
        ImGui::PopTextWrapPos();
        if (LinkButton("Open python.org downloads")) OpenUrl(kDownloads);
        ImGui::Spacing();
    }

    if (SecondaryButton(ICON_FA_FOLDER_OPEN "  Choose an interpreter...", !scanning_, "Wait for the scan to finish",
                        ButtonSize::Regular))
        Browse();
    ImGui::SameLine(0.0f, t.space_md);
    if (SecondaryButton(ICON_FA_ROTATE "  Scan again", !scanning_, "A scan is running", ButtonSize::Regular)) StartScan();
    ImGui::SameLine(0.0f, t.space_lg);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", scanning_ ? "Scanning..." : view.scan_line.c_str());
    if (!browse_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        StatusText(browse_ok_ ? Status::Verified : Status::Failed, browse_note_.c_str());
        ImGui::PopTextWrapPos();
    }

    const bool can_use = !view.usable.empty() && !scanning_;
    const bool keep = view.ready && choice_ == view.selected;
    const char* primary = keep ? "Done" : "Use this Python";
    const char* secondary = view.ready ? "Cancel" : "Continue without scripting";
    const DialogResult r = EndDialog(primary, secondary, can_use, scanning_ ? "Wait for the scan to finish" : "No usable Python yet");
    if (r == DialogResult::Primary) {
        if (!keep) {
            const std::string path = view.usable[choice_].path;
            auto& config = core::EngineConfig::Instance();
            config.SetSystemPythonPath(path);
            config.Save();
            in_use_ = path;
            if (scan_.configured_path != path) {
                scan_.configured_path = path;
                scan_.configured_path_set = true;
                scan_.configured_ok = true;
            }
            spdlog::info("Python configured: {}", path);
        }
        without_scripting_ = false;
        browse_note_.clear();
        return Outcome::Ready;
    }
    if (r == DialogResult::Secondary) {
        browse_note_.clear();
        if (view.ready) return Outcome::Ready;
        spdlog::warn("Continuing without Python: scripts, the Python console and project environments are unavailable");
        without_scripting_ = true;
        return Outcome::ContinueWithout;
    }
    return Outcome::None;
}

}  // namespace cyxwiz
