#include "new_script_dialog.h"

#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/start_page_presentation.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace {

constexpr const char* kTitle = "New script###new_script";

void Copy(char* dest, size_t size, const std::string& text) {
    std::strncpy(dest, text.c_str(), size - 1);
    dest[size - 1] = '\0';
}

}  // namespace

void NewScriptDialog::Open(const std::string& folder) {
    std::string start = folder;
    auto& pm = ProjectManager::Instance();
    if (start.empty() && pm.HasActiveProject()) start = pm.GetScriptsPath();
    Copy(folder_, sizeof(folder_), start);
    name_[0] = '\0';
    error_.clear();
    request_open_ = true;
    focus_name_ = true;
}

bool NewScriptDialog::TryCreate() {
    namespace fs = std::filesystem;
    const startpage::ScriptInputs in{name_, folder_, python_, false};
    const std::string file_name = startpage::ScriptFileName(in);
    const fs::path path = fs::path(folder_) / file_name;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        error_ = "The script could not be created in " + path.parent_path().string() +
                 ". Check that the folder is writable; details are in the log.";
        spdlog::error("Failed to create script: {}", path.string());
        return false;
    }
    out << startpage::ScriptTemplate(file_name, python_);
    out.close();
    created_path_ = path.string();
    spdlog::info("Created script: {}", created_path_);
    open_ = false;
    return true;
}

NewScriptDialog::Result NewScriptDialog::Render() {
    using namespace ui;
    if (request_open_) {
        ImGui::OpenPopup(kTitle);
        request_open_ = false;
        open_ = true;
    }
    if (!open_) return Result::None;

    DialogOptions options;
    options.size = ImVec2(560.0f, 360.0f);
    options.min_size = ImVec2(460.0f, 300.0f);
    if (!BeginDialog(kTitle, options)) {
        open_ = false;
        return Result::None;
    }
    const Tokens& t = CurrentTokens();

    std::error_code ec;
    const startpage::ScriptInputs probe{name_, folder_, python_, false};
    const std::string file_name = startpage::ScriptFileName(probe);
    const bool exists = !file_name.empty() && folder_[0] &&
                        std::filesystem::exists(std::filesystem::path(folder_) / file_name, ec);
    const startpage::CreateCheck check = startpage::CheckNewScript({name_, folder_, python_, exists});

    bool submit = false;
    const float label_w = ImGui::CalcTextSize("Folder").x + t.space_xl;
    if (ImGui::BeginTable("##fields", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, label_w);
        ImGui::TableSetupColumn("field", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Name");
        ImGui::TableNextColumn();
        if (focus_name_) {
            ImGui::SetKeyboardFocusHere();
            focus_name_ = false;
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        submit = ImGui::InputTextWithHint("##name", "train_model", name_, sizeof(name_), ImGuiInputTextFlags_EnterReturnsTrue);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Type");
        ImGui::TableNextColumn();
        const char* types[] = {"Python script (.py)", "CyxWiz script (.cyx)"};
        int type = python_ ? 0 : 1;
        if (SegmentedControl("##type", types, 2, &type)) python_ = type == 0;

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Folder");
        ImGui::TableNextColumn();
        const float browse_w = ButtonWidth("Browse...", ButtonSize::Small);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - browse_w - t.space_md);
        ImGui::InputTextWithHint("##folder", "Choose a folder for the script", folder_, sizeof(folder_));
        ImGui::SameLine(0.0f, t.space_md);
        if (SecondaryButton("Browse...")) {
            if (auto folder = FileDialogs::SelectFolder("Folder for the new script", folder_)) Copy(folder_, sizeof(folder_), *folder);
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "Creates");
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(check.target.empty() ? t.text_dim : t.text, "%s", check.preview.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    if (!ProjectManager::Instance().HasActiveProject())
        ImGui::TextColored(t.text_dim, "No project is open, so choose where the script goes.");
    else
        ImGui::TextColored(t.text_dim, "The project's scripts folder is the default.");
    if (!check.warning.empty()) StatusText(Status::NeedsDriverUpdate, check.warning.c_str());
    if (!error_.empty()) StatusText(Status::NotSupported, error_.c_str());
    ImGui::TextColored(t.text_dim, "The new script opens in the Script Editor.");
    ImGui::PopTextWrapPos();

    if (submit && check.can_create && TryCreate()) {
        ImGui::CloseCurrentPopup();
        EndDialog("Create and open", "Cancel", check.can_create, check.reason.c_str());
        return Result::Created;
    }
    const DialogResult r = EndDialog("Create and open", "Cancel", check.can_create, check.reason.c_str());
    if (r == DialogResult::Primary) {
        if (TryCreate()) return Result::Created;
        request_open_ = true;  // reopen to show the error
        return Result::None;
    }
    if (r == DialogResult::Secondary) {
        open_ = false;
        return Result::Cancelled;
    }
    return Result::None;
}

}  // namespace cyxwiz
