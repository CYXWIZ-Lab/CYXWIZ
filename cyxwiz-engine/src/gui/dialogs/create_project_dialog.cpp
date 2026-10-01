#include "create_project_dialog.h"

#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/start_page_presentation.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace cyxwiz {

namespace {

constexpr const char* kTitle = "Create a new project###create_project";

void Copy(char* dest, size_t size, const std::string& text) {
    std::strncpy(dest, text.c_str(), size - 1);
    dest[size - 1] = '\0';
}

std::string DefaultLocation() {
#ifdef _WIN32
    if (const char* profile = std::getenv("USERPROFILE"))
        return (std::filesystem::path(profile) / "Documents" / "CyxWiz Projects").string();
#else
    if (const char* home = std::getenv("HOME")) return (std::filesystem::path(home) / "CyxWiz Projects").string();
#endif
    return {};
}

}  // namespace

bool CreateProjectDialog::TryCreate() {
    auto& pm = ProjectManager::Instance();
    if (!pm.CreateProject(name_, location_)) {
        error_ = "The project could not be created. Check that the location is writable; details are in the log.";
        return false;
    }
    pm.GetConfig().description = startpage::ProjectTemplates()[template_index_].description;
    pm.SaveProject();
    created_path_ = pm.GetProjectFilePath();
    spdlog::info("Created project {} at {}", name_, location_);
    name_[0] = '\0';
    error_.clear();
    open_ = false;
    return true;
}

CreateProjectDialog::CreateProjectDialog() {
    Copy(location_, sizeof(location_), DefaultLocation());
}

void CreateProjectDialog::Open(int template_index) {
    SelectTemplate(template_index);
    error_.clear();
    request_open_ = true;
    focus_name_ = true;
}

void CreateProjectDialog::SelectTemplate(int index) {
    const auto& templates = startpage::ProjectTemplates();
    template_index_ = std::clamp(index, 0, static_cast<int>(templates.size()) - 1);
    if (name_[0] == '\0' && template_index_ > 0) Copy(name_, sizeof(name_), templates[template_index_].default_project_name);
}

CreateProjectDialog::Result CreateProjectDialog::Render() {
    using namespace ui;
    if (request_open_) {
        ImGui::OpenPopup(kTitle);
        request_open_ = false;
        open_ = true;
    }
    if (!open_) return Result::None;

    DialogOptions options;
    options.size = ImVec2(680.0f, 520.0f);
    options.min_size = ImVec2(520.0f, 420.0f);
    if (!BeginDialog(kTitle, options)) {
        open_ = false;
        return Result::None;
    }

    const Tokens& t = CurrentTokens();
    const auto& templates = startpage::ProjectTemplates();

    // Template cards, three per row.
    ImGui::TextColored(t.text_dim, "Template");
    if (ImGui::BeginTable("##templates", 3, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings)) {
        for (int i = 0; i < static_cast<int>(templates.size()); ++i) {
            ImGui::TableNextColumn();
            ImGui::PushID(i);
            const bool on = i == template_index_;
            const ImVec2 start = ImGui::GetCursorScreenPos();
            const float width = ImGui::GetContentRegionAvail().x;
            const float height = ImGui::GetTextLineHeight() * 3.4f + t.space_md * 2.0f;
            if (ImGui::InvisibleButton("##template", ImVec2(width, height))) SelectTemplate(i);
            const bool hovered = ImGui::IsItemHovered();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 end(start.x + width, start.y + height);
            if (on) dl->AddRectFilled(start, end, ToU32(WithAlpha(t.accent, 0.12f)), t.rounding_button);
            else if (hovered) dl->AddRectFilled(start, end, ToU32(t.hover), t.rounding_button);
            dl->AddRect(start, end, ToU32(on ? t.accent : t.border), t.rounding_button);
            const ImVec2 text_pos(start.x + t.space_md, start.y + t.space_md);
            dl->AddText(text_pos, ToU32(t.text_bright), templates[i].name);
            ImGui::PushClipRect(start, end, true);
            dl->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 0.92f,
                        ImVec2(text_pos.x, text_pos.y + ImGui::GetTextLineHeightWithSpacing()), ToU32(t.text_dim),
                        templates[i].description, nullptr, width - t.space_md * 2.0f);
            ImGui::PopClipRect();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "A template sets the project name and description. Starter graphs are added from the start page's Task starter list.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    // Fields.
    std::error_code ec;
    const std::filesystem::path target = std::filesystem::path(location_) / name_;
    const bool exists = name_[0] && location_[0] && std::filesystem::exists(target, ec);
    const startpage::CreateCheck check = startpage::CheckCreate({name_, location_, exists});

    const float label_width = ImGui::CalcTextSize("Location").x + t.space_xl;
    if (ImGui::BeginTable("##fields", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, label_width);
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
        const bool enter = ImGui::InputText("##name", name_, sizeof(name_), ImGuiInputTextFlags_EnterReturnsTrue);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Location");
        ImGui::TableNextColumn();
        const float browse_w = ButtonWidth("Browse...", ButtonSize::Small);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - browse_w - t.space_md);
        ImGui::InputText("##location", location_, sizeof(location_));
        ImGui::SameLine(0.0f, t.space_md);
        if (SecondaryButton("Browse...")) {
            if (auto folder = FileDialogs::SelectFolder("Select Project Location", location_)) Copy(location_, sizeof(location_), *folder);
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "Creates");
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(check.target.empty() ? t.text_dim : t.text, "%s", check.preview.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTable();

        submit_ = enter && check.can_create;
    }

    if (!check.warning.empty()) {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.0f);
        StatusText(Status::NeedsDriverUpdate, check.warning.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!error_.empty()) {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.0f);
        StatusText(Status::NotSupported, error_.c_str());
        ImGui::PopTextWrapPos();
    }

    // Enter in the name field creates, as the Create button does.
    if (submit_) {
        submit_ = false;
        if (TryCreate()) {
            ImGui::CloseCurrentPopup();
            EndDialog("Create", "Cancel", check.can_create, check.reason.c_str());
            return Result::Created;
        }
    }

    const DialogResult r = EndDialog("Create", "Cancel", check.can_create, check.reason.c_str());
    if (r == DialogResult::Primary) {
        if (TryCreate()) return Result::Created;
        request_open_ = true;  // EndDialog closed it; reopen to show the error
        return Result::None;
    }
    if (r == DialogResult::Secondary) {
        error_.clear();
        open_ = false;
        return Result::Cancelled;
    }
    return Result::None;
}

}  // namespace cyxwiz
