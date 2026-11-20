// Actions > Save selection as pattern... (TOFIX178 4e, approved mockup): the
// dialog that says what the selection saves as, and the notice on the canvas
// after a save. The words come from core/pattern_browser_presentation; the
// file is written by PatternLibrary::SavePatternFromSelection.
#include "node_editor.h"

#include "icons.h"
#include "separate_windows.h"
#include "patterns/pattern_library.h"
#include "ui_buttons.h"
#include "ui_widgets.h"
#include "../core/pattern_browser_presentation.h"

#include <imgui.h>

#include <cstring>
#include <filesystem>

namespace gui {

namespace {

namespace pb = cyxwiz::pattern_browser;
namespace ui = cyxwiz::ui;
using patterns::PatternCategory;

constexpr const char* kTitle = "Save selection as pattern###SaveSelectionAsPattern";

struct CategoryChoice {
    PatternCategory category;
    const char* label;
};
constexpr CategoryChoice kCategories[] = {
    {PatternCategory::Basic, "Basic"},
    {PatternCategory::CNN, "CNN"},
    {PatternCategory::RNN, "RNN"},
    {PatternCategory::Transformer, "Transformer"},
    {PatternCategory::Generative, "Generative"},
    {PatternCategory::ReinforcementLearning, "Reinforcement learning"},
    {PatternCategory::BuildingBlocks, "Blocks"},
};
constexpr int kBlocks = 6;

void Wrapped(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

}  // namespace

void NodeEditor::OpenSavePatternDialog() {
    save_pattern_.selected = selected_node_ids_;
    save_pattern_.name[0] = '\0';
    save_pattern_.description[0] = '\0';
    save_pattern_.error.clear();
    bool data_input = false, output = false;
    for (int id : selected_node_ids_) {
        if (const MLNode* node = FindNodeById(id)) {
            data_input = data_input || node->type == NodeType::DataInput;
            output = output || node->type == NodeType::Output;
        }
    }
    save_pattern_.category = data_input && output ? 0 : kBlocks;  // a full graph starts under Basic
    save_pattern_.open_request = true;
}

void NodeEditor::DrawSavePatternDialog() {
    if (save_pattern_.open_request) {
        ImGui::OpenPopup(kTitle);
        save_pattern_.open_request = false;
    }
    ui::DialogOptions options;
    options.size = ImVec2(580.0f, 440.0f);
    if (!ui::BeginDialog(kTitle, options)) return;
    const ui::Tokens& t = ui::CurrentTokens();
    auto& library = patterns::PatternLibrary::Instance();

    std::vector<pb::SelectionNode> canvas;
    std::vector<pb::SelectionLink> wires;
    for (const auto& node : nodes_)
        canvas.push_back({node.id, node.name, node.type == NodeType::DataInput, node.type == NodeType::Output});
    for (const auto& link : links_) wires.push_back({link.from_node, link.to_node});
    const std::string graph = current_file_path_.empty() ? std::string() : GetGraphStem();
    const auto summary = pb::SummarizeSelection(canvas, wires, save_pattern_.selected, graph);

    Wrapped(summary.text, t.text);
    ImGui::Spacing();
    const std::string kind = pb::KindLabel(summary.kind);
    ui::Chip(kind.c_str(), summary.kind == pb::Kind::Graph ? t.accent_text : t.text_dim);
    ImGui::SameLine();
    Wrapped(summary.kind_note, t.text_dim);

    ui::FieldLabel("Name");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    ImGui::InputTextWithHint("##name", "What to call it in My patterns", save_pattern_.name, sizeof(save_pattern_.name));
    const std::string name = save_pattern_.name;
    const bool named = name.find_first_not_of(" \t") != std::string::npos;
    const std::filesystem::path file = library.UserPatternFile(name);
    std::error_code ec;
    if (named && std::filesystem::exists(file, ec)) Wrapped(pb::NameTakenText(name), t.warning);

    ui::FieldLabel("Description");
    ImGui::InputTextMultiline("##description", save_pattern_.description, sizeof(save_pattern_.description),
                              ImVec2(-1.0f, ImGui::GetTextLineHeight() * 3.0f + ImGui::GetStyle().FramePadding.y * 2.0f));
    if (save_pattern_.description[0] == '\0')
        Wrapped("What it does and what it expects (shown in the Pattern Browser)", t.text_faint);

    ui::FieldLabel("Category");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::BeginCombo("##category", kCategories[save_pattern_.category].label)) {
        for (int i = 0; i < static_cast<int>(std::size(kCategories)); ++i) {
            if (ImGui::Selectable(kCategories[i].label, i == save_pattern_.category)) save_pattern_.category = i;
            if (i == save_pattern_.category) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::Spacing();
    Wrapped("Node settings are kept as they are on the canvas. Listed under My patterns; saved as " +
                (named ? file.string() : library.GetUserPatternsDirectory()),
            t.text_dim);
    if (!save_pattern_.error.empty()) {
        ImGui::Spacing();
        Wrapped(std::string(ICON_FA_TRIANGLE_EXCLAMATION) + "  Could not save: " + save_pattern_.error + ".", t.caution);
    }

    const auto result = ui::EndDialog("Save", "Cancel", named, "Give it a name to save", false,
                                      /*close_on_primary=*/false);
    if (result != ui::DialogResult::Primary) return;
    // Places of the selected nodes on the canvas, as drawn.
    std::map<int, ImVec2> positions;
    for (int id : save_pattern_.selected) {
        const auto at = cached_node_positions_.find(id);
        if (at != cached_node_positions_.end()) positions[id] = at->second;
    }
    std::string error;
    const std::string id = library.SavePatternFromSelection(nodes_, links_, positions, save_pattern_.selected, name,
                                                            save_pattern_.description,
                                                            kCategories[save_pattern_.category].category, &error);
    if (id.empty()) {
        save_pattern_.error = error;
        return;
    }
    save_pattern_.error.clear();
    save_pattern_.saved_id = id;
    save_pattern_.saved_text = pb::SavedText(name);
    save_pattern_.saved_until = ImGui::GetTime() + 10.0;
    if (ImGui::BeginPopupModal(kTitle)) {  // the save worked: close the dialog
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void NodeEditor::RenderSavedPatternNotice() {
    if (save_pattern_.saved_text.empty()) return;
    if (ImGui::GetTime() > save_pattern_.saved_until) {
        save_pattern_.saved_text.clear();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                   ImGuiWindowFlags_NoMove;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.card_padding);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, t.bg_raised);
    ::gui::NextWindowFollowsCurrent();
    // Above the canvas footer (node and link counts), as the minimap is.
    ImGui::SetNextWindowPos(
        ImVec2(canvas_origin_.x + 12.0f, canvas_origin_.y + canvas_size_.y - kCanvasFooterHeight - 8.0f),
        ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    if (ImGui::Begin("##SavedPatternNotice", nullptr, flags)) {
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, t.success);
        ImGui::TextUnformatted(ICON_FA_CIRCLE_CHECK);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted(save_pattern_.saved_text.c_str());
        if (open_pattern_) {
            ImGui::SameLine();
            if (ui::LinkButton("Show in Pattern Browser")) {
                open_pattern_(save_pattern_.saved_id);
                save_pattern_.saved_text.clear();
            }
        }
        // Hovering keeps it up.
        if (ImGui::IsWindowHovered()) save_pattern_.saved_until = ImGui::GetTime() + 3.0;
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

}  // namespace gui
