#include "pattern_browser.h"
#include "../console.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../patterns/pattern_library.h"
#include "../separate_windows.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include "../../core/recycle_bin.h"
#include <imgui.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace cyxwiz {

namespace pb = pattern_browser;
using ::gui::patterns::Pattern;
using ::gui::patterns::PatternLibrary;

namespace {

// The chip a pattern is listed under.
std::string ChipKey(const Pattern& pattern) {
    if (pattern.user) return pb::kMine;
    if (pattern.category == ::gui::patterns::PatternCategory::BuildingBlocks) return "BuildingBlocks";
    return ::gui::patterns::PatternCategoryToString(pattern.category);
}

pb::Entry ToEntry(const Pattern& pattern) {
    pb::Entry entry;
    entry.id = pattern.id;
    entry.name = pattern.name;
    entry.description = pattern.description;
    entry.category = ChipKey(pattern);
    entry.tags = pattern.tags;
    entry.kind = pattern.kind == ::gui::patterns::PatternKind::Graph ? pb::Kind::Graph : pb::Kind::Block;
    entry.needs = pattern.needs;
    entry.node_count = pattern.template_data.nodes.size();
    entry.mine = pattern.user;
    for (const auto& node : pattern.template_data.nodes)
        if (node.name == "Block input") entry.block_input = true;
    return entry;
}

// `text` cut to `lines` wrapped lines at `width`, ending in "..." when cut.
std::string ClampLines(const std::string& text, float width, int lines) {
    const float limit = ImGui::GetTextLineHeight() * static_cast<float>(lines) + 1.0f;
    const auto height = [width](const std::string& s) {
        return ImGui::CalcTextSize(s.c_str(), nullptr, false, width).y;
    };
    if (height(text) <= limit) return text;
    size_t low = 0, high = text.size();
    while (low < high) {  // the longest prefix that fits with "..."
        const size_t mid = (low + high + 1) / 2;
        if (height(text.substr(0, mid) + "...") <= limit) low = mid;
        else high = mid - 1;
    }
    std::string cut = text.substr(0, low);
    if (const auto space = cut.find_last_of(' '); space != std::string::npos && space > low / 2) cut.resize(space);
    while (!cut.empty() && (cut.back() == ' ' || cut.back() == ',' || cut.back() == '.')) cut.pop_back();
    return cut + "...";
}

void WrappedText(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void DimText(const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

// An icon and wrapped text in one colour (the result and failure lines).
void StatusLine(const char* icon, const std::string& text, const ImVec4& colour) {
    WrappedText(std::string(icon) + "  " + text, colour);
}

// A raised area inside a card (the placement question, the delete question).
void BeginInset(const char* id) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.card_padding);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    ImGui::BeginChild(id, ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
}

void EndInset() {
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

double ParseNumber(const std::string& text) { return std::strtod(text.c_str(), nullptr); }

}  // namespace

PatternBrowserPanel::PatternBrowserPanel() : Panel("Pattern Browser", false) {
    PatternLibrary::Instance().Initialize();
}

void PatternBrowserPanel::RebuildEntries() {
    auto& library = PatternLibrary::Instance();
    if (entries_revision_ == library.Revision()) return;
    entries_.clear();
    for (const auto& pattern : library.GetAllPatterns()) entries_.push_back(ToEntry(pattern));
    entries_revision_ = library.Revision();
    if (!open_id_.empty() && !library.GetPattern(open_id_)) open_id_.clear();  // deleted
}

void PatternBrowserPanel::Render() {
    if (!visible_) return;
    RebuildEntries();

    ImGui::SetNextWindowSize(ImVec2(440, 680), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_NoCollapse)) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
        ui::SearchField("##pattern_search", search_, sizeof(search_), "Search names, descriptions and tags", 0.0f,
                        ImGuiInputTextFlags_EscapeClearsAll);
        ImGui::Spacing();
        const auto view = pb::List(entries_, chip_, search_);
        RenderChips(view);
        RenderList(view);
    }
    ImGui::End();
}

void PatternBrowserPanel::RenderChips(const pb::ListView& view) {
    ui::FlowRow row(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX());
    for (const auto& chip : pb::Chips(entries_)) {
        const std::string count = std::to_string(chip.count);
        row.Next(ui::FilterChipWidth(chip.label.c_str(), count.c_str()));
        // While searching no chip is chosen: the search covers them all.
        if (ui::FilterChip(chip.key.c_str(), chip.label.c_str(), count.c_str(),
                           !view.searching && chip_ == chip.key)) {
            chip_ = chip.key;
            search_[0] = '\0';
            open_id_.clear();
            SetStage({}, Stage::None);
        }
    }
    if (view.searching) DimText(view.note);
    ImGui::Spacing();
}

void PatternBrowserPanel::RenderList(const pb::ListView& view) {
    const float footer = view.on_mine ? ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y : 0.0f;
    ImGui::BeginChild("##patterns", ImVec2(0.0f, -footer), ImGuiChildFlags_None);
    auto& library = PatternLibrary::Instance();
    if (!view.empty_text.empty()) {
        ImGui::Spacing();
        WrappedText(view.empty_text, ui::CurrentTokens().text_dim);
    }
    for (const auto* entry : view.ready)
        if (const Pattern* pattern = library.GetPattern(entry->id)) RenderCard(*pattern, *entry);
    if (!view.not_ready.empty()) {
        ui::SettingsSection(view.not_ready_heading.c_str());
        for (const auto* entry : view.not_ready)
            if (const Pattern* pattern = library.GetPattern(entry->id)) RenderCard(*pattern, *entry);
    }
    ImGui::EndChild();
    if (view.on_mine) RenderFolderLine();
}

void PatternBrowserPanel::RenderCard(const Pattern& pattern, const pb::Entry& entry) {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool open = open_id_ == pattern.id;
    const bool ready = entry.needs.empty();
    // Copies: a Delete below removes the pattern the references point at.
    const std::string id = pattern.id;
    ui::BeginCard(id.c_str());

    // The header: a click opens or closes the card.
    ImGui::BeginGroup();
    ImGui::PushStyleColor(ImGuiCol_Text, ready ? t.text_bright : t.text_dim);
    ImGui::TextUnformatted(pattern.name.c_str());
    ImGui::PopStyleColor();
    const std::string kind = pb::KindLabel(entry.kind);
    ImGui::SameLine();
    ui::Chip(kind.c_str(), !ready ? t.text_faint : entry.kind == pb::Kind::Graph ? t.accent_text : t.text_dim);
    if (!ready) {
        ImGui::SameLine();
        ui::Chip("Not ready", t.warning);
    }
    const std::string nodes = pb::NodesLabel(entry.node_count);
    ui::SameLineRight(ImGui::CalcTextSize(nodes.c_str()).x);
    DimText(nodes);
    if (!open && !pattern.description.empty())
        WrappedText(ClampLines(pattern.description, ImGui::GetContentRegionAvail().x, 2),
                    ready ? t.text_dim : t.text_faint);
    ImGui::EndGroup();
    bool toggle = false;
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (!ready) ui::Tooltip(pb::NotReadyText(entry.needs).c_str());
        toggle = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    }
    if (open && scroll_to_open_) {
        ImGui::SetScrollHereY(0.0f);
        scroll_to_open_ = false;
    }

    if (open) {
        // A pattern of yours may have no description: no empty line for it.
        if (!pattern.description.empty()) {
            ImGui::Spacing();
            WrappedText(pattern.description, t.text);
        }
        if (!ready) {
            ImGui::Spacing();
            StatusLine(ICON_FA_CIRCLE_EXCLAMATION, pb::NotReadyText(entry.needs), t.warning);
        }
        RenderSettings(pattern);
        ImGui::Spacing();
        RenderInsertLine(pattern, entry);
        RenderStage(pattern, entry);
    }
    ui::EndCard();
    if (toggle) {
        if (open) open_id_.clear();
        else if (const Pattern* still = PatternLibrary::Instance().GetPattern(id)) Open(*still);
    }
}

void PatternBrowserPanel::RenderSettings(const Pattern& pattern) {
    if (pattern.parameters.empty()) return;
    using ::gui::patterns::ParameterType;
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Settings");
    for (const auto& param : pattern.parameters) {
        ImGui::PushID(param.name.c_str());
        auto& value = params_[param.name];
        const bool is_number = param.type == ParameterType::Int || param.type == ParameterType::Float;
        const bool is_float = param.type == ParameterType::Float;

        ImGui::BeginGroup();
        ImGui::TextUnformatted(param.name.c_str());
        if (!param.description.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
            ImGui::TextUnformatted(param.description.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();

        if (is_number) {
            ui::NumberFieldOptions options;
            options.min = param.min_value.empty() ? -1e300 : ParseNumber(param.min_value);
            options.max = param.max_value.empty() ? 1e300 : ParseNumber(param.max_value);
            options.step = pb::NumberStep(is_float, param.default_value);
            options.format = is_float ? "%g" : "%.0f";
            options.width = 72.0f;
            options.hint = param.description.empty() ? nullptr : param.description.c_str();
            const float steps = 2.0f * (ui::ButtonWidth("+", ui::ButtonSize::Small) + ImGui::GetStyle().ItemSpacing.x);
            ui::SameLineRight(options.width + steps);
            double number = ParseNumber(value);
            if (ui::NumberField("##value", &number, options)) value = pb::FormatNumber(number, is_float);
        } else if (param.type == ParameterType::Bool) {
            bool on = value == "true" || value == "1";
            const char* word = on ? "On##value" : "Off##value";
            ui::SameLineRight(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x +
                              ImGui::CalcTextSize("Off").x);
            if (ImGui::Checkbox(word, &on)) value = on ? "true" : "false";
        } else if (!param.options.empty()) {
            float width = 0.0f;
            for (const auto& option : param.options) width = std::max(width, ImGui::CalcTextSize(option.c_str()).x);
            width += ImGui::GetFrameHeight() + ImGui::GetStyle().FramePadding.x * 2.0f;
            ui::SameLineRight(width);
            ImGui::SetNextItemWidth(width);
            if (ImGui::BeginCombo("##value", value.c_str())) {
                for (const auto& option : param.options) {
                    if (ImGui::Selectable(option.c_str(), option == value)) value = option;
                    if (option == value) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        } else {
            constexpr float kTextWidth = 160.0f;
            ui::SameLineRight(kTextWidth);
            ImGui::SetNextItemWidth(kTextWidth);
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer), "%s", value.c_str());
            if (ImGui::InputText("##value", buffer, sizeof(buffer))) value = buffer;
        }
        ImGui::PopID();
    }
}

void PatternBrowserPanel::RenderInsertLine(const Pattern& pattern, const pb::Entry& entry) {
    // While a question is open its buttons are the card's actions.
    if (stage_id_ == pattern.id && (stage_ == Stage::Choose || stage_ == Stage::ConfirmDelete)) return;
    const bool ready = entry.needs.empty();
    const std::string label = pb::InsertLabel(entry.kind);
    if (ui::PrimaryButton(label.c_str(), ready, ready ? nullptr : entry.needs.c_str())) {
        Insert(pattern, entry, std::nullopt);
        return;
    }
    if (ready && node_editor_) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        DimText(pb::InsertHint(entry.kind, node_editor_->GetNodes().size()));
    }
    if (entry.mine) {
        const float show = ui::ButtonWidth("Show in folder", ui::ButtonSize::Small);
        const float remove = ui::ButtonWidth("Delete", ui::ButtonSize::Small);
        ui::SameLineRight(show + remove + ImGui::GetStyle().ItemSpacing.x);
        if (ui::LinkButton("Show in folder")) ::gui::OpenPathWithSystem(pattern.file_path, true);
        ImGui::SameLine();
        if (ui::DangerButton("Delete")) SetStage(pattern.id, Stage::ConfirmDelete);
    }
}

void PatternBrowserPanel::RenderStage(const Pattern& pattern, const pb::Entry& entry) {
    if (stage_ == Stage::Done && node_editor_ && node_editor_->GetNodes().size() != done_canvas_nodes_)
        SetStage({}, Stage::None);
    if (stage_id_ != pattern.id || stage_ == Stage::None) return;
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    switch (stage_) {
        case Stage::Choose: {
            const std::string graph = GraphName();
            const size_t canvas_nodes = node_editor_ ? node_editor_->GetNodes().size() : 0;
            BeginInset("##choose");
            WrappedText(pb::PlacementQuestion(canvas_nodes, graph), t.text);
            ImGui::Spacing();
            std::optional<Where> where;
            if (ui::PrimaryButton("Replace the canvas")) where = Where::Replace;
            ImGui::SameLine();
            if (ui::SecondaryButton("Add beside it", true, nullptr, ui::ButtonSize::Regular)) where = Where::Beside;
            ImGui::SameLine();
            if (ui::LinkButton("Cancel")) SetStage({}, Stage::None);
            ImGui::Spacing();
            WrappedText(pb::ReplaceNote(graph), t.text_dim);
            EndInset();
            if (where) Insert(pattern, entry, where);
            break;
        }
        case Stage::ConfirmDelete: {
            BeginInset("##delete");
            WrappedText(pb::DeleteQuestion(pattern.name, RecycleBinName()), t.text);
            ImGui::Spacing();
            const bool confirmed = ui::DangerButton("Delete", true, nullptr, ui::ButtonSize::Regular);
            ImGui::SameLine();
            if (ui::LinkButton("Cancel")) SetStage({}, Stage::None);
            EndInset();
            if (confirmed) {
                // Copies: the pattern is gone after the call.
                const std::string id = pattern.id, name = pattern.name;
                std::string error;
                if (PatternLibrary::Instance().DeleteUserPattern(id, &error)) {
                    open_id_.clear();
                    SetStage({}, Stage::None);
                } else {
                    SetStage(id, Stage::Failed, "Could not delete " + name + ": " + error + ".");
                }
            }
            break;
        }
        case Stage::Done: StatusLine(ICON_FA_CIRCLE_CHECK, stage_text_, t.success); break;
        case Stage::Failed: StatusLine(ICON_FA_TRIANGLE_EXCLAMATION, stage_text_, t.caution); break;
        case Stage::None: break;
    }
}

void PatternBrowserPanel::RenderFolderLine() {
    const std::string folder = PatternLibrary::Instance().GetUserPatternsDirectory();
    ImGui::AlignTextToFramePadding();
    DimText("Saved in " + folder);
    ImGui::SameLine();
    if (ui::LinkButton("Open folder")) {
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);  // it exists from the first save on
        ::gui::OpenPathWithSystem(folder, false);
    }
}

void PatternBrowserPanel::Open(const Pattern& pattern) {
    open_id_ = pattern.id;
    params_.clear();
    for (const auto& param : pattern.parameters) params_[param.name] = param.default_value;
    SetStage({}, Stage::None);
}

void PatternBrowserPanel::SetStage(const std::string& id, Stage stage, std::string text) {
    stage_id_ = id;
    stage_ = stage;
    stage_text_ = std::move(text);
}

std::string PatternBrowserPanel::GraphName() const {
    if (!node_editor_ || node_editor_->GetCurrentFilePath().empty()) return {};
    return node_editor_->GetGraphStem();
}

void PatternBrowserPanel::Insert(const Pattern& pattern, const pb::Entry& entry, std::optional<Where> where) {
    if (!node_editor_) return;
    const size_t canvas_nodes = node_editor_->GetNodes().size();
    if (!where && pb::NeedsPlacementChoice(entry.kind, canvas_nodes)) {
        SetStage(pattern.id, Stage::Choose);
        return;
    }
    std::vector<::gui::MLNode> nodes;
    std::vector<::gui::NodeLink> links;
    // Ids only need to be unique within the pattern: the canvas gives fresh ones.
    int next_node_id = 1;
    int next_link_id = 1;
    ::gui::patterns::NodeCreatorCallback creator = [this](::gui::NodeType type, const std::string& name) {
        ::gui::MLNode node = node_editor_->CreateNode(type, name);
        node_editor_->ApplyUserNodeDefaults(node);
        return node;
    };
    std::string error;
    if (!PatternLibrary::Instance().InstantiatePatternWithCreator(pattern.id, params_, nodes, links, next_node_id,
                                                                  next_link_id, ImVec2(0.0f, 0.0f), creator, &error)) {
        SetStage(pattern.id, Stage::Failed, pb::FailText(pattern.name, error));
        return;
    }
    const std::string graph = GraphName();
    using Placement = ::gui::NodeEditor::PatternPlacement;
    const Placement placement = where == Where::Replace  ? Placement::Replace
                                : where == Where::Beside ? Placement::Beside
                                                         : Placement::Below;
    error = node_editor_->InsertPattern(nodes, links, placement);
    if (!error.empty()) {
        SetStage(pattern.id, Stage::Failed, pb::FailText(pattern.name, error));
        return;
    }
    const pb::Placement said = canvas_nodes == 0         ? pb::Placement::Empty
                               : where == Where::Replace ? pb::Placement::Replace
                               : where == Where::Beside  ? pb::Placement::Beside
                                                         : pb::Placement::Below;
    SetStage(pattern.id, Stage::Done, pb::DoneText(entry, said, graph));
    done_canvas_nodes_ = node_editor_->GetNodes().size();
}

void PatternBrowserPanel::OpenWithPattern(const std::string& pattern_id) {
    Show();
    ::gui::BringToFront(GetName());
    RebuildEntries();
    const Pattern* pattern = PatternLibrary::Instance().GetPattern(pattern_id);
    if (!pattern) return;
    chip_ = ChipKey(*pattern);
    search_[0] = '\0';
    Open(*pattern);
    scroll_to_open_ = true;
}

}  // namespace cyxwiz
