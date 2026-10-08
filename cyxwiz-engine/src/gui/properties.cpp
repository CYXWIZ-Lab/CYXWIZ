// The Properties panel (TOFIX129 A7). Draws the PropertiesView of
// core/properties_presentation: header, Settings (the node's parameters with
// their truth on each row), the AS COMPILED and fusion cards, Advanced. The
// view is rebuilt only when the node, the graph, the compile or an edit
// changes; every other frame draws the cached one.
#include "properties.h"
#include "properties_contract.h"
#include "properties_executor.h"
#include "properties_node_editors.h"
#include "properties_parameter_rules.h"
#include "properties_rows.h"
#include "properties_truth.h"
#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/extension_node_metadata.h"
#include "../core/file_dialogs.h"
#include "../core/live_graph_compile.h"
#include "../core/node_executors/node_executor_factory.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/dashboard/dashboard_model.h"
#include "../core/plot/plot_model.h"
#include "../core/transformer_configuration_policy.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_fonts.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "node_editor.h"
#include "node_config_dialog.h"
#include <imgui.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gui {

namespace {

using cyxwiz::properties_view::ChipKind;
using cyxwiz::properties_view::SettingsRoute;

std::string ParamOrEmpty(const MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

properties_truth::DatasetTruthFact BuildDatasetTruthFact(const MLNode& node) {
    properties_truth::DatasetTruthFact fact;
    fact.dataset_name = ParamOrEmpty(node, "dataset_name");
    if (fact.dataset_name.empty()) {
        fact.dataset_name = ParamOrEmpty(node, "dataset");
    }
    if (fact.dataset_name.empty()) {
        return fact;
    }

    auto& registry = cyxwiz::DataRegistry::Instance();
    if (const auto* text = registry.GetTextDatasetEntry(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "TextDataset";
        fact.rows = text->num_samples;
        fact.has_labels = text->has_labels;
        fact.has_label_column_metadata = !text->label_column.empty();
        fact.label_column = text->label_column;
        fact.has_class_count = true;
        fact.class_count = text->num_classes;
        return fact;
    }

    if (auto arrow_dataset = registry.GetArrowDataset(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "Arrow";
        fact.rows = static_cast<size_t>(std::max<int64_t>(0, arrow_dataset->GetNumRows()));
        fact.columns = arrow_dataset->GetColumnNames();
        fact.has_labels = !ParamOrEmpty(node, "label_column").empty() ||
                          !ParamOrEmpty(node, "text_label_column").empty();
        return fact;
    }

    if (auto parquet_dataset = registry.GetParquetBackedDataset(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "Parquet";
        fact.rows = static_cast<size_t>(std::max<int64_t>(0, parquet_dataset->GetNumRows()));
        fact.columns = parquet_dataset->GetColumnNames();
        fact.has_labels = !ParamOrEmpty(node, "label_column").empty() ||
                          !ParamOrEmpty(node, "text_label_column").empty();
        return fact;
    }

    if (const auto* image = registry.GetImageDatasetEntry(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "ImageDataset";
        fact.rows = image->num_images;
        fact.has_labels = true;
        fact.has_class_count = true;
        fact.class_count = image->num_classes;
        return fact;
    }

    if (const auto* audio = registry.GetAudioDatasetEntry(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "AudioDataset";
        fact.rows = audio->num_samples;
        fact.has_labels = audio->labeled_subdirs || !audio->label_col.empty();
        fact.has_label_column_metadata = !audio->label_col.empty();
        fact.label_column = audio->label_col;
        fact.has_class_count = true;
        fact.class_count = audio->num_classes;
        return fact;
    }

    if (registry.HasDataset(fact.dataset_name)) {
        auto handle = registry.GetDataset(fact.dataset_name);
        if (handle.IsValid()) {
            const auto info = handle.GetInfo();
            fact.found = true;
            fact.backing_store = "Dataset";
            fact.rows = info.num_samples;
            fact.has_labels = info.num_classes > 0;
            fact.has_class_count = true;
            fact.class_count = info.num_classes;
            return fact;
        }
    }

    fact.message = "Dataset '" + fact.dataset_name + "' is not loaded.";
    return fact;
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// A section header in the panel: a tinted bar with a triangle, remembering
// its open state per panel (not per node).
bool Section(const char* label, bool& open) {
    ImGui::SetNextItemOpen(open, ImGuiCond_Always);
    const bool now_open = ImGui::CollapsingHeader(label);
    open = now_open;
    return now_open;
}

}  // namespace

Properties::Properties() : show_window_(true) {
}

Properties::~Properties() = default;

void Properties::SetSelectedNode(MLNode* node) {
    selected_node_ = node;
}

bool Properties::ConfigureNode(MLNode* node) {
    if (!node) {
        return false;
    }
    // A Plot node opens its Plot window, owned by the node editor.
    if ((node->type == NodeType::Plot || node->type == NodeType::Dashboard) && node_editor_) {
        node_editor_->OpenNodeConfiguration(node->id);
        return true;
    }
    selected_node_ = node;
    show_window_ = true;

    auto& factory = NodeConfigDialogFactory::Instance();
    if (!factory.HasDialog(node->type)) {
        return false;
    }
    active_dialog_ = factory.CreateDialog(node);
    if (!active_dialog_) {
        return false;
    }
    active_dialog_->SetNodeEditor(node_editor_);
    active_dialog_->Open();
    return true;
}

void Properties::ClearSelection() {
    selected_node_ = nullptr;
}

void Properties::ClearNodeReferences() {
    selected_node_ = nullptr;
    active_dialog_.reset();
    view_valid_ = false;
}

void Properties::SetBackendPlacementFacts(
    std::vector<properties_truth::BackendPlacementTruthFact> facts) {
    backend_placement_facts_ = std::move(facts);
    ++facts_serial_;
}

void Properties::ClearBackendPlacementFacts() {
    backend_placement_facts_.clear();
    ++facts_serial_;
}

// ---------------------------------------------------------------------------
// The view

void Properties::RefreshDatasetFacts() {
    // Registry probes (up to seven locks per Data Input) at most once a second.
    const auto now = std::chrono::steady_clock::now();
    if (facts_time_ != std::chrono::steady_clock::time_point{} && now - facts_time_ < std::chrono::seconds(1)) return;
    facts_time_ = now;
    std::vector<properties_truth::DatasetTruthFact> facts;
    if (node_editor_) {
        for (const auto& graph_node : node_editor_->GetNodes()) {
            if (graph_node.type != NodeType::DataInput) continue;
            auto fact = BuildDatasetTruthFact(graph_node);
            if (!fact.dataset_name.empty()) facts.push_back(std::move(fact));
        }
    } else if (selected_node_ && selected_node_->type == NodeType::DataInput) {
        facts.push_back(BuildDatasetTruthFact(*selected_node_));
    }
    // Only a change bumps the serial, so the view is not rebuilt every second.
    bool same = facts.size() == dataset_facts_.size();
    for (size_t i = 0; same && i < facts.size(); ++i) {
        const auto& a = facts[i];
        const auto& b = dataset_facts_[i];
        same = a.dataset_name == b.dataset_name && a.found == b.found && a.rows == b.rows &&
               a.columns.size() == b.columns.size() && a.backing_store == b.backing_store &&
               a.class_count == b.class_count && a.label_column == b.label_column;
    }
    if (!same) {
        dataset_facts_ = std::move(facts);
        ++facts_serial_;
    }
}

void Properties::RefreshView(MLNode& node) {
    RefreshDatasetFacts();
    cyxwiz::properties_view::Key key;
    key.node_id = node.id;
    key.graph_revision = live_compile_ ? live_compile_->GraphRevision() : 0;
    key.compile_serial = live_compile_ ? live_compile_->ResultSerial() + (live_compile_->ParametersCounting() ? 1u << 20 : 0) +
                                             (live_compile_->ParametersCounted() ? 1u << 21 : 0) +
                                             (static_cast<uint64_t>(live_compile_->State()) << 24)
                                       : 0;
    key.edit_serial = edit_serial_;
    key.facts_serial = facts_serial_;
    if (view_valid_ && key == view_key_) return;

    view_metadata_ = cyxwiz::ResolveNodeMetadata(node);
    const cyxwiz::NodeMetadata* metadata = view_metadata_ ? &*view_metadata_ : nullptr;

    properties_truth::NodeTruthContext truth_context;
    if (node_editor_) {
        truth_context.nodes = &node_editor_->GetNodes();
        truth_context.links = &node_editor_->GetLinks();
    }
    if (!backend_placement_facts_.empty()) truth_context.backend_placements = &backend_placement_facts_;
    if (!dataset_facts_.empty()) truth_context.dataset_facts = &dataset_facts_;

    cyxwiz::properties_view::Inputs in;
    in.node = &node;
    in.metadata = metadata;
    in.nodes = node_editor_ ? &node_editor_->GetNodes() : nullptr;
    in.links = node_editor_ ? &node_editor_->GetLinks() : nullptr;
    in.truth = properties_truth::ResolveNodeTruth(node, truth_context);
    const std::string dataset_name = ParamOrEmpty(node, "dataset_name").empty() ? ParamOrEmpty(node, "dataset") : ParamOrEmpty(node, "dataset_name");
    for (const auto& fact : dataset_facts_) {
        if (fact.dataset_name == dataset_name) in.dataset = &fact;
    }
    in.has_dialog = ShouldShowOpenDialogButton(node.type);
    in.has_executor = cyxwiz::NodeExecutorFactory::Instance().HasExecutor(node.type);
    in.custom_editor = properties_contract::IsCustomPropertiesNode(metadata);
    in.dialog_only = properties_contract::IsDialogOnlyPropertiesNode(metadata);

    cyxwiz::CompiledNodeInputs compiled;
    if (node_editor_) {
        compiled.type_label = metadata ? metadata->name : std::string();
        if (live_compile_) {
            compiled.state = live_compile_->State();
            compiled.config = live_compile_->Config();
            compiled.layer_parameters = live_compile_->LayerParameters();
            compiled.parameters_counted = live_compile_->ParametersCounted();
            compiled.parameters_counting = live_compile_->ParametersCounting();
        }
        compiled.data_loaded = std::any_of(dataset_facts_.begin(), dataset_facts_.end(),
                                           [](const properties_truth::DatasetTruthFact& f) { return f.found; });
        in.compiled = &compiled;
    }

    view_ = cyxwiz::properties_view::Build(in);
    view_key_ = key;
    view_valid_ = true;
}

// ---------------------------------------------------------------------------
// Render

void Properties::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Properties", &show_window_)) {
        if (!selected_node_) {
            RenderEmpty();
        } else {
            MLNode& node = *selected_node_;
            RefreshView(node);
            shown_truth_keys_.clear();
            properties_rows::BeginFrame(&view_, &shown_truth_keys_, settings_details_open_.count(node.id) > 0);

            // Scope every widget below by node: parameter widgets are keyed by
            // parameter name, so without this two nodes of the same type share
            // widget IDs, and a text box still active when the selection
            // changes writes its buffer into the newly selected node.
            ImGui::PushID(node.id);
            RenderHeader(node);
            RenderSettings(node);
            RenderCompiledCard(node);
            RenderFusionCard(node);
            RenderAdvanced(node);
            if (view_.has_executor) {
                ImGui::Spacing();
                RenderExecutorSection(node);
            }
            ImGui::PopID();
            properties_rows::BeginFrame(nullptr, nullptr, false);
        }
    }
    ImGui::End();

    // Render active configuration dialog (if open)
    if (active_dialog_ && active_dialog_->IsOpen()) {
        if (!active_dialog_->Render()) {
            // Dialog was closed
            active_dialog_.reset();
        }
    }
}

void Properties::RenderEmpty() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    cyxwiz::ui::EmptyState(ICON_FA_SLIDERS, "Select a node to edit its properties",
                           "Double-click a node to open its dialog.");
    if (!node_editor_) return;
    // The graph in one line (A7-4).
    const auto e = cyxwiz::properties_view::BuildEmpty(
        node_editor_->GetCurrentFilePath(), node_editor_->GetNodes().size(), node_editor_->GetLinks().size(),
        live_compile_ ? live_compile_->State() : cyxwiz::LiveCompileState::NotCompiled,
        live_compile_ ? live_compile_->Config() : nullptr, &node_editor_->GetNodes());
    if (e.nodes == 0) return;
    ImGui::Dummy(ImVec2(0.0f, t.space_xl));
    std::string line = e.graph_name + " \xC2\xB7 " + std::to_string(e.nodes) + (e.nodes == 1 ? " node, " : " nodes, ") +
                       std::to_string(e.links) + (e.links == 1 ? " link" : " links");
    float width = ImGui::CalcTextSize(line.c_str()).x;
    ImGui::SetCursorPosX(std::max(0.0f, (ImGui::GetWindowContentRegionMax().x - width) * 0.5f));
    ImGui::TextColored(t.text_dim, "%s", line.c_str());
    if (e.status.empty()) return;
    const ImVec4 colour = e.kind == cyxwiz::CompiledStatusKind::Ok ? t.success
                          : e.kind == cyxwiz::CompiledStatusKind::Failed ? t.caution
                          : e.kind == cyxwiz::CompiledStatusKind::Pending ? t.running : t.pending;
    width = cyxwiz::ui::ChipWidth(e.status.c_str()) + (e.first_error.empty() ? 0.0f : t.space_sm + ImGui::CalcTextSize(e.first_error.c_str()).x);
    ImGui::SetCursorPosX(std::max(0.0f, (ImGui::GetWindowContentRegionMax().x - width) * 0.5f));
    cyxwiz::ui::Chip(e.status.c_str(), colour);
    if (!e.first_error.empty()) {
        ImGui::SameLine(0.0f, t.space_sm);
        ImGui::TextColored(t.text_dim, "%s", e.first_error.c_str());
    }
}

// ---------------------------------------------------------------------------
// Header: icon, name, one muted line, the primary action and Actions.

void Properties::RenderHeader(MLNode& node) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    const auto& h = view_.header;

    // Name: an input that looks like a heading; saved on Enter or when it
    // loses focus (not on every keystroke).
    if (name_buffer_node_ != node.id) {
        std::strncpy(name_buffer_, node.name.c_str(), sizeof(name_buffer_) - 1);
        name_buffer_[sizeof(name_buffer_) - 1] = '\0';
        name_buffer_node_ = node.id;
    }
    if (!h.icon.empty()) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.accent_text, "%s", h.icon.c_str());
        ImGui::SameLine(0.0f, t.space_md);
    }
    {
        cyxwiz::ui::FontScope medium(cyxwiz::ui::Font::Medium);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, t.bg_input);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, t.bg_input);
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_bright);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##node_name", name_buffer_, sizeof(name_buffer_), ImGuiInputTextFlags_EnterReturnsTrue);
        const bool commit = ImGui::IsItemDeactivatedAfterEdit();
        ImGui::PopStyleColor(4);
        cyxwiz::ui::Tooltip("Click to rename; Enter or clicking away saves it");
        if (commit && node.name != name_buffer_) {
            node.name = name_buffer_;
            NotifyEdited();
        }
    }

    // type · category · #id · status · badge
    ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s \xC2\xB7 #%d", h.type_name.c_str(), h.category.c_str(), h.id);
    if (!h.implementation.empty()) {
        ImGui::SameLine(0.0f, t.space_sm);
        if (h.implementation_kind == ChipKind::Ok) {
            ImGui::TextColored(t.text_faint, "%s", h.implementation.c_str());
        } else {
            cyxwiz::ui::Chip(h.implementation.c_str(), properties_rows::ChipColour(h.implementation_kind));
        }
        cyxwiz::ui::Tooltip("Implementation status of this node type");
    }
    if (!h.badge.empty()) {
        ImGui::SameLine(0.0f, t.space_sm);
        cyxwiz::ui::Chip(h.badge.c_str(), t.text_dim);
    }

    ImGui::Spacing();
    if (!h.primary_label.empty()) {
        if (cyxwiz::ui::PrimaryButton(h.primary_label.c_str(), true, nullptr, cyxwiz::ui::ButtonSize::Small)) {
            ConfigureNode(&node);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(h.primary_tip.c_str());
            ImGui::TextColored(t.text_dim, "%s", h.primary_tip_detail.c_str());
            ImGui::EndTooltip();
        }
        ImGui::SameLine(0.0f, t.space_md);
    }
    if (cyxwiz::ui::SecondaryButton("Actions " ICON_FA_CARET_DOWN)) ImGui::OpenPopup("##actions");
    RenderActionsMenu(node);
    ImGui::Spacing();
}

const cyxwiz::node_presets::Store& Properties::Presets() {
    if (!presets_loaded_) {
        presets_loaded_ = true;
        if (!presets_.Load(cyxwiz::node_presets::DefaultStoreFile(), &presets_error_)) {
            spdlog::warn("Properties: {}", presets_error_);
        }
    }
    return presets_;
}

void Properties::RenderActionsMenu(MLNode& node) {
    if (!ImGui::BeginPopup("##actions")) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    const std::string& type_name = view_.header.type_name;
    const auto presets = cyxwiz::node_presets::AllPresets(node.type, type_name, Presets());
    if (ImGui::BeginMenu("Apply preset", !presets.empty())) {
        for (const auto& preset : presets) {
            ImGui::PushID(preset.name.c_str());
            if (ImGui::MenuItem(preset.name.c_str())) {
                cyxwiz::node_presets::Apply(preset, node);
                NotifyEdited();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                for (const auto& [k, v] : preset.parameters) ImGui::Text("%s = %s", k.c_str(), v.c_str());
                if (!preset.builtin) ImGui::TextColored(t.text_dim, "Saved preset");
                ImGui::EndTooltip();
            }
            ImGui::PopID();
        }
        ImGui::EndMenu();
    }
    if (presets.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("No presets for %s yet: save the current settings as one", type_name.c_str());
    }
    if (ImGui::BeginMenu("Save current as preset...")) {
        ImGui::SetNextItemWidth(180.0f);
        const bool enter = ImGui::InputText("##preset_name", preset_name_buffer_, sizeof(preset_name_buffer_),
                                            ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if ((cyxwiz::ui::PrimaryButton("Save", preset_name_buffer_[0] != '\0', "Give the preset a name",
                                       cyxwiz::ui::ButtonSize::Small) || enter) && preset_name_buffer_[0] != '\0') {
            Presets();
            presets_.Put(type_name, preset_name_buffer_, cyxwiz::node_presets::ParametersToSave(node));
            if (!presets_.Save(cyxwiz::node_presets::DefaultStoreFile(), &presets_error_)) {
                spdlog::warn("Properties: {}", presets_error_);
            }
            preset_name_buffer_[0] = '\0';
            ImGui::CloseCurrentPopup();
        }
        if (!presets_error_.empty()) ImGui::TextColored(t.error, "%s", presets_error_.c_str());
        ImGui::EndMenu();
    }
    std::vector<cyxwiz::node_presets::Preset> saved = Presets().PresetsFor(type_name);
    if (ImGui::BeginMenu("Delete saved preset", !saved.empty())) {
        for (const auto& preset : saved) {
            if (ImGui::MenuItem(preset.name.c_str())) {
                presets_.Remove(type_name, preset.name);
                presets_.Save(cyxwiz::node_presets::DefaultStoreFile(), &presets_error_);
            }
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();
    const bool has_defaults = view_metadata_ && !view_metadata_->parameters.empty();
    if (ImGui::MenuItem("Reset all settings to defaults", nullptr, false, has_defaults)) {
        for (const auto& param : view_metadata_->parameters) {
            if (param.default_value.empty()) node.parameters.erase(param.name);
            else node.parameters[param.name] = param.default_value;
        }
        NotifyEdited();
    }
    if (ImGui::MenuItem("Copy settings as JSON")) {
        nlohmann::ordered_json j;
        j["name"] = node.name;
        j["type"] = view_.header.type_name;
        j["parameters"] = node.parameters;
        ImGui::SetClipboardText(j.dump(2).c_str());
    }
    std::vector<std::string> unused;
    for (const auto& raw : view_.raw) {
        if (raw.cleanup_allowed) unused.push_back(raw.key);
    }
    if (ImGui::MenuItem("Remove unused keys", nullptr, false, !unused.empty())) {
        for (const auto& key : unused) node.parameters.erase(key);
        NotifyEdited();
    }
    if (unused.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Nothing to remove: every stored key maps to a setting");
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Show node in canvas", nullptr, false, node_editor_ != nullptr)) {
        node_editor_->FocusNode(node.id);
    }
    ImGui::EndPopup();
}

// ---------------------------------------------------------------------------
// Settings

void Properties::RenderSettings(MLNode& node) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    if (!Section("Settings", section_settings_open_)) return;

    switch (view_.route) {
        case SettingsRoute::Metadata:
            RenderMetadataRows(node);
            break;
        case SettingsRoute::Custom:
        case SettingsRoute::Fallback:
            RenderNodeProperties(node);
            break;
        case SettingsRoute::ViewSettings:
            RenderViewSettings(node);
            break;
        case SettingsRoute::DialogOnly:
            break;
    }

    // The data row (Data Input) and the truths no row claimed.
    const bool any_loose = std::any_of(view_.loose_truths.begin(), view_.loose_truths.end(),
                                       [this](const cyxwiz::properties_view::Truth* tr) { return !shown_truth_keys_.count(tr->canonical_key); });
    if (view_.data || any_loose) RenderLooseTruths(node);
    if (!view_.settings_note.empty()) Wrapped(t.text_dim, view_.settings_note);

    if (!view_.truths.empty()) {
        const bool open = settings_details_open_.count(node.id) > 0;
        if (properties_rows::Disclosure("Details##settings", open) != open) {
            if (open) settings_details_open_.erase(node.id);
            else settings_details_open_.insert(node.id);
        }
        cyxwiz::ui::Tooltip("Where each setting's value comes from and what the rules say about it");
    }
    ImGui::Spacing();
}

void Properties::RenderMetadataRows(MLNode& node) {
    for (const auto& group : view_.groups) {
        if (group.name.empty()) {
            properties_rows::Rows rows("##flat");
            if (rows.ok) for (const auto& row : group.rows) RenderMetadataRow(node, row);
            continue;
        }
        if (!ImGui::TreeNodeEx(group.name.c_str(), group.advanced ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) continue;
        {
            properties_rows::Rows rows("##group");
            if (rows.ok) for (const auto& row : group.rows) RenderMetadataRow(node, row);
        }
        ImGui::TreePop();
    }
}

void Properties::RenderMetadataRow(MLNode& node, const cyxwiz::properties_view::Setting& row) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    const cyxwiz::ParameterDefinition& param = *row.param;
    ImGui::PushID(param.name.c_str());

    std::string tip = param.description;
    if (row.label != cyxwiz::properties_view::ShortLabel(row.label, "") || !param.display_name.empty()) {
        // The full name first when the row shows a shortened one.
        const std::string full = param.display_name.empty() ? param.name : param.display_name;
        if (full != row.label) tip = full + (tip.empty() ? "" : "\n" + tip);
    }
    if (!param.default_value.empty()) tip += std::string(tip.empty() ? "" : "\n") + "Default: " + param.default_value;
    tip += std::string(tip.empty() ? "" : "\n") + (param.required ? "Required" : "Optional");
    properties_rows::Label(row.label.c_str(), row.required, tip.c_str());

    // Widgets work on a copy; the node is written only on a change.
    std::string value = row.value;
    bool changed = false;
    const properties_rules::NumericRange range = properties_rules::ParseNumericRange(param.validation);
    const bool has_reset = row.differs_from_default;
    const float reset_width = has_reset ? ImGui::CalcTextSize("Reset").x + t.space_md : 0.0f;
    const float editor_width = ImGui::GetContentRegionAvail().x - reset_width;

    if (param.type == "int") {
        int int_val = 0;
        properties_rules::TryParseIntStrict(value, int_val);
        ImGui::SetNextItemWidth(editor_width);
        const bool slider = param.name != "epochs" && range.has_range && range.max_value - range.min_value >= 1.0 &&
                            range.max_value - range.min_value <= 10000.0;
        if (slider) {
            int_val = std::clamp(int_val, static_cast<int>(range.min_value), static_cast<int>(range.max_value));
            if (ImGui::SliderInt("##value", &int_val, static_cast<int>(range.min_value), static_cast<int>(range.max_value))) {
                value = std::to_string(int_val);
                changed = true;
            }
        } else if (ImGui::InputInt("##value", &int_val)) {
            if (range.has_range) int_val = std::clamp(int_val, static_cast<int>(range.min_value), static_cast<int>(range.max_value));
            value = std::to_string(int_val);
            changed = true;
        }
    } else if (param.type == "float") {
        double parsed = 0.0;
        properties_rules::TryParseDoubleStrict(value, parsed);
        float float_val = static_cast<float>(parsed);
        ImGui::SetNextItemWidth(editor_width);
        char buf[32];
        if (range.has_range) {
            float_val = std::clamp(float_val, static_cast<float>(range.min_value), static_cast<float>(range.max_value));
            if (ImGui::SliderFloat("##value", &float_val, static_cast<float>(range.min_value), static_cast<float>(range.max_value), "%.4f")) {
                std::snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        } else if (ImGui::InputFloat("##value", &float_val, 0.01f, 0.1f, "%.4f")) {
            std::snprintf(buf, sizeof(buf), "%.4f", float_val);
            value = buf;
            changed = true;
        }
    } else if (param.type == "bool") {
        bool bool_val = (value == "true" || value == "1");
        if (ImGui::Checkbox("##value", &bool_val)) {
            value = bool_val ? "true" : "false";
            changed = true;
        }
    } else if ((param.type == "enum" || param.type == "dropdown") && !param.enum_values.empty()) {
        int current = 0;
        for (size_t i = 0; i < param.enum_values.size(); ++i) {
            if (param.enum_values[i] == value) current = static_cast<int>(i);
        }
        ImGui::SetNextItemWidth(editor_width);
        if (ImGui::BeginCombo("##value", param.enum_values[current].c_str())) {
            for (size_t i = 0; i < param.enum_values.size(); ++i) {
                if (ImGui::Selectable(param.enum_values[i].c_str(), static_cast<int>(i) == current)) {
                    value = param.enum_values[i];
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
    } else if (param.type == "file" || param.type == "directory" || param.type == "folder") {
        const bool folder = param.type != "file";
        char buf[512];
        std::strncpy(buf, value.c_str(), sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        const float browse = cyxwiz::ui::ButtonWidth("Browse", cyxwiz::ui::ButtonSize::Small);
        ImGui::SetNextItemWidth(std::max(40.0f, editor_width - browse - t.space_sm));
        if (ImGui::InputText("##value", buf, sizeof(buf))) {
            value = buf;
            changed = true;
        }
        ImGui::SameLine(0.0f, t.space_sm);
        if (cyxwiz::ui::SecondaryButton("Browse")) {
            std::optional<std::string> picked;
            if (folder) {
                picked = cyxwiz::FileDialogs::SelectFolder("Select Folder", value.empty() ? nullptr : value.c_str());
            } else if (properties_rules::ShouldUseSaveFileDialog(node, param)) {
                picked = cyxwiz::FileDialogs::SaveFile("Save Fitted State",
                                                       {{"CyxWiz Fitted State", "cyxstate.json"}, {"JSON", "json"}},
                                                       value.empty() ? nullptr : value.c_str(), "fitted_state.cyxstate.json");
            } else {
                picked = cyxwiz::FileDialogs::OpenFile(
                    param.name == "state_path" ? "Load Fitted State" : "Select File",
                    param.name == "state_path"
                        ? cyxwiz::FileDialogs::FilterList{{"CyxWiz Fitted State", "cyxstate.json"}, {"JSON", "json"}}
                        : cyxwiz::FileDialogs::FilterList{{"All Files", "*"}},
                    value.empty() ? nullptr : value.c_str());
            }
            if (picked) {
                value = *picked;
                changed = true;
            }
        }
    } else {
        const bool multiline = param.type == "multiline" || param.type == "text";
        const bool password = param.type == "password";
        char buf[2048];
        std::strncpy(buf, value.c_str(), sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        ImGui::SetNextItemWidth(editor_width);
        if (multiline) {
            if (ImGui::InputTextMultiline("##value", buf, sizeof(buf), ImVec2(editor_width, ImGui::GetTextLineHeight() * 4.0f))) {
                value = buf;
                changed = true;
            }
        } else if (ImGui::InputText("##value", buf, sizeof(buf), password ? ImGuiInputTextFlags_Password : 0)) {
            value = buf;
            changed = true;
        }
    }

    if (has_reset) {
        ImGui::SameLine(0.0f, t.space_sm);
        if (cyxwiz::ui::LinkButton("Reset")) {
            value = param.default_value;
            changed = true;
        }
        cyxwiz::ui::Tooltip(("Reset to default: " + param.default_value).c_str());
    }

    if (changed) {
        node.parameters[param.name] = value;
        // Architecture preset <-> block fields (TransformerDecoder, tofix112).
        cyxwiz::ApplyTransformerPresetEdit(node.type, node.parameters, param.name);
        NotifyEdited();
    }
    properties_rows::Status(param.name.c_str());
    if (!row.validation_error.empty()) properties_rows::Note(row.validation_error.c_str());
    ImGui::PopID();
}

void Properties::RenderLooseTruths(MLNode& node) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    properties_rows::Rows rows("##loose");
    if (!rows.ok) return;
    if (view_.data) {
        const auto& d = *view_.data;
        properties_rows::Label("Data", false, d.loaded ? d.store.c_str() : nullptr);
        ImGui::AlignTextToFramePadding();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_bright, "%s", d.label.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TableSetColumnIndex(2);
        if (d.loaded) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.success, "%s", ICON_FA_CIRCLE_CHECK);
            cyxwiz::ui::Tooltip(("Loaded as " + d.store).c_str());
        } else {
            cyxwiz::ui::Chip("not loaded", t.pending);
            cyxwiz::ui::Tooltip(d.note.c_str());
        }
        if (!d.detail.empty()) properties_rows::Note(d.detail.c_str());
    }
    for (const auto* truth : view_.loose_truths) {
        if (shown_truth_keys_.count(truth->canonical_key)) continue;
        ImGui::PushID(truth->canonical_key.c_str());
        if (truth->quick_editable && !truth->canonical_key.empty()) {
            properties_rows::Label(truth->label.c_str());
            char buf[256];
            std::strncpy(buf, truth->value.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            ImGui::InputText("##effective", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue);
            if (ImGui::IsItemDeactivatedAfterEdit() && truth->value != buf) {
                properties_truth::WriteCanonicalAndAliases(node, truth->canonical_key, buf);
                NotifyEdited();
            }
            properties_rows::Status(truth->canonical_key.c_str());
        } else {
            properties_rows::ReadOnly(truth->label.c_str(), truth->value, truth->canonical_key.c_str());
        }
        ImGui::PopID();
    }
}

void Properties::RenderNodeProperties(MLNode& node) {
    properties_node_editors::RenderNodeProperties(
        node,
        properties_node_editors::RenderNodePropertiesContext{
            node_editor_,
            scope_buffers_,
            [this]() { NotifyEdited(); }
        });
}

// Plot and Dashboard nodes: the settings saved by their window, in words, with
// the saved text under Details (read-only). They are changed in the window.
void Properties::RenderViewSettings(MLNode& node) {
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    const bool plot = node.type == NodeType::Plot;
    const char* key = plot ? "plot_spec" : "dashboard_spec";
    auto it = node.parameters.find(key);
    const std::string json = it != node.parameters.end() ? it->second : std::string();
    std::vector<std::pair<std::string, std::string>> rows;
    std::string problem;
    if (plot) {
        cyxwiz::plot::PlotSpec spec;
        if (json.empty() || cyxwiz::plot::SpecFromJson(json, spec, &problem)) rows = cyxwiz::plot::SpecSummary(spec);
    } else {
        cyxwiz::dashboard::DashboardSpec spec;
        if (json.empty() || cyxwiz::dashboard::DashboardFromJson(json, spec, &problem)) rows = cyxwiz::dashboard::DashboardSummary(spec);
    }
    ImGui::PushTextWrapPos(0.0f);
    if (!problem.empty()) {
        ImGui::TextColored(t.warning, "The saved settings could not be read: %s", problem.c_str());
        ImGui::TextColored(t.text_dim, "Opening the %s starts from the defaults.", plot ? "Plot window" : "Dashboard");
    } else if (json.empty() && plot) {
        ImGui::TextColored(t.text_dim, "Not set up yet: open the Plot window to pick the plot type and columns.");
    }
    if (!rows.empty() && ImGui::BeginTable("##view_settings", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##k", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Widgets ").x + 8.0f);
        ImGui::TableSetupColumn("##v", ImGuiTableColumnFlags_WidthStretch);
        for (const auto& [k, v] : rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(t.text_dim, "%s", k.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextWrapped("%s", v.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::TextColored(t.text_faint, "Data is read when the %s opens; only these settings are saved in the graph.",
                       plot ? "plot" : "dashboard");
    ImGui::PopTextWrapPos();
    if (!json.empty() && ImGui::TreeNode("Details##view_settings_json")) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "Saved as %s (set by the %s, read-only here)", key, plot ? "Plot window" : "Dashboard");
        ImGui::PopTextWrapPos();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
        ImGui::BeginChild("##json", ImVec2(-1, ImGui::GetTextLineHeight() * 8), ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(json.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
        if (cyxwiz::ui::SecondaryButton("Copy", true, nullptr, cyxwiz::ui::ButtonSize::Small)) ImGui::SetClipboardText(json.c_str());
        ImGui::TreePop();
    }
}

// ---------------------------------------------------------------------------
// Advanced: position, connections, one raw-parameter table.

void Properties::RenderAdvanced(MLNode& node) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::Spacing();
    if (!Section("Advanced", section_advanced_open_)) return;

    std::vector<cyxwiz::ui::KeyValue> facts;
    if (view_.has_position) {
        char pos[64];
        std::snprintf(pos, sizeof(pos), "(%.1f, %.1f)", view_.position_x, view_.position_y);
        facts.push_back({"Position", pos});
    }
    facts.push_back({"Connections", std::to_string(view_.links_in) + " in, " + std::to_string(view_.links_out) + " out"});
    cyxwiz::ui::KeyValueTable("##advanced_facts", facts, 1);

    if (view_.raw.empty()) return;
    ImGui::Spacing();
    ImGui::TextUnformatted("Raw parameters");
    ImGui::SameLine(0.0f, t.space_sm);
    ImGui::TextColored(t.text_dim, "what the graph file stores for this node");
    const float remove_width = cyxwiz::ui::ButtonWidth("Remove", cyxwiz::ui::ButtonSize::Small);
    if (ImGui::BeginTable("##raw", 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("Maps to", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("##remove", ImGuiTableColumnFlags_WidthFixed, remove_width);
        ImGui::TableNextRow();
        for (int c = 0; c < 3; ++c) {
            ImGui::TableSetColumnIndex(c);
            ImGui::TextColored(t.text_dim, "%s", c == 0 ? "Key" : c == 1 ? "Value" : "Maps to");
        }
        std::string remove_key;
        for (const auto& raw : view_.raw) {
            ImGui::PushID(raw.key.c_str());
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(raw.key.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableSetColumnIndex(1);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_bright, "%s", raw.value.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableSetColumnIndex(2);
            ImGui::PushTextWrapPos(0.0f);
            if (!raw.maps_to.empty()) ImGui::TextUnformatted(raw.maps_to.c_str());
            else ImGui::TextColored(t.text_faint, "%s", raw.cleanup_allowed ? "nothing (unused)" : "not mapped by the current rules");
            for (const auto& chip : raw.chips) {
                if (chip.kind == ChipKind::Ok) continue;
                properties_rows::Chip(chip);
            }
            if (!raw.cleanup_reason.empty()) ImGui::TextColored(t.text_dim, "%s", raw.cleanup_reason.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableSetColumnIndex(3);
            if (raw.cleanup_allowed && cyxwiz::ui::DangerButton("Remove")) remove_key = raw.key;
            ImGui::PopID();
        }
        ImGui::EndTable();
        if (!remove_key.empty()) {
            node.parameters.erase(remove_key);
            NotifyEdited();
        }
    }
}

// ==================== Node Executor Integration ====================

void Properties::RenderExecutorSection(MLNode& node) {
    properties_executor::RenderExecutorSection(node_editor_, node);
}

} // namespace gui
