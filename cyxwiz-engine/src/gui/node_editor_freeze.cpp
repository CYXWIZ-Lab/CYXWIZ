// Freezing layers from the canvas (fine-tuning): the Freeze menu's actions,
// the "all but the last N" popup, and the Trainable / Frozen count. The
// actions themselves are core/freeze_actions; this file applies them to the
// editor's graph with undo.

#include "node_editor.h"

#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/engine_config.h"
#include "../core/freeze_actions.h"
#include "../core/live_graph_compile.h"
#include "../core/project_manager.h"
#include "../core/weight_file.h"
#include "../core/weight_match.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <functional>
#include <set>
#include <string>

namespace gui {

namespace frozen = cyxwiz::frozen;
namespace ui = cyxwiz::ui;

namespace {

// The weights file the graph's Data Loader starts from, as the run resolves it; empty when none is set.
std::string WeightsFileOf(const std::vector<MLNode>& nodes) {
    for (const auto& node : nodes) {
        if (node.type != NodeType::DataLoader) continue;
        const auto it = node.parameters.find("initialize_from_weights_file");
        if (it == node.parameters.end() || it->second.empty()) continue;
        std::filesystem::path path = it->second;
        if (!path.is_absolute()) {
            const std::string& root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
            if (!root.empty()) path = (std::filesystem::path(root) / path).lexically_normal();
        }
        return path.string();
    }
    return {};
}

}  // namespace

bool NodeEditor::HasLayersWithWeights() const {
    for (const auto& node : nodes_) {
        if (frozen::LayerNodeHoldsWeights(node.type)) return true;
    }
    return false;
}

// Applies an action to a copy first, so an action that changes nothing leaves
// no undo step. One outcome line per action; `nothing` says why no node changed.
void NodeEditor::ApplyFreeze(const std::function<size_t(std::vector<MLNode>&)>& action, const std::string& what,
                             const std::string& nothing) {
    std::vector<MLNode> changed_nodes = nodes_;
    const size_t changed = action(changed_nodes);
    if (changed == 0) {
        freeze_message_ = nothing;
        spdlog::info("Freeze: {}: nothing changed", what);
        return;
    }
    SaveUndoState();
    nodes_ = std::move(changed_nodes);
    spdlog::info("Freeze: {}: {} node{} changed", what, changed, changed == 1 ? "" : "s");
}

void NodeEditor::FreezeSelectedNodes(bool freeze) {
    std::set<int> ids(selected_node_ids_.begin(), selected_node_ids_.end());
    if (selected_node_id_ >= 0) ids.insert(selected_node_id_);
    ApplyFreeze([&](std::vector<MLNode>& nodes) { return frozen::SetFrozen(nodes, ids, freeze); },
                freeze ? "freeze selected" : "unfreeze selected",
                freeze ? "The selection has no layer with weights that is not frozen already."
                       : "The selection has no frozen layer.");
}

void NodeEditor::UnfreezeAllLayers() {
    ApplyFreeze([](std::vector<MLNode>& nodes) { return frozen::UnfreezeAll(nodes); }, "unfreeze all",
                "No layer is frozen.");
}

void NodeEditor::FreezeAllButLastLayers(size_t keep) {
    const cyxwiz::TrainingConfiguration* config = live_compile_ ? live_compile_->CleanConfig() : nullptr;
    if (!config) {
        freeze_message_ = "The graph has to compile without errors first: the order of its layers comes from the "
                          "compiled model.";
        return;
    }
    const std::vector<int> order = frozen::ModelOrder(config->layers);
    ApplyFreeze([&](std::vector<MLNode>& nodes) { return frozen::FreezeAllButLast(nodes, order, keep); },
                "freeze all but the last " + std::to_string(keep), "The layers are already frozen that way.");
}

void NodeEditor::FreezeLayersLoadedFromFile() {
    const std::string path = WeightsFileOf(nodes_);
    if (path.empty()) {
        freeze_message_ = "No weights file is chosen. Set Start from to A weights file on the Data Loader "
                          "(Checkpoints tab) first.";
        return;
    }
    if (!live_compile_ || !live_compile_->UpToDate() || live_compile_->WeightSlots().empty()) {
        freeze_message_ = "The graph has to compile without errors first, so the file can be matched to its layers.";
        return;
    }
    // Names, shapes and types only: no tensor is read, so this is quick.
    cyxwiz::weight_file::WeightFile file;
    std::string error;
    if (!cyxwiz::weight_file::Read(path, file, error, false)) {
        freeze_message_ = "The weights file could not be read: " + error;
        return;
    }
    const auto report = cyxwiz::weight_match::Match(file, live_compile_->WeightSlots());
    std::set<std::string> loaded;
    for (const auto& row : report.layers) {
        if (row.loaded) loaded.insert(row.node_name);
    }
    if (loaded.empty()) {
        freeze_message_ = "No layer of this graph takes weights from the file.";
        return;
    }
    ApplyFreeze([&](std::vector<MLNode>& nodes) { return frozen::FreezeNamed(nodes, loaded); },
                "freeze the layers loaded from the file", "Every layer the file fills is frozen already.");
}

// A node added from the palette starts from the user's saved defaults.
void NodeEditor::ApplyUserNodeDefaults(MLNode& node) const {
    if (frozen::LayerNodeHasFrozenStatistics(node.type)) {
        const auto defaults = cyxwiz::core::EngineConfig::Instance().GetTrainingDefaults();
        node.parameters[frozen::kStatistics] =
            defaults.frozen_batchnorm_keeps_statistics ? frozen::kStatisticsKeep : frozen::kStatisticsUpdate;
    }
}

void NodeEditor::RunFreezeCommand(const std::string& command) {
    if (command == "selected") FreezeSelectedNodes(true);
    else if (command == "unselected") FreezeSelectedNodes(false);
    else if (command == "loaded") FreezeLayersLoadedFromFile();
    else if (command == "but_head") FreezeAllButLastLayers(1);
    else if (command == "but_last") freeze_last_popup_ = true;
    else if (command == "unfreeze_all") UnfreezeAllLayers();
}

// The Freeze submenu of the canvas menu (the Nodes menu has the same commands).
void NodeEditor::RenderFreezeMenuItems() {
    const bool selection = !selected_node_ids_.empty() || selected_node_id_ >= 0;
    const bool layers = HasLayersWithWeights();
    const auto item = [&](const char* label, const char* command, bool enabled) {
        if (ImGui::MenuItem(label, nullptr, false, enabled && !is_training_)) {
            RunFreezeCommand(command);
            ImGui::CloseCurrentPopup();
        }
    };
    item("Freeze selected", "selected", selection);
    item("Unfreeze selected", "unselected", selection);
    ImGui::Separator();
    item("Freeze every layer loaded from the file", "loaded", layers);
    item("Freeze all but the head", "but_head", layers);
    item("Freeze all but the last N layers...", "but_last", layers);
    ImGui::Separator();
    item("Unfreeze all", "unfreeze_all", layers);
}

// "All but the last N" asks for N; a message says why an action did nothing.
void NodeEditor::RenderFreezePopups() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (freeze_last_popup_) {
        ImGui::OpenPopup("Freeze all but the last layers");
        freeze_last_popup_ = false;
    }
    const ImVec2 centre = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Freeze all but the last layers", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Keep training the last");
        ImGui::SameLine();
        ui::NumberField("##freeze_keep", &freeze_keep_layers_, 1, 100000, 1, 70.0f, "Layers with weights, counted from the output");
        ImGui::SameLine();
        ImGui::TextUnformatted(freeze_keep_layers_ == 1 ? "layer with weights" : "layers with weights");
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextColored(t.text_dim, "Every layer before them is frozen; these are unfrozen. Layers are counted in the "
                                       "model's order, from the output back. 1 keeps only the head.");
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        if (ui::PrimaryButton("Freeze", true, nullptr, ui::ButtonSize::Small)) {
            FreezeAllButLastLayers(static_cast<size_t>(freeze_keep_layers_));
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ui::SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if (!freeze_message_.empty() && !ImGui::IsPopupOpen("Freeze")) ImGui::OpenPopup("Freeze");
    ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Freeze", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(freeze_message_.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        if (ui::SecondaryButton("Close")) {
            freeze_message_.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// "Trainable 1,026 · Frozen 11,176,512" for the footer; empty when nothing is
// frozen or the model has not been counted for the graph as it is now.
std::string NodeEditor::FrozenTotalsText() const {
    if (!live_compile_ || !live_compile_->ParametersCounted() || !live_compile_->UpToDate()) return {};
    const cyxwiz::TrainingConfiguration* config = live_compile_->Config();
    if (!config) return {};
    const auto totals = frozen::CountTotals(config->layers, live_compile_->LayerParameters());
    return totals.frozen > 0 ? frozen::TotalsText(totals) : std::string();
}

}  // namespace gui
