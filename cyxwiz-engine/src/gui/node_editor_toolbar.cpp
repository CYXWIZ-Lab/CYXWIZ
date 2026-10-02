// CyxWiz Studio nav (approved board 7, owner 2026-10-02): one row in groups
// with no separator lines. Left: the graph (its menu holds Save, Save As,
// Load, Export, the Custom Node Editor and Clear) and Save. Middle: edit and
// view tools as icons with their names on hover, and the counts. Right: the
// code controls, then the run group with Train as the one main button. On a
// narrow canvas the edit, code and view groups move into a "more" menu, so
// nothing depends on widening the window. Inventory of every item:
// Data Studio/tofix134/inventory_studio_toolbar.md.
#include "node_editor.h"

#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../core/async_task_manager.h"
#include "../core/graph_executor.h"
#include "../core/rl_training_executor.h"
#include "../core/training_manager.h"

#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>

namespace gui {

namespace {

namespace ui = cyxwiz::ui;

constexpr float kGroupGap = 14.0f;
constexpr const char* kModes[] = {"Code Gen", "Data Pipeline", "Local Training"};
constexpr const char* kFrameworks[] = {"PyTorch", "TensorFlow", "Keras", "PyCyxWiz"};
constexpr float kModeWidth = 130.0f;
constexpr float kFrameworkWidth = 110.0f;

float W(const char* label) { return ui::ButtonWidth(label, ui::ButtonSize::Small); }
float Spacing() { return ImGui::GetStyle().ItemSpacing.x; }

}  // namespace

void NodeEditor::UpdateUnsavedState() {
    // A cheap check twice a second: the graph as it would be saved, against
    // the last save or load. A load settles for a few frames first.
    if (fingerprint_baseline_frames_ > 0) {
        if (--fingerprint_baseline_frames_ == 0) {
            saved_fingerprint_ = std::hash<std::string>{}(GraphDocumentJson().dump());
            unsaved_ = false;
        }
        return;
    }
    const double now = ImGui::GetTime();
    if (now - fingerprint_checked_at_ < 0.5) return;
    fingerprint_checked_at_ = now;
    unsaved_ = std::hash<std::string>{}(GraphDocumentJson().dump()) != saved_fingerprint_;
}

void NodeEditor::DrawGraphMenu() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ImGui::BeginPopup("##graph_menu")) return;
    if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK "  Save", "Ctrl+S")) SaveCurrentGraph();
    if (ImGui::MenuItem(ICON_FA_FILE_EXPORT "  Save As...", "Ctrl+Shift+S")) ShowSaveDialog();
    if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN "  Load...")) ShowLoadDialog();
    ImGui::Separator();
    if (ImGui::MenuItem(ICON_FA_FILE_EXPORT "  Export...")) ShowExportDialog();
    if (open_custom_node_editor_callback_ && ImGui::MenuItem(ICON_FA_WAND_MAGIC_SPARKLES "  Custom Node Editor"))
        open_custom_node_editor_callback_();
    if (open_custom_node_editor_callback_ && ImGui::IsItemHovered())
        ImGui::SetTooltip("Create or edit custom node types (pins, parameters, code templates)");
    ImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, t.error);
    if (ImGui::MenuItem(ICON_FA_ERASER "  Clear the canvas...", nullptr, false, !nodes_.empty())) clear_confirm_requested_ = true;
    ImGui::PopStyleColor();
    ImGui::EndPopup();
}

void NodeEditor::DrawClearConfirm() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (clear_confirm_requested_) {
        ImGui::OpenPopup("Clear the canvas?###clear_canvas");
        clear_confirm_requested_ = false;
    }
    if (!ImGui::BeginPopupModal("Clear the canvas?###clear_canvas", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    ImGui::Text("This removes all %zu nodes and %zu links from the canvas.", nodes_.size(), links_.size());
    ImGui::TextColored(t.text_dim, "Ctrl+Z brings them back. The graph file changes only when you save.");
    ImGui::Spacing();
    if (ui::DangerButton("Clear##confirm", true, nullptr, ui::ButtonSize::Regular)) {
        ClearGraph();
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel##clear", true, nullptr, ui::ButtonSize::Regular) || ImGui::IsKeyPressed(ImGuiKey_Escape))
        ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

void NodeEditor::ShowToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    UpdateUnsavedState();

    // ---- What the run group shows in this state.
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    const bool training_active = training_mgr.IsTrainingActive();
    const auto pipeline_task = pipeline_task_id_ == 0 ? nullptr : cyxwiz::AsyncTaskManager::Instance().GetTask(pipeline_task_id_);
    const bool pipeline_running = execution_mode_ == ExecutionMode::DuckDBPipeline && pipeline_execution_active_ && pipeline_task;
    const bool has_sim = HasSimulationNodes();
    const bool has_rl = HasRLNodes();
    const bool rl_running = rl_script_running_ || (rl_executor_ && rl_executor_->IsTraining());
    const int num_selected = ImNodes::NumSelectedNodes();

    // ---- Labels and widths of each group (to decide what fits).
    const std::string name = current_file_path_.empty() ? std::string("Untitled graph")
                                                       : std::filesystem::path(current_file_path_).filename().string();
    const std::string chip_prefix = std::string(ICON_FA_DIAGRAM_PROJECT "  ") + name;
    const std::string chip = chip_prefix + (unsaved_ ? "    " : "  ") + ICON_FA_CHEVRON_DOWN + "##graph_chip";
    const float sp = Spacing();
    const float w_left = W(chip.c_str()) + sp + W(ICON_FA_FLOPPY_DISK " Save");
    const float w_zoom = W(ICON_FA_MINUS) + sp + ImGui::CalcTextSize("100%").x + 8.0f + sp + W(ICON_FA_PLUS);
    const float w_edit = W(ICON_FA_OBJECT_GROUP) + W(ICON_FA_COPY) + W(ICON_FA_TRASH) + 2 * sp;
    const float w_view = W(ICON_FA_EXPAND " Fit") + sp + W(ICON_FA_SITEMAP " Minimap");
    std::string stats_text = std::to_string(nodes_.size()) + " nodes \xC2\xB7 " + std::to_string(links_.size()) + " links";
    if (num_selected > 0) stats_text += " \xC2\xB7 " + std::to_string(num_selected) + " selected";
    const char* stats = stats_text.c_str();
    const float w_stats = ImGui::CalcTextSize(stats).x + 10.0f;
    float w_code = kModeWidth + sp + W(ICON_FA_FILE_EXPORT " Export");
    if (execution_mode_ == ExecutionMode::CodeGeneration) w_code += kFrameworkWidth + sp + W(ICON_FA_GEARS " Generate") + sp;
    else if (!pipeline_running) w_code += W(ICON_FA_PLAY " Execute Pipeline") + sp;
    const float w_pipeline_run = pipeline_running ? 150.0f + 50.0f + W(ICON_FA_STOP " Cancel") + 2 * sp : 0.0f;
    const float w_debug = debug_callback_ ? W(ICON_FA_BUG " Local Debug") + sp : 0.0f;
    float w_extras = 0.0f;  // simulation and RL
    if (has_sim) w_extras += W(ICON_FA_PLAY " Run Sim") + 70.0f + sp;
    if (has_rl) w_extras += W(ICON_FA_PLAY " Train RL") + W(ICON_FA_FILE_EXPORT " Export ONNX") + 90.0f + 2 * sp;
    const float w_run = (compile_callback_ ? W(ICON_FA_CHECK_DOUBLE " Compile") + sp : 0.0f) +
                        (training_active ? 220.0f : W(ICON_FA_PLAY " Train"));
    const float w_more = W(ICON_FA_ELLIPSIS) + sp;

    // Narrowing, in order: edit, code, view and counts, then Local Debug and
    // the simulation / RL group move into the "more" menu.
    const float avail = ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().WindowPadding.x;
    int level = 0;
    const auto need = [&](int l) {
        float w = w_left + kGroupGap + w_zoom + kGroupGap + w_run + w_pipeline_run;
        if (l < 1) w += w_edit + kGroupGap;
        if (l < 2) w += w_code + kGroupGap;
        if (l < 3) w += w_view + sp + w_stats;
        if (l < 4) w += w_debug + w_extras;
        if (l > 0) w += w_more;
        return w;
    };
    while (level < 4 && need(level) > avail) ++level;
    const bool show_edit = level < 1, show_code = level < 2, show_view = level < 3, show_debug_extras = level < 4;

    // ---- The bar: one row on the toolbar surface.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_bar);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
    ImGui::BeginChild("##studio_nav", ImVec2(0, ImGui::GetFrameHeight() + 12.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    // Left: the graph and Save.
    const ImVec2 chip_min = ImGui::GetCursorScreenPos();
    if (ui::GhostButton(chip.c_str(), true, nullptr, true)) ImGui::OpenPopup("##graph_menu");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s%s", current_file_path_.empty() ? "Not saved to a file yet" : current_file_path_.c_str(),
                          unsaved_ && !current_file_path_.empty() ? "\nChanged since it was saved" : "");
    if (unsaved_) {
        // The unsaved mark, inside the chip after the name.
        const float pad = (W(chip.c_str()) - ImGui::CalcTextSize(chip.c_str(), nullptr, true).x - 2.0f) * 0.5f;
        const float x = chip_min.x + pad + 1.0f + ImGui::CalcTextSize(chip_prefix.c_str()).x +
                        ImGui::CalcTextSize("    ").x * 0.5f;
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(x, chip_min.y + ImGui::GetFrameHeight() * 0.5f), 3.5f,
                                                    ui::ToU32(t.text_dim));
    }
    DrawGraphMenu();
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_FLOPPY_DISK " Save")) SaveCurrentGraph();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", current_file_path_.empty() ? "Save the graph (Ctrl+S): asks for a file name the first time"
                                                           : ("Save to " + name + " (Ctrl+S)").c_str());

    // Middle: edit, zoom, view, counts.
    const char* no_selection = "Select nodes first";
    if (show_edit) {
        ImGui::SameLine(0.0f, kGroupGap);
        if (ui::GhostButton(ICON_FA_OBJECT_GROUP "##select_all", !nodes_.empty(), "The canvas is empty")) SelectAll();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Select all (Ctrl+A)");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_COPY "##duplicate", num_selected > 0, no_selection)) DuplicateSelection();
        if (ImGui::IsItemHovered() && num_selected > 0) ImGui::SetTooltip("Duplicate (Ctrl+D)");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_TRASH "##delete", num_selected > 0, no_selection)) DeleteSelected();
        if (ImGui::IsItemHovered() && num_selected > 0) ImGui::SetTooltip("Delete selected (Delete)");
    }
    ImGui::SameLine(0.0f, kGroupGap);
    if (ui::GhostButton(ICON_FA_MINUS "##zoom_out")) zoom_ = std::max(ZOOM_MIN, zoom_ - 0.1f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom out");
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%.0f%%", zoom_ * 100.0f);
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_PLUS "##zoom_in")) zoom_ = std::min(ZOOM_MAX, zoom_ + 0.1f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom in");
    if (show_view) {
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_EXPAND " Fit", !nodes_.empty(), "The canvas is empty")) FrameAll();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Frame all nodes (F; with nodes selected, F frames them)");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_SITEMAP " Minimap", true, nullptr, show_minimap_)) show_minimap_ = !show_minimap_;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(show_minimap_ ? "Hide the minimap (M)" : "Show the minimap (M)");
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%zu nodes \xC2\xB7 %zu links", nodes_.size(), links_.size());
        if (num_selected > 0) {
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(t.text_dim, " \xC2\xB7 ");
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(t.accent_text, "%d selected", num_selected);
        }
    }

    // Right: more (narrow), code, run.
    const float right = (level > 0 ? w_more : 0.0f) + (show_code ? w_code + kGroupGap : 0.0f) + w_pipeline_run +
                        (show_debug_extras ? w_debug + w_extras : 0.0f) + w_run;
    ImGui::SameLine();
    const float x_right = ImGui::GetWindowContentRegionMax().x - right;
    if (x_right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x_right);

    if (level > 0) {
        if (ui::GhostButton(ICON_FA_ELLIPSIS "##nav_more")) ImGui::OpenPopup("##nav_more_menu");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("More: the tools that do not fit this width");
        if (ImGui::BeginPopup("##nav_more_menu")) {
            if (!show_edit) {
                if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP "  Select all", "Ctrl+A", false, !nodes_.empty())) SelectAll();
                if (ImGui::MenuItem(ICON_FA_COPY "  Duplicate", "Ctrl+D", false, num_selected > 0)) DuplicateSelection();
                if (ImGui::MenuItem(ICON_FA_TRASH "  Delete selected", "Delete", false, num_selected > 0)) DeleteSelected();
                ImGui::Separator();
            }
            if (!show_view) {
                if (ImGui::MenuItem(ICON_FA_EXPAND "  Fit", "F", false, !nodes_.empty())) FrameAll();
                if (ImGui::MenuItem(ICON_FA_SITEMAP "  Minimap", "M", show_minimap_)) show_minimap_ = !show_minimap_;
                ImGui::TextDisabled("%s", stats);
                ImGui::Separator();
            }
            if (!show_code) {
                if (ImGui::BeginMenu(ICON_FA_CODE "  Mode")) {
                    for (int i = 0; i < 3; ++i)
                        if (ImGui::MenuItem(kModes[i], nullptr, static_cast<int>(execution_mode_) == i))
                            execution_mode_ = static_cast<ExecutionMode>(i);
                    ImGui::EndMenu();
                }
                if (execution_mode_ == ExecutionMode::CodeGeneration) {
                    if (ImGui::BeginMenu("    Framework")) {
                        for (int i = 0; i < 4; ++i)
                            if (ImGui::MenuItem(kFrameworks[i], nullptr, static_cast<int>(selected_framework_) == i))
                                selected_framework_ = static_cast<CodeFramework>(i);
                        ImGui::EndMenu();
                    }
                    if (ImGui::MenuItem(ICON_FA_GEARS "  Generate code")) GeneratePythonCode();
                } else if (execution_mode_ == ExecutionMode::DuckDBPipeline && !pipeline_running) {
                    if (ImGui::MenuItem(ICON_FA_PLAY "  Execute Pipeline")) ExecuteDataPipeline();
                }
                if (ImGui::MenuItem(ICON_FA_FILE_EXPORT "  Export...")) ShowExportDialog();
                ImGui::Separator();
            }
            if (!show_debug_extras) {
                if (debug_callback_ && ImGui::MenuItem(ICON_FA_BUG "  Local Debug", "F6")) debug_callback_();
                if (has_sim) {
                    if (is_simulating_) {
                        if (ImGui::MenuItem(ICON_FA_STOP "  Stop Sim")) OnStopSimulation();
                    } else if (ImGui::MenuItem(ICON_FA_PLAY "  Run Sim", nullptr, false, !rl_running)) {
                        OnRunSimulation();
                    }
                }
                if (has_rl) {
                    if (rl_running) {
                        if (ImGui::MenuItem(ICON_FA_STOP "  Stop RL")) OnStopRLTraining();
                    } else if (ImGui::MenuItem(ICON_FA_PLAY "  Train RL", nullptr, false, !is_simulating_)) {
                        OnStartRLTraining();
                    }
                    const bool has_trained = rl_executor_ && !rl_executor_->IsTraining() && rl_executor_->GetMetrics().episode_count > 0;
                    if (ImGui::MenuItem(ICON_FA_FILE_EXPORT "  Export ONNX...", nullptr, false, has_trained)) export_onnx_dialog_open_ = true;
                }
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
    }

    // Code: mode, framework, Generate / Execute Pipeline, Export.
    if (show_code) {
        ImGui::SetNextItemWidth(kModeWidth);
        int mode = static_cast<int>(execution_mode_);
        if (ImGui::Combo("##ExecMode", &mode, kModes, 3)) {
            execution_mode_ = static_cast<ExecutionMode>(mode);
            spdlog::info("Execution mode changed to: {}", kModes[mode]);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Execution Mode:\nCode Gen - Generate Python code\nData Pipeline - Execute with DuckDB/Arrow\n"
                              "Local Training - Train ML model locally");
        ImGui::SameLine();
        if (execution_mode_ == ExecutionMode::CodeGeneration) {
            ImGui::SetNextItemWidth(kFrameworkWidth);
            int framework = static_cast<int>(selected_framework_);
            if (ImGui::Combo("##Framework", &framework, kFrameworks, 4)) {
                selected_framework_ = static_cast<CodeFramework>(framework);
                spdlog::info("Code generation framework changed to: {}", kFrameworks[framework]);
            }
            ImGui::SameLine();
            if (ui::GhostButton(ICON_FA_GEARS " Generate")) GeneratePythonCode();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Generate Python code for the chosen framework");
            ImGui::SameLine();
        } else if (execution_mode_ == ExecutionMode::DuckDBPipeline && !pipeline_running) {
            if (ui::GhostButton(ICON_FA_PLAY " Execute Pipeline")) ExecuteDataPipeline();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Execute data transformation pipeline using DuckDB");
            ImGui::SameLine();
        }
        if (ui::GhostButton(ICON_FA_FILE_EXPORT " Export")) ShowExportDialog();
        ImGui::SameLine(0.0f, kGroupGap);
    }

    // A running data pipeline: progress and Cancel stay in every width.
    if (pipeline_running) {
        const auto info = pipeline_task->GetInfo();
        ImGui::AlignTextToFramePadding();
        ImGui::ProgressBar(info.progress, ImVec2(150.0f, 6.0f), "");
        if (ImGui::IsItemHovered() && !info.status_message.empty()) ImGui::SetTooltip("%s", info.status_message.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%.0f%%", info.progress * 100.0f);
        ImGui::SameLine();
        if (ui::DangerButton(ICON_FA_STOP " Cancel")) cyxwiz::AsyncTaskManager::Instance().Cancel(pipeline_task_id_);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cancel the pipeline");
        ImGui::SameLine();
    }

    // Simulation and RL graphs: their run group next to Train.
    if (show_debug_extras && has_sim) {
        if (is_simulating_) {
            if (ui::DangerButton(ICON_FA_STOP " Stop Sim")) OnStopSimulation();
            if (graph_executor_) {
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.success, "t=%.2fs", graph_executor_->GetSimTime());
            }
        } else if (ui::GhostButton(ICON_FA_PLAY " Run Sim", !rl_running, "Stop RL training first")) {
            OnRunSimulation();
        }
        ImGui::SameLine();
    }
    if (show_debug_extras && has_rl) {
        if (rl_running) {
            if (ui::DangerButton(ICON_FA_STOP " Stop RL")) OnStopRLTraining();
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            if (rl_script_running_) {
                ImGui::TextColored(t.success, "Training via Python...");
            } else if (rl_executor_) {
                const auto m = rl_executor_->GetMetrics();
                ImGui::TextColored(t.success, "Ep %d | R:%.1f", m.episode_count, m.mean_episode_reward);
            }
        } else if (ui::GhostButton(ICON_FA_PLAY " Train RL", !is_simulating_, "Stop graph simulation first")) {
            OnStartRLTraining();
        }
        ImGui::SameLine();
        const bool has_trained = rl_executor_ && !rl_executor_->IsTraining() && rl_executor_->GetMetrics().episode_count > 0;
        if (ui::GhostButton(ICON_FA_FILE_EXPORT " Export ONNX", has_trained, "Train an RL agent first")) export_onnx_dialog_open_ = true;
        ImGui::SameLine(0.0f, kGroupGap);
    }

    // Run: Compile, Local Debug, Train (or the epoch and Stop).
    if (compile_callback_) {
        if (ui::GhostButton(ICON_FA_CHECK_DOUBLE " Compile")) {
            spdlog::info("NodeEditor: Compile Graph invoked from toolbar");
            compile_callback_();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Compile the graph (F7) - validates structure and reports config without training");
        ImGui::SameLine();
        if (show_debug_extras && debug_callback_) {
            if (ui::GhostButton(ICON_FA_BUG " Local Debug")) {
                spdlog::info("NodeEditor: Local Debug invoked from toolbar");
                debug_callback_();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Local Debug (F6) - run one forward + one backward pass on synthetic data. "
                                  "Catches shape / NaN / dead-gradient bugs before real training starts.");
            ImGui::SameLine();
        }
    }
    if (training_active) {
        const auto metrics = training_mgr.GetCurrentMetrics();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.success, ICON_FA_SPINNER " Epoch %d / %d", metrics.current_epoch, metrics.total_epochs);
        ImGui::SameLine();
        if (ui::DangerButton(ICON_FA_STOP " Stop")) training_mgr.StopTraining();
    } else {
        const bool can_train = IsGraphValid() && train_callback_;
        const char* train_blocked = !train_callback_ ? "Training is not available yet (no dataset loaded?)"
                                                     : "The graph is not ready to train. It needs a dataset input, "
                                                       "model layers and a loss.";
        if (ui::PrimaryButton(ICON_FA_PLAY " Train", can_train, train_blocked, ui::ButtonSize::Small)) {
            spdlog::info("NodeEditor: Starting training from graph");
            train_callback_(nodes_, links_);
        }
    }
    // Simulation speed while simulating.
    if (is_simulating_ && last_eval_time_ms_ > 0.0f) {
        ImGui::SameLine();
        const float fps = last_eval_time_ms_ > 0.001f ? 1000.0f / last_eval_time_ms_ : 0.0f;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(last_eval_time_ms_ < 16.0f ? t.success : t.warning, "%.1fms (%.0f FPS)", last_eval_time_ms_, fps);
    }
    ImGui::EndChild();

    DrawClearConfirm();

    // ONNX Export dialog
    if (export_onnx_dialog_open_) {
        ImGui::OpenPopup("Export Policy (ONNX)");
        export_onnx_dialog_open_ = false;
    }
    if (ImGui::BeginPopupModal("Export Policy (ONNX)", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Export trained RL policy to ONNX format.");
        ImGui::Spacing();
        static char onnx_path[512] = "policy.onnx";
        ImGui::TextColored(t.text_dim, "Output path");
        ImGui::InputText("##onnx_path", onnx_path, sizeof(onnx_path));
        ImGui::Spacing();
        if (ui::PrimaryButton("Export##onnx", true, nullptr, ui::ButtonSize::Regular)) {
            ExportPolicyONNX(std::string(onnx_path));
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ui::SecondaryButton("Cancel##onnx", true, nullptr, ui::ButtonSize::Regular)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

}  // namespace gui
