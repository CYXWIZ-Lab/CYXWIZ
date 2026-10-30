// CyxWiz Studio canvas bar (TOFIX178 UI-1, approved mockup
// https://claude.ai/artifact/3NUdq1C7mfW2QhiG6NTBD8): the graph and Save, edit
// and view; the mode as a segmented control with its action (or the
// simulation / RL group); Compile > Local Debug > Train, each with its last
// result. It folds by width only (core/studio_toolbar_presentation): labels go
// first, then Edit and View become one-level menus, then the mode; no "more"
// menu and no menu inside a menu. The counts and what the canvas is doing are
// in the canvas footer (RenderCanvasFooter).
#include "node_editor.h"

#include "icons.h"
#include "separate_windows.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../core/async_task_manager.h"
#include "../core/graph_executor.h"
#include "../core/live_graph_compile.h"
#include "../core/rl_training_executor.h"
#include "../core/studio_toolbar_presentation.h"
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
constexpr const char* kFrameworks[] = {"PyTorch", "TensorFlow", "Keras", "PyCyxWiz"};
constexpr float kFrameworkWidth = 110.0f;

float W(const char* label) { return ui::ButtonWidth(label, ui::ButtonSize::Small); }
float Spacing() { return ImGui::GetStyle().ItemSpacing.x; }

}  // namespace

size_t NodeEditor::FingerprintOf(const nlohmann::json& document) {
    // Invalid UTF-8 (for example a text field cut inside a character) is
    // replaced instead of throwing.
    return std::hash<std::string>{}(
        document.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
}

std::optional<size_t> NodeEditor::GraphFingerprint() {
    // Runs every half second from the toolbar, so it must never throw: capturing
    // an expanded subgraph throws on a missing node, pin or boundary target, and
    // an uncaught exception here could close the Engine without a log line (TOFIX146).
    try {
        const size_t fingerprint = FingerprintOf(GraphDocumentJson());
        fingerprint_error_.clear();
        return fingerprint;
    } catch (const std::exception& e) {
        if (fingerprint_error_ != e.what()) {
            fingerprint_error_ = e.what();
            spdlog::error("Unsaved-changes check could not capture the graph: {}", fingerprint_error_);
        }
        return std::nullopt;
    }
}

void NodeEditor::UpdateUnsavedState() {
    // A cheap check twice a second: the graph as it would be saved, against
    // the last save or load. A load settles for a few frames first.
    if (fingerprint_baseline_frames_ > 0) {
        if (--fingerprint_baseline_frames_ == 0) {
            const auto fingerprint = GraphFingerprint();
            saved_fingerprint_ = fingerprint.value_or(0);
            unsaved_ = !fingerprint.has_value();
        }
        return;
    }
    const double now = ImGui::GetTime();
    if (now - fingerprint_checked_at_ < 0.5) return;
    fingerprint_checked_at_ = now;
    // A graph that cannot be captured counts as unsaved; Save reports why.
    const auto fingerprint = GraphFingerprint();
    unsaved_ = !fingerprint.has_value() || *fingerprint != saved_fingerprint_;
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
    namespace st = cyxwiz::studio_toolbar;
    const ui::Tokens& t = ui::CurrentTokens();
    UpdateUnsavedState();

    // ---- The state the bar shows.
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    const bool training_active = training_mgr.IsTrainingActive();
    const auto pipeline_task = pipeline_task_id_ == 0 ? nullptr : cyxwiz::AsyncTaskManager::Instance().GetTask(pipeline_task_id_);
    const bool pipeline_running = execution_mode_ == ExecutionMode::DuckDBPipeline && pipeline_execution_active_ && pipeline_task;
    const bool has_sim = HasSimulationNodes();
    const bool has_rl = HasRLNodes();
    const bool rl_running = rl_script_running_ || (rl_executor_ && rl_executor_->IsTraining());
    const bool sim_graph = has_sim || has_rl;
    const int num_selected = ImNodes::NumSelectedNodes();

    // Compile: the live compile's latest result (TOFIX123); Local Debug: the
    // last run against the graph now (MainWindow).
    st::CompileInputs compile_in;
    compile_in.nodes = static_cast<int>(nodes_.size());
    compile_in.live_check_off = execution_mode_ == ExecutionMode::DuckDBPipeline;
    if (live_compile_) {
        switch (live_compile_->State()) {
            case cyxwiz::LiveCompileState::NotCompiled: compile_in.state = st::CompileState::NotCompiled; break;
            case cyxwiz::LiveCompileState::Compiling: compile_in.state = st::CompileState::Compiling; break;
            case cyxwiz::LiveCompileState::Compiled: compile_in.state = st::CompileState::Compiled; break;
            case cyxwiz::LiveCompileState::Failed: compile_in.state = st::CompileState::Failed; break;
        }
        compile_in.stale = !live_compile_->UpToDate();
        if (const auto* config = live_compile_->Config()) {
            for (const auto& issue : config->issues) {
                if (issue.level == cyxwiz::IssueLevel::Error) ++compile_in.errors;
                else if (issue.level == cyxwiz::IssueLevel::Warning) ++compile_in.warnings;
            }
        }
    }
    const st::StepChip compile_chip = st::CompileChip(compile_in);
    const st::StepChip debug_chip = st::LocalDebugChip(local_debug_status_);
    st::TrainInputs train_in;
    train_in.train_available = static_cast<bool>(train_callback_);
    train_in.graph_valid = IsGraphValid();
    train_in.require_debug = require_debug_before_train_;
    train_in.debug = local_debug_status_;
    const st::TrainButton train = st::TrainState(train_in);
    const auto tone = [&t](st::Tone value) {
        switch (value) {
            case st::Tone::Ok: return t.success;
            case st::Tone::Warning: return t.warning;
            case st::Tone::Error: return t.error;
            case st::Tone::Busy: return t.running;
            case st::Tone::Neutral: break;
        }
        return t.pending;
    };

    // ---- Every group's width in each of its forms (approved mockup).
    const std::string name = current_file_path_.empty() ? std::string("Untitled graph")
                                                       : std::filesystem::path(current_file_path_).filename().string();
    const std::string stem = current_file_path_.empty() ? std::string("Untitled")
                                                       : std::filesystem::path(current_file_path_).stem().string();
    const auto chip_label = [&](const std::string& shown) {
        return std::string(ICON_FA_DIAGRAM_PROJECT "  ") + shown + (unsaved_ ? "    " : "  ") + ICON_FA_CHEVRON_DOWN +
               "##graph_chip";
    };
    const char* kModeFull[] = {ICON_FA_CODE " Code Gen", ICON_FA_SHARE_NODES " Data Pipeline",
                               ICON_FA_CHART_LINE " Local Training"};
    const char* kModeShort[] = {"Code", "Pipeline", "Training"};
    const char* kModeMenu[] = {"Code " ICON_FA_CHEVRON_DOWN "##mode_menu", "Pipeline " ICON_FA_CHEVRON_DOWN "##mode_menu",
                               "Training " ICON_FA_CHEVRON_DOWN "##mode_menu"};
    const int mode = static_cast<int>(execution_mode_);
    const std::string zoom_text = std::to_string(static_cast<int>(zoom_ * 100.0f + 0.5f)) + "%";
    const std::string view_menu_label = "View \xC2\xB7 " + zoom_text + " " ICON_FA_CHEVRON_DOWN "##view_menu";
    const char* view_menu_narrow = "View " ICON_FA_CHEVRON_DOWN "##view_menu";
    const char* edit_menu_label = "Edit " ICON_FA_CHEVRON_DOWN "##edit_menu";
    const std::string compile_full = ICON_FA_CHECK " Compile";
    const std::string debug_full = ICON_FA_BUG " Local Debug";
    const float sp = Spacing();
    const float arrow_w = ImGui::CalcTextSize(ICON_FA_CHEVRON_RIGHT).x + 2.0f * sp;

    st::BarWidths widths;
    widths.fixed = W(chip_label(name).c_str()) + sp + W(ICON_FA_FLOPPY_DISK " Save") + sp +
                   (training_active ? 230.0f : W(ICON_FA_PLAY " Train")) +
                   (pipeline_running ? 150.0f + 50.0f + W(ICON_FA_STOP " Cancel") + 3.0f * sp : 0.0f);
    widths.edit_icons = W(ICON_FA_OBJECT_GROUP) + W(ICON_FA_COPY) + W(ICON_FA_TRASH) + 2.0f * sp;
    widths.edit_menu = W(edit_menu_label);
    widths.view_full = W(ICON_FA_MINUS) + W(zoom_text.c_str()) + W(ICON_FA_PLUS) + W(ICON_FA_EXPAND) + W(ICON_FA_SITEMAP) +
                       4.0f * sp;
    widths.view_menu = W(view_menu_label.c_str());
    float actions_full = 0.0f, actions_icons = 0.0f;
    if (sim_graph) {
        // Simulation and RL graphs: their run group in place of the mode.
        if (has_sim) {
            actions_full += W(ICON_FA_PLAY " Run Sim") + 160.0f + sp;
            actions_icons += W(ICON_FA_PLAY) + 70.0f + sp;
        }
        if (has_rl) {
            actions_full += W(ICON_FA_PLAY " Train RL") + W(ICON_FA_FILE_EXPORT " Export ONNX") + 110.0f + 2.0f * sp;
            actions_icons += W(ICON_FA_PLAY) + W(ICON_FA_FILE_EXPORT) + 90.0f + 2.0f * sp;
        }
    } else {
        widths.mode_full = ui::SegmentedControlWidth(kModeFull, 3);
        widths.mode_short = ui::SegmentedControlWidth(kModeShort, 3);
        widths.mode_menu = W(kModeMenu[mode]);
        if (execution_mode_ == ExecutionMode::CodeGeneration) {
            actions_full = kFrameworkWidth + sp + W(ICON_FA_GEARS " Generate") + sp;
            actions_icons = kFrameworkWidth + sp + W(ICON_FA_GEARS) + sp;
        } else if (execution_mode_ == ExecutionMode::DuckDBPipeline && !pipeline_running) {
            actions_full = W(ICON_FA_PLAY " Execute Pipeline") + sp;
            actions_icons = W(ICON_FA_PLAY) + sp;
        }
        actions_full += W(ICON_FA_FILE_EXPORT) + sp;
        actions_icons += W(ICON_FA_FILE_EXPORT) + sp;
    }
    widths.actions_full = actions_full;
    widths.actions_icons = actions_icons;
    widths.steps_full = ui::StepButtonWidth(compile_full.c_str(), compile_chip.label.c_str()) +
                        ui::StepButtonWidth(debug_full.c_str(), debug_chip.label.c_str()) + 2.0f * arrow_w;
    widths.steps_short = ui::StepButtonWidth(ICON_FA_CHECK, compile_chip.short_label.c_str()) +
                         ui::StepButtonWidth("Debug", debug_chip.short_label.c_str()) + 2.0f * arrow_w;
    widths.steps_dots = ui::StepButtonWidth("", "") * 2.0f + 2.0f * sp;
    widths.gaps = 4.0f * kGroupGap;
    const float avail = ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().WindowPadding.x;
    const int level = st::FoldLevel(widths, avail);
    if (level != toolbar_fold_level_) {
        toolbar_fold_level_ = level;
        spdlog::debug("Studio bar: {:.0f} px; needs {:.0f} / {:.0f} / {:.0f} / {:.0f} (full, short labels, menus, narrowest): form {}",
                      avail, st::BarWidth(widths, 0), st::BarWidth(widths, 1), st::BarWidth(widths, 2),
                      st::BarWidth(widths, 3), level);
    }
    const bool edit_as_menu = level >= 2, view_as_menu = level >= 2, mode_as_menu = level >= 3;
    const bool short_labels = level >= 1, dots = level >= 4;

    // ---- The bar: one row on the toolbar surface.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_bar);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
    ImGui::BeginChild("##studio_nav", ImVec2(0, ImGui::GetFrameHeight() + 12.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    // Left: the graph, Save, edit and view.
    const std::string chip = chip_label(dots ? stem : name);
    const std::string chip_prefix = std::string(ICON_FA_DIAGRAM_PROJECT "  ") + (dots ? stem : name);
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
                                                    ui::ToU32(t.warning));
    }
    DrawGraphMenu();
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_FLOPPY_DISK " Save")) SaveCurrentGraph();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", current_file_path_.empty() ? "Save the graph (Ctrl+S): asks for a file name the first time"
                                                           : ("Save to " + name + " (Ctrl+S)").c_str());

    const char* no_selection = "Select nodes first";
    const auto zoom_in = [this]() { zoom_ = std::min(ZOOM_MAX, zoom_ + 0.1f); };
    const auto zoom_out = [this]() { zoom_ = std::max(ZOOM_MIN, zoom_ - 0.1f); };
    ImGui::SameLine(0.0f, kGroupGap);
    if (edit_as_menu) {
        if (ui::GhostButton(edit_menu_label, true, nullptr, true)) ImGui::OpenPopup("##edit_menu_popup");
    } else {
        if (ui::GhostButton(ICON_FA_OBJECT_GROUP "##select_all", !nodes_.empty(), "The canvas is empty")) SelectAll();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Select all (Ctrl+A)");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_COPY "##duplicate", num_selected > 0, "Duplicate (Ctrl+D): select nodes first"))
            DuplicateSelection();
        if (ImGui::IsItemHovered() && num_selected > 0) ImGui::SetTooltip("Duplicate (Ctrl+D)");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_TRASH "##delete", num_selected > 0, "Delete selected (Delete): select nodes first"))
            DeleteSelected();
        if (ImGui::IsItemHovered() && num_selected > 0) ImGui::SetTooltip("Delete selected (Delete)");
    }
    if (ImGui::BeginPopup("##edit_menu_popup")) {
        if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP "  Select all", "Ctrl+A", false, !nodes_.empty())) SelectAll();
        if (ImGui::MenuItem(ICON_FA_COPY "  Duplicate", "Ctrl+D", false, num_selected > 0)) DuplicateSelection();
        if (num_selected == 0) ImGui::TextColored(t.text_dim, "      %s", no_selection);
        if (ImGui::MenuItem(ICON_FA_TRASH "  Delete selected", "Delete", false, num_selected > 0)) DeleteSelected();
        if (num_selected == 0) ImGui::TextColored(t.text_dim, "      %s", no_selection);
        ImGui::EndPopup();
    }
    ImGui::SameLine(0.0f, kGroupGap);
    if (view_as_menu) {
        if (ui::GhostButton(dots ? view_menu_narrow : view_menu_label.c_str(), true, nullptr, true))
            ImGui::OpenPopup("##view_menu_popup");
    } else {
        if (ui::GhostButton(ICON_FA_MINUS "##zoom_out")) zoom_out();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom out");
        ImGui::SameLine(0.0f, 0.0f);
        if (ui::GhostButton((zoom_text + "##zoom_pct").c_str())) ImGui::OpenPopup("##view_menu_popup");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom options");
        ImGui::SameLine(0.0f, 0.0f);
        if (ui::GhostButton(ICON_FA_PLUS "##zoom_in")) zoom_in();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom in");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_EXPAND "##fit", !nodes_.empty(), "Fit all nodes (F): the canvas is empty")) FrameAll();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fit all nodes (F; with nodes selected, F frames them)");
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_SITEMAP "##minimap", true, nullptr, show_minimap_)) show_minimap_ = !show_minimap_;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(show_minimap_ ? "Minimap (M): shown" : "Minimap (M): hidden");
    }
    if (ImGui::BeginPopup("##view_menu_popup")) {
        if (ImGui::MenuItem("Zoom in", "+ / wheel", false, zoom_ < ZOOM_MAX)) zoom_in();
        if (ImGui::MenuItem("Zoom out", "- / wheel", false, zoom_ > ZOOM_MIN)) zoom_out();
        if (ImGui::MenuItem("Zoom to 100%")) zoom_ = 1.0f;
        if (ImGui::MenuItem("Fit all nodes", "F", false, !nodes_.empty())) FrameAll();
        if (ImGui::MenuItem("Minimap", "M", show_minimap_)) show_minimap_ = !show_minimap_;
        ImGui::Separator();
        std::string counts = st::CountsText(nodes_.size(), links_.size());
        if (num_selected > 0) counts += " \xC2\xB7 " + st::SelectionText(num_selected);
        ImGui::TextColored(t.text_dim, "%s  \xC2\xB7  range 25%% - 200%%", counts.c_str());
        ImGui::EndPopup();
    }
    const float left_end = ImGui::GetItemRectMax().x;

    // Right: Compile > Local Debug > Train (or the epoch and Stop).
    const std::string compile_label = dots ? std::string() : short_labels ? std::string(ICON_FA_CHECK) : compile_full;
    const std::string debug_label = dots ? std::string() : short_labels ? std::string("Debug") : debug_full;
    const std::string compile_status = dots ? std::string() : short_labels ? compile_chip.short_label : compile_chip.label;
    const std::string debug_status = dots ? std::string() : short_labels ? debug_chip.short_label : debug_chip.label;
    const float steps_w = compile_callback_ ? ui::StepButtonWidth(compile_label.c_str(), compile_status.c_str()) +
                                                  ui::StepButtonWidth(debug_label.c_str(), debug_status.c_str()) +
                                                  (dots ? 2.0f * sp : 2.0f * arrow_w)
                                            : 0.0f;
    const float run_w = training_active ? 230.0f : W(ICON_FA_PLAY " Train");
    const float right_w = steps_w + run_w;

    // Middle: the mode and its actions (or the simulation / RL group),
    // centred between the two sides; a running pipeline's progress after it.
    float middle_w = (mode_as_menu ? widths.mode_menu : short_labels ? widths.mode_short : widths.mode_full) +
                     (short_labels ? widths.actions_icons : widths.actions_full) +
                     (pipeline_running ? 150.0f + 50.0f + W(ICON_FA_STOP " Cancel") + 3.0f * sp : 0.0f);
    if (!sim_graph && widths.mode_full > 0.0f) middle_w += sp;
    const float window_x = ImGui::GetWindowPos().x;
    const float right_start = window_x + ImGui::GetWindowContentRegionMax().x - right_w;
    const float middle_start = std::max(left_end + kGroupGap, left_end + (right_start - left_end - middle_w) * 0.5f);
    ImGui::SameLine();
    if (middle_start - window_x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(middle_start - window_x);

    if (!sim_graph) {
        if (mode_as_menu) {
            if (ui::GhostButton(kModeMenu[mode], true, nullptr, true)) ImGui::OpenPopup("##mode_popup");
            if (ImGui::BeginPopup("##mode_popup")) {
                for (int i = 0; i < 3; ++i)
                    if (ImGui::MenuItem(kModeFull[i], nullptr, mode == i)) execution_mode_ = static_cast<ExecutionMode>(i);
                ImGui::EndPopup();
            }
        } else {
            int chosen = mode;
            if (ui::SegmentedControl("##exec_mode", short_labels ? kModeShort : kModeFull, 3, &chosen)) {
                execution_mode_ = static_cast<ExecutionMode>(chosen);
                spdlog::info("Execution mode changed to: {}", kModeShort[chosen]);
            }
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
            if (ui::GhostButton(short_labels ? ICON_FA_GEARS "##generate" : ICON_FA_GEARS " Generate")) GeneratePythonCode();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Generate Python code for the chosen framework");
            ImGui::SameLine();
        } else if (execution_mode_ == ExecutionMode::DuckDBPipeline && !pipeline_running) {
            const ImVec4 run_colour = t.success;
            if (ui::GhostButton(short_labels ? ICON_FA_PLAY "##execute" : ICON_FA_PLAY " Execute Pipeline", true, nullptr,
                                true, ui::ButtonSize::Small, &run_colour))
                ExecuteDataPipeline();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Execute Pipeline: run the data steps (DuckDB / Arrow); training nodes are left to training");
            ImGui::SameLine();
        }
        if (ui::GhostButton(ICON_FA_FILE_EXPORT "##export")) ShowExportDialog();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Export the graph or its code");
        ImGui::SameLine();
    } else {
        if (has_sim) {
            if (is_simulating_) {
                if (ui::DangerButton(short_labels ? ICON_FA_STOP "##stop_sim" : ICON_FA_STOP " Stop Sim")) OnStopSimulation();
                if (graph_executor_) {
                    ImGui::SameLine();
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextColored(t.success, "t = %.2f s", graph_executor_->GetSimTime());
                }
            } else if (ui::GhostButton(short_labels ? ICON_FA_PLAY "##run_sim" : ICON_FA_PLAY " Run Sim", !rl_running,
                                       "Stop RL training first", true)) {
                OnRunSimulation();
            }
            if (ImGui::IsItemHovered() && !rl_running) ImGui::SetTooltip("Run the graph simulation");
            ImGui::SameLine();
        }
        if (has_rl) {
            if (rl_running) {
                if (ui::DangerButton(short_labels ? ICON_FA_STOP "##stop_rl" : ICON_FA_STOP " Stop RL")) OnStopRLTraining();
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                if (rl_script_running_) {
                    ImGui::TextColored(t.success, "Training via Python...");
                } else if (rl_executor_) {
                    const auto m = rl_executor_->GetMetrics();
                    ImGui::TextColored(t.success, "Ep %d | R: %.1f", m.episode_count, m.mean_episode_reward);
                }
            } else if (ui::GhostButton(short_labels ? ICON_FA_PLAY "##train_rl" : ICON_FA_PLAY " Train RL", !is_simulating_,
                                       "Stop graph simulation first", true)) {
                OnStartRLTraining();
            }
            ImGui::SameLine();
            const bool has_trained = rl_executor_ && !rl_executor_->IsTraining() && rl_executor_->GetMetrics().episode_count > 0;
            if (ui::GhostButton(short_labels ? ICON_FA_FILE_EXPORT "##onnx" : ICON_FA_FILE_EXPORT " Export ONNX", has_trained,
                                "Train an RL agent first"))
                export_onnx_dialog_open_ = true;
            ImGui::SameLine();
        }
        // Simulation speed while simulating.
        if (is_simulating_ && last_eval_time_ms_ > 0.0f) {
            const float fps = last_eval_time_ms_ > 0.001f ? 1000.0f / last_eval_time_ms_ : 0.0f;
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(last_eval_time_ms_ < 16.0f ? t.success : t.warning, "%.1f ms (%.0f FPS)", last_eval_time_ms_,
                               fps);
            ImGui::SameLine();
        }
    }

    // A running data pipeline: progress, its step and Cancel stay at every width.
    if (pipeline_running) {
        const auto info = pipeline_task->GetInfo();
        ImGui::AlignTextToFramePadding();
        ImGui::ProgressBar(info.progress, ImVec2(150.0f, 6.0f), "");
        if (ImGui::IsItemHovered() && !info.status_message.empty()) ImGui::SetTooltip("%s", info.status_message.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%.0f%%", info.progress * 100.0f);
        if (ImGui::IsItemHovered() && !info.status_message.empty()) ImGui::SetTooltip("%s", info.status_message.c_str());
        ImGui::SameLine();
        if (ui::DangerButton(ICON_FA_STOP " Cancel")) cyxwiz::AsyncTaskManager::Instance().Cancel(pipeline_task_id_);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cancel the pipeline");
        ImGui::SameLine();
    }

    if (right_start - window_x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right_start - window_x);
    const auto arrow = [&]() {
        ImGui::SameLine(0.0f, dots ? sp : sp);
        if (dots) return;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_faint, ICON_FA_CHEVRON_RIGHT);
        ImGui::SameLine(0.0f, sp);
    };
    if (compile_callback_) {
        const std::string compile_tip = "Compile (F7): validate the graph and open the compile report\n" + compile_chip.headline;
        switch (ui::StepButton("##compile_step", compile_label.c_str(), compile_status.c_str(), tone(compile_chip.tone),
                               compile_tip.c_str(), compile_chip.headline.c_str())) {
            case ui::StepPart::Label:
                spdlog::info("NodeEditor: Compile Graph invoked from toolbar");
                compile_callback_();
                break;
            case ui::StepPart::Status:
                ImGui::OpenPopup("##compile_popup");
                break;
            case ui::StepPart::None:
                break;
        }
        if (ImGui::BeginPopup("##compile_popup")) {
            ImGui::TextColored(tone(compile_chip.tone), ICON_FA_CHECK "  %s", compile_chip.headline.c_str());
            ImGui::TextColored(t.text_dim, compile_in.live_check_off ? "Checked when you press Compile"
                                                                     : "Checked as you edit \xC2\xB7 %s",
                               st::CountsText(nodes_.size(), links_.size()).c_str());
            ImGui::Separator();
            if (ImGui::MenuItem("Open the compile report", "F7")) compile_callback_();
            int first_error_node = -1;
            if (live_compile_ && live_compile_->Config())
                for (const auto& issue : live_compile_->Config()->issues)
                    if (issue.level == cyxwiz::IssueLevel::Error && issue.node_id >= 0) {
                        first_error_node = issue.node_id;
                        break;
                    }
            if (ImGui::MenuItem("Show the first finding on the canvas", nullptr, false, first_error_node >= 0))
                FocusNode(first_error_node);
            ImGui::EndPopup();
        }
        arrow();
        if (debug_callback_) {
            const std::string debug_tip =
                "Local Debug (F6): one forward and backward pass in the Studio Debugger\n" + debug_chip.headline;
            switch (ui::StepButton("##debug_step", debug_label.c_str(), debug_status.c_str(), tone(debug_chip.tone),
                                   debug_tip.c_str(), debug_chip.headline.c_str())) {
                case ui::StepPart::Label:
                    spdlog::info("NodeEditor: Local Debug invoked from toolbar");
                    debug_callback_();
                    break;
                case ui::StepPart::Status:
                    ImGui::OpenPopup("##debug_popup");
                    break;
                case ui::StepPart::None:
                    break;
            }
            if (ImGui::BeginPopup("##debug_popup")) {
                ImGui::TextColored(tone(debug_chip.tone), ICON_FA_BUG "  %s", debug_chip.headline.c_str());
                ImGui::TextColored(t.text_dim, "One synthetic batch: build, forward, loss, backward, step");
                ImGui::Separator();
                if (ImGui::MenuItem("Run Local Debug", "F6", false, !local_debug_status_.running)) debug_callback_();
                if (ImGui::MenuItem("Open the Studio Debugger", nullptr, false, static_cast<bool>(open_debugger_callback_)))
                    open_debugger_callback_();
                ImGui::EndPopup();
            }
            arrow();
        }
    }
    if (training_active) {
        const auto metrics = training_mgr.GetCurrentMetrics();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.success, ICON_FA_SPINNER " Epoch %d / %d", metrics.current_epoch, metrics.total_epochs);
        ImGui::SameLine();
        if (ui::DangerButton(ICON_FA_STOP " Stop")) training_mgr.StopTraining();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stop training");
    } else {
        if (ui::PrimaryButton(ICON_FA_PLAY " Train", train.enabled, train.disabled_reason.c_str(), ui::ButtonSize::Small)) {
            spdlog::info("NodeEditor: Starting training from graph");
            train_callback_(nodes_, links_);
        }
        if (train.enabled && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", train.tooltip.c_str());
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

void NodeEditor::RenderCanvasFooter() {
    namespace st = cyxwiz::studio_toolbar;
    const ui::Tokens& t = ui::CurrentTokens();
    if (canvas_size_.x < 200.0f || canvas_size_.y < 3.0f * kCanvasFooterHeight) return;
    const int num_selected = ImNodes::NumSelectedNodes();
    const std::string block = GetGraphReplacementBlockReason();
    std::string doing;
    if (!block.empty()) {
        doing = "Graph replacement locked: " + block;
    } else if (execution_mode_ == ExecutionMode::CodeGeneration) {
        doing = std::string("Code Gen mode \xC2\xB7 Generate writes ") + kFrameworks[static_cast<int>(selected_framework_)] +
                " code";
    } else if (execution_mode_ == ExecutionMode::DuckDBPipeline) {
        doing = "Data Pipeline mode \xC2\xB7 training nodes run in Train";
    } else {
        doing = "Local Training mode \xC2\xB7 Train runs the whole graph";
    }

    // A window of its own along the canvas bottom, like the minimap: the
    // canvas is a child window drawn over its parent.
    ::gui::NextWindowFollowsCurrent();
    ImGui::SetNextWindowPos(ImVec2(canvas_origin_.x, canvas_origin_.y + canvas_size_.y - kCanvasFooterHeight),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(canvas_size_.x, kCanvasFooterHeight), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, t.bg_bar);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoInputs |
                                   ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("##CanvasFooter", nullptr, flags)) {
        ImGui::SetCursorPosY((kCanvasFooterHeight - ImGui::GetTextLineHeight()) * 0.5f);
        ImGui::TextColored(t.text_dim, "%s", st::CountsText(nodes_.size(), links_.size()).c_str());
        if (num_selected > 0) {
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(t.text_dim, " \xC2\xB7 ");
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(t.accent_text, "%s", st::SelectionText(num_selected).c_str());
        }
        const float doing_w = ImGui::CalcTextSize(doing.c_str()).x;
        ImGui::SameLine();
        const float x = ImGui::GetWindowWidth() - doing_w - 12.0f;
        if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
        ImGui::TextColored(block.empty() ? t.text_dim : t.warning, "%s", doing.c_str());
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

}  // namespace gui
