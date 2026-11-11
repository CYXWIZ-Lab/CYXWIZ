// CyxWiz Studio canvas bar (TOFIX129 A3, approved mockup
// https://claude.ai/artifact/9nAjNG1P6NWCFm9Sf92vT5), one row:
//   the graph | Save | Edit | View | the mode | the framework | Code (or Run
//   pipeline, or the simulation group) | More ... Compile > Local Debug > Train
// Train, or a run's line with Pause / Resume and Stop, sits at the right and
// never folds. What does not fit folds into More in a fixed order
// (core/studio_toolbar_presentation). The counts and what the canvas is
// doing are in the canvas footer (RenderCanvasFooter).
#include "node_editor.h"

#include "icons.h"
#include "separate_windows.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/async_task_manager.h"
#include "../core/graph_executor.h"
#include "../core/live_graph_compile.h"
#include "../core/rl/rl_training_run.h"
#include "../core/rl_graph_compile.h"
#include "../core/studio_toolbar_presentation.h"
#include "../core/training_manager.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>

namespace gui {

namespace {

namespace ui = cyxwiz::ui;
namespace st = cyxwiz::studio_toolbar;

constexpr float kGroupGap = 14.0f;
constexpr const char* kFrameworks[] = {"PyTorch", "TensorFlow", "Keras", "PyCyxWiz"};
const char* const kModeFull[] = {ICON_FA_CODE " Code Gen", ICON_FA_DATABASE " Data Pipeline",
                                 ICON_FA_CHART_LINE " Local Training"};

float W(const char* label) { return ui::ButtonWidth(label, ui::ButtonSize::Small); }
float W(const std::string& label) { return W(label.c_str()); }
float Spacing() { return ImGui::GetStyle().ItemSpacing.x; }

st::Mode ModeOf(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::CodeGeneration: return st::Mode::CodeGen;
        case ExecutionMode::DuckDBPipeline: return st::Mode::DataPipeline;
        default: return st::Mode::LocalTraining;
    }
}

// A menu item that, when it cannot run, stays visible with its reason under it.
bool ItemOrReason(const char* label, const char* shortcut, const st::Availability& available, bool selected = false,
                  const std::string& hint = {}) {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool clicked = ImGui::MenuItem(label, shortcut, selected, available.enabled);
    if (!available.enabled && !available.reason.empty()) {
        ImGui::Indent(ImGui::GetFontSize() * 1.2f);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetFontSize() * 22.0f);
        ImGui::TextColored(t.text_dim, "%s", available.reason.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Unindent(ImGui::GetFontSize() * 1.2f);
    } else if (available.enabled && !hint.empty()) {
        ImGui::Indent(ImGui::GetFontSize() * 1.2f);
        ImGui::TextColored(t.text_dim, "%s", hint.c_str());
        ImGui::Unindent(ImGui::GetFontSize() * 1.2f);
    }
    return clicked;
}

// A section's name inside More.
void Section(const char* name) {
    ImGui::Spacing();
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", name);
}

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
    st::GraphInputs in;
    in.nodes = nodes_.size();
    in.locked = GetGraphReplacementBlockReason();
    const st::GraphMenu menu = st::GraphMenuFor(in, ImNodes::NumSelectedNodes());
    if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK "  Save", "Ctrl+S")) SaveFromStudio();
    if (ImGui::MenuItem(ICON_FA_FILE_EXPORT "  Save As...", "Ctrl+Shift+S")) SaveAsFromStudio();
    if (ItemOrReason(ICON_FA_FOLDER_OPEN "  Load...", nullptr, menu.load)) ShowLoadDialog();
    ImGui::Separator();
    if (ItemOrReason(ICON_FA_COPY "  Export a copy of the graph...", nullptr, menu.export_graph)) ExportGraphCopy();
    if (open_custom_node_editor_callback_ && ImGui::MenuItem(ICON_FA_WAND_MAGIC_SPARKLES "  Custom Node Editor"))
        open_custom_node_editor_callback_();
    if (open_custom_node_editor_callback_ && ImGui::IsItemHovered())
        ImGui::SetTooltip("Create or edit custom node types (pins, parameters, code templates)");
    ImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, menu.clear.enabled ? t.error : t.text_faint);
    const bool clear = ImGui::MenuItem(ICON_FA_ERASER "  Clear the canvas...", nullptr, false, menu.clear.enabled);
    ImGui::PopStyleColor();
    if (!menu.clear.enabled) {
        ImGui::Indent(ImGui::GetFontSize() * 1.2f);
        ImGui::TextColored(t.text_dim, "%s", menu.clear.reason.c_str());
        ImGui::Unindent(ImGui::GetFontSize() * 1.2f);
    }
    if (clear) clear_confirm_requested_ = true;
    ImGui::EndPopup();
}

void NodeEditor::DrawClearConfirm() {
    if (clear_confirm_requested_) {
        ImGui::OpenPopup("Clear the canvas?###clear_canvas");
        clear_confirm_requested_ = false;
    }
    const std::string message = "This removes all " + std::to_string(nodes_.size()) + " nodes and " +
                                std::to_string(links_.size()) +
                                " links from the canvas.\n\nCtrl+Z brings them back. The graph file changes only when you save.";
    if (ui::ConfirmDialog("Clear the canvas?###clear_canvas", message.c_str(), "Clear the canvas", true) ==
        ui::DialogResult::Primary)
        ClearGraph();
}

void NodeEditor::ZoomAroundCentre(int direction) {
    // The grid point at the canvas centre stays at the centre.
    const ImVec2 panning = ImNodes::EditorContextGetPanning();
    const ImVec2 centre((canvas_size_.x * 0.5f - panning.x) / zoom_, (canvas_size_.y * 0.5f - panning.y) / zoom_);
    zoom_ = st::ZoomStep(zoom_, direction);
    CenterViewOn(centre);
}

void NodeEditor::DrawEditItems(int selected) {
    st::GraphInputs in;
    in.nodes = nodes_.size();
    in.locked = GetGraphReplacementBlockReason();
    const st::GraphMenu menu = st::GraphMenuFor(in, selected);
    st::Availability select_all;
    if (nodes_.empty()) select_all = {false, "The canvas is empty."};
    if (ItemOrReason(ICON_FA_OBJECT_GROUP "  Select all", "Ctrl+A", select_all)) SelectAll();
    if (ItemOrReason(ICON_FA_COPY "  Duplicate", "Ctrl+D", menu.duplicate)) DuplicateSelection();
    if (ItemOrReason(ICON_FA_TRASH "  Delete selected", "Delete", menu.remove)) DeleteSelected();
}

void NodeEditor::DrawViewItems() {
    const ui::Tokens& t = ui::CurrentTokens();
    const st::Availability out = st::ZoomOut(zoom_), in = st::ZoomIn(zoom_);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Zoom");
    ImGui::SameLine(ImGui::GetFontSize() * 5.0f);
    if (ui::GhostButton(ICON_FA_MINUS "##zoom_out", out.enabled, out.reason.c_str())) ZoomAroundCentre(-1);
    if (out.enabled && ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom out (-); the view keeps its centre");
    ImGui::SameLine(0.0f, 0.0f);
    const std::string pct = st::ZoomText(zoom_);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text, " %s ", pct.c_str());
    ImGui::SameLine(0.0f, 0.0f);
    if (ui::GhostButton(ICON_FA_PLUS "##zoom_in", in.enabled, in.reason.c_str())) ZoomAroundCentre(1);
    if (in.enabled && ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom in (+); the view keeps its centre");
    st::Availability hundred;
    if (pct == "100%") hundred = {false, "Already at 100%."};
    if (ItemOrReason("Zoom to 100%", nullptr, hundred)) {
        const ImVec2 panning = ImNodes::EditorContextGetPanning();
        const ImVec2 centre((canvas_size_.x * 0.5f - panning.x) / zoom_, (canvas_size_.y * 0.5f - panning.y) / zoom_);
        zoom_ = 1.0f;
        CenterViewOn(centre);
    }
    st::Availability fit;
    if (nodes_.empty()) fit = {false, "The canvas is empty."};
    if (ItemOrReason("Fit all nodes", "F", fit)) FrameAll();
    if (ImGui::MenuItem("Minimap", "M", show_minimap_)) show_minimap_ = !show_minimap_;
    if (ImGui::MenuItem("Find in the graph", "Ctrl+F")) OpenFindBar();
}

void NodeEditor::DrawFrameworkItems() {
    for (int i = 0; i < 4; ++i)
        if (ImGui::MenuItem(kFrameworks[i], nullptr, static_cast<int>(selected_framework_) == i)) {
            selected_framework_ = static_cast<CodeFramework>(i);
            code_result_.clear();
            spdlog::info("Code generation framework: {}", kFrameworks[i]);
        }
}

void NodeEditor::DrawCodeItems() {
    const ui::Tokens& t = ui::CurrentTokens();
    st::CodeInputs in;
    in.mode = ModeOf(execution_mode_);
    in.framework = kFrameworks[static_cast<int>(selected_framework_)];
    in.graph = current_file_path_.empty() ? std::string() : GetGraphStem();
    in.problem = CodeProblem();
    const st::CodeMenu menu = st::CodeMenuFor(in);
    if (ItemOrReason(menu.generate_label.c_str(), nullptr, menu.available, false, menu.generate_hint))
        GenerateCodeIntoEditor();
    if (ItemOrReason(menu.save_label.c_str(), nullptr, menu.available, false, menu.save_hint)) SaveCodeAsPython();
    if (!menu.framework_on_bar)
        ImGui::TextColored(t.text_faint, "The framework is chosen in Code Gen mode (%s now).", in.framework.c_str());
}

void NodeEditor::DrawStepItems(bool compile, bool debug) {
    const ui::Tokens& t = ui::CurrentTokens();
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
    if (compile && compile_callback_) {
        st::CompileInputs compile_in;
        compile_in.nodes = static_cast<int>(nodes_.size());
        compile_in.live_check_off = execution_mode_ == ExecutionMode::DuckDBPipeline;
        int first_error_node = -1;
        if (live_compile_) {
            switch (live_compile_->State()) {
                case cyxwiz::LiveCompileState::NotCompiled: compile_in.state = st::CompileState::NotCompiled; break;
                case cyxwiz::LiveCompileState::Compiling: compile_in.state = st::CompileState::Compiling; break;
                case cyxwiz::LiveCompileState::Compiled: compile_in.state = st::CompileState::Compiled; break;
                case cyxwiz::LiveCompileState::Failed: compile_in.state = st::CompileState::Failed; break;
            }
            compile_in.stale = !live_compile_->UpToDate();
            if (const auto* config = live_compile_->Config())
                for (const auto& issue : config->issues) {
                    if (issue.level == cyxwiz::IssueLevel::Error) {
                        ++compile_in.errors;
                        if (first_error_node < 0 && issue.node_id >= 0) first_error_node = issue.node_id;
                    } else if (issue.level == cyxwiz::IssueLevel::Warning) {
                        ++compile_in.warnings;
                    }
                }
        }
        const st::StepChip chip = st::CompileChip(compile_in);
        ImGui::TextColored(tone(chip.tone), ICON_FA_CHECK "  %s", chip.headline.c_str());
        ImGui::TextColored(t.text_dim, compile_in.live_check_off ? "Checked when you press Compile"
                                                                 : "Checked as you edit \xC2\xB7 %s",
                           st::CountsText(nodes_.size(), links_.size()).c_str());
        if (ImGui::MenuItem("Open the compile report", "F7")) compile_callback_();
        st::Availability first;
        if (first_error_node < 0) first = {false, "No finding on a node."};
        if (ItemOrReason("Show the first finding on the canvas", nullptr, first)) FocusNode(first_error_node);
    }
    if (debug && debug_callback_) {
        const st::StepChip chip = st::LocalDebugChip(local_debug_status_);
        ImGui::TextColored(tone(chip.tone), ICON_FA_BUG "  %s", chip.headline.c_str());
        ImGui::TextColored(t.text_dim, "One synthetic batch: build, forward, loss, backward, step");
        if (ItemOrReason("Run Local Debug", "F6", st::LocalDebugStart(local_debug_status_, nodes_.size())))
            debug_callback_();
        st::Availability open;
        if (!open_debugger_callback_) open = {false, "The Studio Debugger is not available."};
        if (ItemOrReason("Open the Studio Debugger", nullptr, open)) open_debugger_callback_();
    }
}

void NodeEditor::ShowToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    UpdateUnsavedState();

    // ---- The state the bar shows.
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    const bool training_active = training_mgr.IsTrainingActive();
    const bool training_paused = training_active && training_mgr.IsPaused();
    const auto pipeline_task = pipeline_task_id_ == 0 ? nullptr : cyxwiz::AsyncTaskManager::Instance().GetTask(pipeline_task_id_);
    const bool pipeline_running = execution_mode_ == ExecutionMode::DuckDBPipeline && pipeline_execution_active_ && pipeline_task;
    const bool sim_graph = HasSimulationNodes();
    const bool rl_active = IsRLTrainingActive();  // the Engine's PPO on an RL graph
    const bool rl_paused = rl_active && rl_run_ && rl_run_->Paused();
    const int num_selected = ImNodes::NumSelectedNodes();
    const st::Mode mode = ModeOf(execution_mode_);
    const std::string locked = GetGraphReplacementBlockReason();

    st::CompileInputs compile_in;
    compile_in.nodes = static_cast<int>(nodes_.size());
    compile_in.live_check_off = mode == st::Mode::DataPipeline;
    if (live_compile_) {
        switch (live_compile_->State()) {
            case cyxwiz::LiveCompileState::NotCompiled: compile_in.state = st::CompileState::NotCompiled; break;
            case cyxwiz::LiveCompileState::Compiling: compile_in.state = st::CompileState::Compiling; break;
            case cyxwiz::LiveCompileState::Compiled: compile_in.state = st::CompileState::Compiled; break;
            case cyxwiz::LiveCompileState::Failed: compile_in.state = st::CompileState::Failed; break;
        }
        compile_in.stale = !live_compile_->UpToDate();
        if (const auto* config = live_compile_->Config())
            for (const auto& issue : config->issues) {
                if (issue.level == cyxwiz::IssueLevel::Error) ++compile_in.errors;
                else if (issue.level == cyxwiz::IssueLevel::Warning) ++compile_in.warnings;
            }
    }
    const st::StepChip compile_chip = st::CompileChip(compile_in);
    const st::StepChip debug_chip = st::LocalDebugChip(local_debug_status_);
    const st::Availability debug_start = st::LocalDebugStart(local_debug_status_, nodes_.size());
    st::TrainInputs train_in;
    train_in.train_available = static_cast<bool>(train_callback_);
    train_in.graph_valid = IsGraphValid();
    // an RL graph has no batch to debug (TOFIX195 S4.3)
    const bool rl_graph = std::any_of(nodes_.begin(), nodes_.end(), [](const MLNode& n) { return cyxwiz::IsRLGraphNode(n.type); });
    train_in.require_debug = require_debug_before_train_ && !rl_graph;
    train_in.rl_graph = rl_graph;
    train_in.debug = local_debug_status_;
    // What keeps the graph busy before a run can start (a Data Input loading, a test, a pipeline)
    if (!training_active && !rl_active) train_in.busy = locked;
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

    // ---- Every item's width (approved mockup).
    const std::string name = current_file_path_.empty() ? std::string("Untitled graph")
                                                       : std::filesystem::path(current_file_path_).filename().string();
    const std::string chip = std::string(ICON_FA_DIAGRAM_PROJECT "  ") + name + (unsaved_ ? "    " : "  ") +
                             ICON_FA_CHEVRON_DOWN "##graph_chip";
    const std::string zoom_text = st::ZoomText(zoom_);
    const std::string view_full = "View \xC2\xB7 " + zoom_text + " " ICON_FA_CHEVRON_DOWN "##view_menu";
    const char* view_short = "View " ICON_FA_CHEVRON_DOWN "##view_menu";
    const char* edit_label = "Edit " ICON_FA_CHEVRON_DOWN "##edit_menu";
    const char* code_label = ICON_FA_CODE " Code " ICON_FA_CHEVRON_DOWN "##code_menu";
    const char* more_label = ICON_FA_ELLIPSIS " More " ICON_FA_CHEVRON_DOWN "##more_menu";
    const char* run_pipeline_label = ICON_FA_PLAY " Execute Pipeline##run_pipeline";
    const std::string mode_menu = std::string(kModeFull[static_cast<int>(execution_mode_)]) + " " ICON_FA_CHEVRON_DOWN "##mode_menu";
    const std::string framework_label =
        std::string(kFrameworks[static_cast<int>(selected_framework_)]) + " " ICON_FA_CHEVRON_DOWN "##framework_menu";
    const std::string compile_full = ICON_FA_CHECK " Compile";
    const std::string debug_full = ICON_FA_BUG " Local Debug";
    const float sp = Spacing();
    const float arrow_w = ImGui::CalcTextSize(ICON_FA_CHEVRON_RIGHT).x + 2.0f * sp;

    // The run group at its widest, so the bar does not refold as a run's line grows.
    const auto metrics = training_active ? training_mgr.GetCurrentMetrics() : cyxwiz::TrainingMetrics{};
    const bool waiting_gpu = training_active && !metrics.waiting_for_gpu.empty();
    const char* pause_label = ICON_FA_PAUSE " Pause##run_pause";
    const char* resume_label = ICON_FA_PLAY " Resume##run_resume";
    const float pause_w = std::max(W(pause_label), W(resume_label)) + sp;
    const float stop_w = W(ICON_FA_STOP " Stop##run_stop") + sp;
    const float spinner_w = ImGui::CalcTextSize(ICON_FA_SPINNER " ").x;
    const std::string rl_widest = std::max(rl_status_, std::string("PPO update 0000 \xC2\xB7 0,000,000 steps \xC2\xB7 OpenCL"),
                                           [](const std::string& a, const std::string& b) { return a.size() < b.size(); });
    st::BarWidths w;
    w.graph = W(chip);
    if (training_active) {
        w.run = spinner_w + ImGui::CalcTextSize("Paused at epoch 0000 / 0000").x + sp + pause_w + stop_w;
        w.run_compact = spinner_w + ImGui::CalcTextSize("Paused 0000 / 0000").x + sp + pause_w + stop_w;
    } else if (rl_active) {
        w.run = spinner_w + ImGui::CalcTextSize((rl_widest + " \xC2\xB7 Paused").c_str()).x + sp + pause_w + stop_w;
        w.run_compact = spinner_w + ImGui::CalcTextSize("Update 0000 \xC2\xB7 Paused").x + sp + pause_w + stop_w;
    } else {
        w.run = w.run_compact = W(ICON_FA_PLAY " Train");
    }
    w.progress = pipeline_running ? 150.0f + 50.0f + W(ICON_FA_STOP " Cancel") + 3.0f * sp : 0.0f;
    w.more = W(more_label);
    w.save = W(ICON_FA_FLOPPY_DISK " Save");
    w.edit = W(edit_label);
    w.view_full = W(view_full);
    w.view_short = W(view_short);
    if (!sim_graph) {
        w.mode_full = ui::SegmentedControlWidth(kModeFull, 3);
        w.mode_menu = W(mode_menu);
        w.code = W(code_label);
        if (mode == st::Mode::CodeGen) w.framework = W(framework_label);
        if (mode == st::Mode::DataPipeline && !pipeline_running) w.run_pipeline = W(run_pipeline_label);
    } else if (simulation_session_.IsActive()) {
        // A run: Pause or Resume, Step, Stop Sim, the time and "Paused"
        const float clock = W("t = 0000.00 s") + W("Paused") + sp;
        w.sim_full = W(ICON_FA_PLAY " Resume") + W(ICON_FA_FORWARD_STEP " Step") + W(ICON_FA_STOP " Stop Sim") + clock + 4.0f * sp;
        w.sim_icons = W(ICON_FA_PLAY) + W(ICON_FA_FORWARD_STEP) + W(ICON_FA_STOP) + clock + 4.0f * sp;
    } else {
        // Run Sim, and the warning of a failed run
        w.sim_full = W(ICON_FA_PLAY " Run Sim") + W(ICON_FA_TRIANGLE_EXCLAMATION) + 2.0f * sp;
        w.sim_icons = W(ICON_FA_PLAY) + W(ICON_FA_TRIANGLE_EXCLAMATION) + 2.0f * sp;
    }
    if (compile_callback_) {
        w.compile = ui::StepButtonWidth(compile_full.c_str(), compile_chip.label.c_str()) + arrow_w;
        if (debug_callback_) w.debug = ui::StepButtonWidth(debug_full.c_str(), debug_chip.label.c_str()) + arrow_w;
    }
    w.gap = kGroupGap;
    const float avail = ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().WindowPadding.x;
    const int level = st::FoldLevel(w, avail);
    if (level != toolbar_fold_level_) {
        toolbar_fold_level_ = level;
        spdlog::debug("Studio bar: {:.0f} px; needs {:.0f} at full width, {:.0f} at the narrowest: fold {}", avail,
                      st::BarWidth(w, 0), st::BarWidth(w, st::kMaxFold), level);
    }
    const st::BarFold fold = st::FoldAt(level);
    const bool show_more = st::ItemsInMore(w, fold) > 0;

    // ---- The bar: one row on the toolbar surface.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_bar);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
    ImGui::BeginChild("##studio_nav", ImVec2(0, ImGui::GetFrameHeight() + 12.0f), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    const auto gap = []() { ImGui::SameLine(0.0f, kGroupGap); };

    // The graph: its name, the unsaved mark and its menu.
    const std::string chip_prefix = std::string(ICON_FA_DIAGRAM_PROJECT "  ") + name;
    const ImVec2 chip_min = ImGui::GetCursorScreenPos();
    if (ui::GhostButton(chip.c_str(), true, nullptr, true)) ImGui::OpenPopup("##graph_menu");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s%s", current_file_path_.empty() ? "Not saved to a file yet" : current_file_path_.c_str(),
                          unsaved_ && !current_file_path_.empty() ? "\nChanged since it was saved" : "");
    if (unsaved_) {
        const float pad = (W(chip) - ImGui::CalcTextSize(chip.c_str(), nullptr, true).x - 2.0f) * 0.5f;
        const float x = chip_min.x + pad + 1.0f + ImGui::CalcTextSize(chip_prefix.c_str()).x +
                        ImGui::CalcTextSize("    ").x * 0.5f;
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(x, chip_min.y + ImGui::GetFrameHeight() * 0.5f), 3.5f,
                                                    ui::ToU32(t.warning));
    }
    DrawGraphMenu();
    if (!fold.save_in_more) {
        ImGui::SameLine();
        if (ui::GhostButton(ICON_FA_FLOPPY_DISK " Save")) SaveFromStudio();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", current_file_path_.empty() ? "Save the graph (Ctrl+S): asks for a file name the first time"
                                                               : ("Save to " + name + " (Ctrl+S)").c_str());
    }
    if (!fold.edit_in_more) {
        gap();
        if (ui::GhostButton(edit_label, true, nullptr, true)) ImGui::OpenPopup("##edit_menu_popup");
    }
    if (ImGui::BeginPopup("##edit_menu_popup")) {
        DrawEditItems(num_selected);
        ImGui::EndPopup();
    }
    if (!fold.view_in_more) {
        gap();
        if (ui::GhostButton(fold.view_short ? view_short : view_full.c_str(), true, nullptr, true))
            ImGui::OpenPopup("##view_menu_popup");
    }
    if (ImGui::BeginPopup("##view_menu_popup")) {
        DrawViewItems();
        ImGui::EndPopup();
    }

    // The mode, the framework and Code (or the simulation group).
    if (!sim_graph) {
        gap();
        int chosen = static_cast<int>(execution_mode_);
        if (fold.mode_menu) {
            if (ui::GhostButton(mode_menu.c_str(), true, nullptr, true)) ImGui::OpenPopup("##mode_popup");
            if (ImGui::BeginPopup("##mode_popup")) {
                for (int i = 0; i < 3; ++i)
                    if (ImGui::MenuItem(kModeFull[i], nullptr, chosen == i)) chosen = i;
                ImGui::EndPopup();
            }
        } else {
            ui::SegmentedControl("##exec_mode", kModeFull, 3, &chosen);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How Train and Code treat this graph:\nCode Gen: Code writes Python for the chosen framework\n"
                              "Data Pipeline: Execute Pipeline runs the data steps; training nodes run in Train\n"
                              "Local Training: Train runs the whole graph on this machine");
        if (chosen != static_cast<int>(execution_mode_)) {
            execution_mode_ = static_cast<ExecutionMode>(chosen);
            code_result_.clear();
            spdlog::info("Execution mode: {}", st::ModeName(ModeOf(execution_mode_)));
        }
        if (mode == st::Mode::CodeGen && !fold.framework_in_more) {
            ImGui::SameLine();
            if (ui::GhostButton(framework_label.c_str(), true, nullptr, true)) ImGui::OpenPopup("##framework_popup");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Code writes Python for this framework");
            if (ImGui::BeginPopup("##framework_popup")) {
                DrawFrameworkItems();
                ImGui::EndPopup();
            }
        }
        if (!fold.code_in_more) {
            ImGui::SameLine();
            if (ui::GhostButton(code_label, true, nullptr, true)) ImGui::OpenPopup("##code_popup");
            if (w.run_pipeline > 0.0f) {
                ImGui::SameLine();
                const ImVec4 run_colour = t.success;
                if (ui::GhostButton(run_pipeline_label, true, nullptr, true, ui::ButtonSize::Small, &run_colour))
                    ExecuteDataPipeline();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Runs the data steps (DuckDB / Arrow) and shows their results; training nodes run in Train");
            }
        }
        if (ImGui::BeginPopup("##code_popup")) {
            DrawCodeItems();
            ImGui::EndPopup();
        }
    } else {
        gap();
        // The graph run's commands, as the Simulation menu and the Simulation Monitor offer them
        using cyxwiz::simulation::Command;
        const auto sim = GetSimulationCommands();
        const auto sim_state = simulation_session_.GetState();
        const bool sim_running = sim_state == cyxwiz::SimulationState::Running;
        const bool icons = fold.sim_icons;
        const auto button_label = [icons](const char* icon, const std::string& text, const char* id) {
            return icons ? std::string(icon) + "##" + id : std::string(icon) + " " + text + "##" + id;
        };
        const auto hint = [](const cyxwiz::simulation::CommandView& view) {
            if (view.enabled && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", view.hint.c_str());
            ImGui::SameLine();
        };
        if (sim_running) {
            if (ui::GhostButton(button_label(ICON_FA_PAUSE, sim.pause.label, "pause_sim").c_str()))
                RunSimulationCommand(Command::Pause);
            hint(sim.pause);
        } else {
            const ImVec4 run_colour = t.success;
            if (ui::GhostButton(button_label(ICON_FA_PLAY, sim.run.label, "run_sim").c_str(), sim.run.enabled,
                                sim.run.disabled_reason.c_str(), true, ui::ButtonSize::Small, &run_colour))
                RunSimulationCommand(Command::Run);
            hint(sim.run);
        }
        if (simulation_session_.IsActive()) {
            if (ui::GhostButton(button_label(ICON_FA_FORWARD_STEP, "Step", "step_sim").c_str()))
                RunSimulationCommand(Command::Step);
            hint(sim.step);
            if (ui::DangerButton(icons ? ICON_FA_STOP "##stop_sim" : ICON_FA_STOP " Stop Sim"))
                RunSimulationCommand(Command::Stop);
            hint(sim.stop);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(sim_running ? t.running : t.pending, "t = %.2f s", simulation_session_.GetSimTime());
            if (!sim_running) {
                ImGui::SameLine();
                ImGui::TextColored(t.pending, "Paused");
            }
        }
        // The last run failed or was refused: the reason on hover.
        if (simulation_session_.GetState() == cyxwiz::SimulationState::Failed) {
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", simulation_session_.GetError().c_str());
        }
    }

    // A running data pipeline: progress, its step and Cancel stay at every width.
    if (pipeline_running) {
        gap();
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
    }

    // More: what does not fit, in the order it folded.
    if (show_more) {
        gap();
        if (ui::GhostButton(more_label, true, nullptr, true)) ImGui::OpenPopup("##more_popup");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("What does not fit on the bar at this width");
    }
    if (ImGui::BeginPopup("##more_popup")) {
        if (fold.debug_in_more && w.debug > 0.0f) {
            Section("Local Debug");
            DrawStepItems(false, true);
        }
        if (fold.compile_in_more && w.compile > 0.0f) {
            Section("Compile");
            DrawStepItems(true, false);
        }
        if (fold.view_in_more) {
            Section("View");
            DrawViewItems();
        }
        if (fold.edit_in_more) {
            Section("Edit");
            DrawEditItems(num_selected);
        }
        if (fold.code_in_more && w.code > 0.0f) {
            Section("Code");
            DrawCodeItems();
            if (w.run_pipeline > 0.0f && ImGui::MenuItem(ICON_FA_PLAY "  Execute Pipeline")) ExecuteDataPipeline();
        }
        if (fold.framework_in_more && w.framework > 0.0f) {
            Section("Framework");
            DrawFrameworkItems();
        }
        if (fold.save_in_more) {
            Section("Graph");
            if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK "  Save", "Ctrl+S")) SaveFromStudio();
        }
        ImGui::EndPopup();
    }

    // Right: Compile > Local Debug > the run group.
    const bool compile_on_bar = w.compile > 0.0f && !fold.compile_in_more;
    const bool debug_on_bar = w.debug > 0.0f && !fold.debug_in_more;
    const float run_w = fold.run_compact ? w.run_compact : w.run;
    const float right_w = (compile_on_bar ? w.compile : 0.0f) + (debug_on_bar ? w.debug : 0.0f) + run_w;
    const float window_x = ImGui::GetWindowPos().x;
    const float right_start = window_x + ImGui::GetWindowContentRegionMax().x - right_w;
    ImGui::SameLine();
    if (right_start - window_x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right_start - window_x);
    const auto arrow = [&]() {
        ImGui::SameLine(0.0f, sp);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_faint, ICON_FA_CHEVRON_RIGHT);
        ImGui::SameLine(0.0f, sp);
    };
    if (compile_on_bar) {
        const std::string compile_tip = "Compile (F7): validate the graph and open the compile report\n" + compile_chip.headline;
        switch (ui::StepButton("##compile_step", compile_full.c_str(), compile_chip.label.c_str(), tone(compile_chip.tone),
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
            DrawStepItems(true, false);
            ImGui::EndPopup();
        }
        arrow();
    }
    if (debug_on_bar) {
        const std::string debug_tip = "Local Debug (F6): one forward and backward pass in the Studio Debugger\n" +
                                      (debug_start.enabled ? debug_chip.headline : debug_start.reason);
        switch (ui::StepButton("##debug_step", debug_full.c_str(), debug_chip.label.c_str(), tone(debug_chip.tone),
                               debug_tip.c_str(), debug_chip.headline.c_str())) {
            case ui::StepPart::Label:
                // Never a second run while one goes (the label stays clickable for its tooltip)
                if (debug_start.enabled) {
                    spdlog::info("NodeEditor: Local Debug invoked from toolbar");
                    debug_callback_();
                }
                break;
            case ui::StepPart::Status:
                ImGui::OpenPopup("##debug_popup");
                break;
            case ui::StepPart::None:
                break;
        }
        if (ImGui::BeginPopup("##debug_popup")) {
            DrawStepItems(false, true);
            ImGui::EndPopup();
        }
        arrow();
    }

    // The run group: Train, or the run's line with Pause / Resume and Stop.
    if (training_active || rl_active) {
        ImGui::AlignTextToFramePadding();
        if (training_active) {
            const std::string line = st::TrainingRunText(metrics.current_epoch, metrics.total_epochs, training_paused,
                                                         waiting_gpu, fold.run_compact);
            const ImVec4 colour = training_paused || waiting_gpu ? t.pending : t.success;
            ImGui::TextColored(colour, "%s %s", waiting_gpu ? ICON_FA_CLOCK : training_paused ? ICON_FA_PAUSE : ICON_FA_SPINNER,
                               line.c_str());
            if (ImGui::IsItemHovered() && waiting_gpu) ImGui::SetTooltip("%s", metrics.waiting_for_gpu.c_str());
        } else {
            const std::string& rl_line = fold.run_compact ? st::CompactRunText(rl_status_) : rl_status_;
            if (rl_paused)  // as the dashboard says it
                ImGui::TextColored(t.pending, ICON_FA_PAUSE " %s \xC2\xB7 Paused", rl_line.c_str());
            else
                ImGui::TextColored(t.success, ICON_FA_SPINNER " %s", rl_line.c_str());
            if (ImGui::IsItemHovered()) {  // the shortened line gives its whole text here
                if (fold.run_compact && !rl_device_tooltip_.empty())
                    ImGui::SetTooltip("%s\n%s", rl_status_.c_str(), rl_device_tooltip_.c_str());
                else if (fold.run_compact)
                    ImGui::SetTooltip("%s", rl_status_.c_str());
                else if (!rl_device_tooltip_.empty())
                    ImGui::SetTooltip("%s", rl_device_tooltip_.c_str());
            }
        }
        const bool paused = training_active ? training_paused : rl_paused;
        ImGui::SameLine();
        if (paused) {
            if (ui::GhostButton(resume_label)) {
                if (training_active) training_mgr.ResumeTraining();
                else rl_run_->Resume();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Resume the run");
        } else {
            if (ui::GhostButton(pause_label, !waiting_gpu, "The run has not started: it waits for the GPU")) {
                if (training_active) training_mgr.PauseTraining();
                else rl_run_->Pause();
            }
            if (!waiting_gpu && ImGui::IsItemHovered()) ImGui::SetTooltip("Pause after the current step; Resume goes on from there");
        }
        ImGui::SameLine();
        if (ui::DangerButton(ICON_FA_STOP " Stop##run_stop")) {
            if (training_active) training_mgr.StopTraining();
            else StopRLTraining();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(training_active ? "Stop training" : "Stop PPO training");
    } else {
        if (ui::PrimaryButton(ICON_FA_PLAY " Train", train.enabled, train.disabled_reason.c_str(), ui::ButtonSize::Small)) {
            spdlog::info("NodeEditor: Starting training from graph");
            train_callback_(nodes_, links_);
        }
        if (train.enabled && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", train.tooltip.c_str());
    }
    ImGui::EndChild();

    // What Generate or Save as .py wrote, until the next one or Hide.
    if (!code_result_.empty()) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(code_result_ok_ ? t.success : t.warning, code_result_ok_ ? ICON_FA_CHECK : ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::PushTextWrapPos(ImGui::GetContentRegionMax().x - W("Hide##code_result") - 2.0f * sp);
        ImGui::TextColored(t.text_dim, "%s", code_result_.c_str());
        ImGui::PopTextWrapPos();
        ImGui::SameLine();
        if (ui::LinkButton("Hide##code_result")) code_result_.clear();
    }

    DrawClearConfirm();
}

void NodeEditor::RenderCanvasFooter() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (canvas_size_.x < 200.0f || canvas_size_.y < 3.0f * kCanvasFooterHeight) return;
    const int num_selected = ImNodes::NumSelectedNodes();
    const st::Footer footer = st::FooterFor(ModeOf(execution_mode_), kFrameworks[static_cast<int>(selected_framework_)],
                                            GetGraphReplacementBlockReason());

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
        // The right side, cut with "..." when it would run into the counts.
        ImGui::SameLine(0.0f, 24.0f);
        const float left = ImGui::GetCursorPosX();
        const float room = ImGui::GetWindowWidth() - 12.0f - left;
        const float text_w = ImGui::CalcTextSize(footer.text.c_str()).x;
        const ImVec4 colour = footer.locked ? t.warning : t.text_dim;
        if (room > 0.0f) {
            ImGui::SetCursorPosX(left + std::max(0.0f, room - text_w));
            const ImVec2 min = ImGui::GetCursorScreenPos();
            const ImVec2 max(min.x + std::min(room, text_w), min.y + ImGui::GetTextLineHeight());
            ImGui::PushStyleColor(ImGuiCol_Text, colour);
            ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), min, max, max.x, max.x, footer.text.c_str(), nullptr,
                                      nullptr);
            ImGui::PopStyleColor();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

}  // namespace gui
