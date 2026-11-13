// Studio Debugger > Layer capture (TOFIX178 6.2d, approved Capture mockup):
// the toolbar control that sets what this session's runs capture, starting
// from the user's defaults in Preferences > Studio Debugger, with the size of
// one snapshot from the compiled model and Save as my default.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"
#include "layer_capture_controls.h"

#include "../../core/count_text.h"
#include "../../core/data_input_presentation.h"
#include "../../core/layer_capture_settings.h"
#include "../../core/live_graph_compile.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <string>

namespace cyxwiz {

namespace {

constexpr const char* kCapturePopup = "StudioDebuggerCapturePopup";

void SizeRow(const std::string& label, uint64_t bytes, bool total = false) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    if (total) ImGui::PushStyleColor(ImGuiCol_Text, DebuggerBright());
    ImGui::TextUnformatted(label.c_str());
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(data_input_view::FormatSize(bytes).c_str());
    if (total) ImGui::PopStyleColor();
}

void WrappedNote(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(DebuggerMuted(), "%s", text);
    ImGui::PopTextWrapPos();
}

}  // namespace

core::StudioDebuggerConfig StudioDebuggerPanel::CaptureSettings() const {
    return capture_session_ ? *capture_session_ : core::EngineConfig::Instance().GetStudioDebugger();
}

void StudioDebuggerPanel::ChangeSessionSettings(const std::function<void(core::StudioDebuggerConfig&)>& change) {
    if (!capture_session_) capture_session_ = core::EngineConfig::Instance().GetStudioDebugger();
    change(*capture_session_);
}

void StudioDebuggerPanel::RenderSaveViewDefaults(
    const char* id, const char* already,
    const std::function<void(const core::StudioDebuggerConfig&, core::StudioDebuggerConfig&)>& copy) {
    auto& engine_config = core::EngineConfig::Instance();
    const core::StudioDebuggerConfig saved = engine_config.GetStudioDebugger();
    core::StudioDebuggerConfig defaults = saved;
    copy(CaptureSettings(), defaults);
    const bool differs = !(defaults == saved);
    if (ui::LinkButton((std::string("Save as my default##") + id).c_str(), differs)) {
        engine_config.SetStudioDebugger(defaults);
        engine_config.Save();
    }
    if (!differs && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", already);
}

void StudioDebuggerPanel::RenderCaptureControl() {
    core::StudioDebuggerConfig settings = CaptureSettings();
    const bool on = layer_capture::CapturesAnything(settings, false);
    const std::string label = std::string(ICON_FA_LAYER_GROUP " Layer capture: ") + layer_capture::Summary(settings) +
                              "##StudioDebuggerCapture";
    // Beside the sample stepper when it fits, else on the next line: never clipped.
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ui::ButtonWidth(label.c_str(), ui::ButtonSize::Small)) ImGui::NewLine();
    if (ui::GhostButton(label.c_str(), true, nullptr, on)) ImGui::OpenPopup(kCapturePopup);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
        ImGui::SetTooltip("What this session's Local Debug and training runs keep of every layer");

    // At most the window's height: past it the popup scrolls, never off screen.
    ImGui::SetNextWindowSizeConstraints(ImVec2(440.0f, 0.0f),
                                        ImVec2(560.0f, ImGui::GetMainViewport()->WorkSize.y * 0.9f));
    if (!ImGui::BeginPopup(kCapturePopup)) return;

    CardHeading("Layer capture");
    ImGui::Spacing();
    if (RenderLayerCaptureControls(settings)) capture_session_ = settings;

    if (layer_capture::CapturesAnything(settings, false)) {
        ImGui::Spacing();
        const ModelMemoryEstimate* model =
            live_compile_ && live_compile_->ParametersCounted() ? &live_compile_->Memory() : nullptr;
        const auto size = model ? layer_capture::EstimateSize(*model, settings, LayerCaptureOptions{}.byte_budget)
                                : layer_capture::SizeEstimate{};
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::CurrentTokens().bg_input);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
        ImGui::BeginChild("CaptureSize", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        // Wrapping is pushed around the notes only: inside the table it would
        // wrap the size column at its own edge.
        if (!size.available) {
            WrappedNote("The size shows once the graph on the canvas compiles and its model is built.");
        } else {
            ImGui::TextColored(DebuggerMuted(), "Size, from this model's compiled shapes");
            if (ImGui::BeginTable("CaptureSizeRows", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed,
                                        ImGui::CalcTextSize("1023.9 MB").x);
                if (settings.capture_activations)
                    SizeRow("Activations (" + CountText(size.activation_values) + " values)", size.activation_bytes);
                if (settings.capture_parameters)
                    SizeRow("Parameters (" + CountText(size.trained_elements) +
                                (size.state_elements ? " + " + CountText(size.state_elements) + " kept)" : ")"),
                            size.parameter_bytes);
                if (settings.capture_gradients) SizeRow("Gradients", size.gradient_bytes);
                SizeRow("One snapshot", size.snapshot_bytes, true);
                const TrainingConfiguration* config = live_compile_->Config();
                if (settings.capture_when == core::LayerCaptureWhen::Training && config) {
                    const auto epochs = layer_capture::SnapshotEpochs(settings.capture_every_epochs, config->epochs);
                    SizeRow(CountText(epochs.size()) + (epochs.size() == 1 ? " snapshot" : " snapshots") + " in a " +
                                std::to_string(config->epochs) + "-epoch run",
                            size.snapshot_bytes * epochs.size());
                }
                ImGui::EndTable();
            }
            if (size.over_budget) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(DebuggerWarning(), "Over the %s a snapshot holds: the layers past it are skipped "
                                                      "and counted.",
                                   data_input_view::FormatSize(LayerCaptureOptions{}.byte_budget).c_str());
                ImGui::PopTextWrapPos();
            }
        }
        WrappedNote("Kept on the compute device while a run is open; a view copies only what it shows. Saved with "
                    "the run in the debug run store.");
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
    const core::StudioDebuggerConfig defaults = core::EngineConfig::Instance().GetStudioDebugger();
    const bool differs = !(settings == defaults);
    if (ui::SecondaryButton("Save as my default", differs, "These are your defaults")) {
        auto& engine_config = core::EngineConfig::Instance();
        engine_config.SetStudioDebugger(settings);
        engine_config.Save();
        capture_session_.reset();
    }
    if (differs) {
        ImGui::SameLine();
        if (ui::LinkButton("Use my defaults")) capture_session_.reset();
    }
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
    ImGui::TextColored(DebuggerMuted(), "%s",
                       differs ? "This session's runs and layer views use these settings and the feature-map colour; your "
                                 "defaults stay as they are."
                               : "Your defaults, from Preferences \xE2\x80\xBA Studio Debugger, with the feature-map "
                                 "colour.");
    if (run_in_progress_)
        ImGui::TextColored(DebuggerMuted(), "The run in progress keeps the settings it started with.");
    ImGui::PopTextWrapPos();
    ImGui::EndPopup();
}

}  // namespace cyxwiz
