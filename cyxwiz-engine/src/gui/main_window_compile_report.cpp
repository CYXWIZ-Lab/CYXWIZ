// The compile report, the memory-risk question before training and the
// "operation failed" notice (TOFIX129 B3). Thin renderers over
// core/compile_report_presentation: the words and their order come from the
// model; this file only draws them and wires the actions.

#include "main_window.h"

#include "icons.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_fonts.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/live_graph_compile.h"
#include "data_studio/data_studio_panel.h"
#include "panels/training_plot_panel.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfloat>
#include <filesystem>

namespace gui {

namespace {

namespace report = cyxwiz::compile_report;

ImVec4 ToneColour(report::Tone tone) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    switch (tone) {
        case report::Tone::Ok: return t.success;
        case report::Tone::Warning: return t.warning;
        case report::Tone::Error: return t.error;
        case report::Tone::Info: return t.info;
        case report::Tone::Neutral: return t.text_dim;
    }
    return t.text_dim;
}

const char* ToneIcon(report::Tone tone) {
    switch (tone) {
        case report::Tone::Ok: return ICON_FA_CIRCLE_CHECK;
        case report::Tone::Warning: return ICON_FA_TRIANGLE_EXCLAMATION;
        case report::Tone::Error: return ICON_FA_CIRCLE_XMARK;
        default: return ICON_FA_CIRCLE_INFO;
    }
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// A row of chips that wraps in a narrow window.
void ChipRow(const std::vector<report::Chip>& chips) {
    cyxwiz::ui::FlowRow row;
    for (const auto& chip : chips) {
        row.Next(cyxwiz::ui::ChipWidth(chip.text.c_str()));
        cyxwiz::ui::Chip(chip.text.c_str(), ToneColour(chip.tone));
    }
}

// The window's first size: what the content needs, at most 90% of the screen.
void NextWindowFits(float width, float height) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2((std::min)(width, viewport->WorkSize.x * 0.9f),
                                    (std::min)(height, viewport->WorkSize.y * 0.9f)),
                             ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320.0f, 200.0f), ImVec2(FLT_MAX, FLT_MAX));
}

bool EscapePressedHere() {
    return ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
           ImGui::IsKeyPressed(ImGuiKey_Escape, false);
}

float FooterHeight() {
    return cyxwiz::ui::ButtonHeight(cyxwiz::ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.y * 2.0f + 4.0f;
}

}  // namespace

void MainWindow::ShowCompileReport(cyxwiz::compile_report::Mode mode, std::string stop_reason) {
    report::Input in;
    in.mode = mode;
    if (node_editor_ && !node_editor_->GetCurrentFilePath().empty())
        in.graph_name = std::filesystem::path(node_editor_->GetCurrentFilePath()).stem().string();
    in.stop_reason = std::move(stop_reason);
    in.compiled_ok = compile_result_success_;
    in.issues = compile_result_issues_;
    in.placements = compile_result_backend_placements_;
    in.has_config = compile_result_has_config_;
    in.config = compile_result_config_;
    // Parameters: the layers' counts of the live compile (the AS COMPILED
    // cards' numbers), when it compiled the same layers.
    if (in.has_config && live_compile_ && live_compile_->ParametersCounted() && live_compile_->Config() &&
        live_compile_->Config()->layers.size() == in.config.layers) {
        long long total = 0;
        for (const auto& [layer, count] : live_compile_->LayerParameters()) total += count;
        in.config.parameters = total;
    }
    compile_report_ = report::BuildReport(in);
    show_compile_result_popup_ = true;
}

void MainWindow::RenderCompileResultPopup() {
    static constexpr const char* kId = "###compile_report";
    if (show_compile_result_popup_) {
        ImGui::OpenPopup(kId);
        show_compile_result_popup_ = false;
    }
    const auto& r = compile_report_;
    const std::string window = r.title + kId;
    NextWindowFits(860.0f, 640.0f);
    int focus_node = -1, open_node = -1, add_tokenizer_after = -1, add_after_input = -1;
    gui::NodeType add_type = gui::NodeType::Resize;
    bool open_data_studio = false, train = false;
    if (ImGui::BeginPopupModal(window.c_str(), nullptr)) {
        const auto& t = cyxwiz::ui::CurrentTokens();
        // Verdict, subtitle, counts.
        {
            cyxwiz::ui::FontScope medium(cyxwiz::ui::Font::Medium);
            ImGui::TextColored(ToneColour(r.verdict_tone), "%s", ToneIcon(r.verdict_tone));
            ImGui::SameLine();
            Wrapped(ToneColour(r.verdict_tone), r.verdict);
        }
        if (!r.subtitle.empty()) Wrapped(t.text_dim, r.subtitle);
        ChipRow(r.chips);
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));

        const float body_h = -FooterHeight();
        if (ImGui::BeginTabBar("##compile_report_tabs")) {
            const std::string findings_tab = "Findings (" + std::to_string(r.findings.size()) + ")###findings";
            if (ImGui::BeginTabItem(findings_tab.c_str())) {
                ImGui::BeginChild("##findings", ImVec2(0.0f, body_h));
                if (r.findings.empty()) ImGui::TextColored(t.text_dim, "No findings.");
                int last_level = -1;
                for (size_t i = 0; i < r.findings.size(); ++i) {
                    const auto& f = r.findings[i];
                    const int level = static_cast<int>(f.level);
                    if (level != last_level) {
                        last_level = level;
                        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
                        ImGui::TextColored(ToneColour(f.tone), "%s",
                                           f.level == cyxwiz::IssueLevel::Error     ? "Errors"
                                           : f.level == cyxwiz::IssueLevel::Warning ? "Warnings"
                                                                                    : "Notes");
                    }
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::TextColored(ToneColour(f.tone), "%s", ToneIcon(f.tone));
                    ImGui::SameLine();
                    ImGui::BeginGroup();
                    Wrapped(t.text, f.message);
                    cyxwiz::ui::FlowRow meta;
                    if (!f.node_name.empty()) {
                        meta.Next(ImGui::CalcTextSize(f.node_name.c_str()).x);
                        ImGui::TextColored(t.text_dim, "%s", f.node_name.c_str());
                    }
                    if (f.node_id >= 0 && node_editor_) {
                        meta.Next(ImGui::CalcTextSize("Show on canvas").x);
                        if (cyxwiz::ui::LinkButton("Show on canvas")) focus_node = f.node_id;
                    }
                    if (f.open_dialog && node_editor_) {
                        meta.Next(ImGui::CalcTextSize("Open its dialog").x);
                        if (cyxwiz::ui::LinkButton("Open its dialog")) open_node = f.node_id;
                    }
                    if (f.add_node == cyxwiz::compile_report::AddNodeFix::TextTokenizer && node_editor_) {
                        meta.Next(ImGui::CalcTextSize("Add a Text Tokenizer").x);
                        if (cyxwiz::ui::LinkButton("Add a Text Tokenizer")) add_tokenizer_after = f.node_id;
                    }
                    // Resize and audio features go where the data enters the model.
                    const auto add_link = [&](const char* label, gui::NodeType type) {
                        meta.Next(ImGui::CalcTextSize(label).x);
                        if (cyxwiz::ui::LinkButton(label)) {
                            add_after_input = f.node_id;
                            add_type = type;
                        }
                    };
                    if (f.add_node == cyxwiz::compile_report::AddNodeFix::Resize && node_editor_)
                        add_link("Add a Resize", gui::NodeType::Resize);
                    if (f.add_node == cyxwiz::compile_report::AddNodeFix::AudioFeatures && node_editor_) {
                        add_link("Add Mel Spectrogram", gui::NodeType::MelSpectrogram);
                        add_link("Add Spectrogram", gui::NodeType::Spectrogram);
                        add_link("Add MFCC", gui::NodeType::MFCC);
                    }
                    if (f.open_data_studio && data_studio_panel_) {
                        meta.Next(ImGui::CalcTextSize("Open Data Studio").x);
                        if (cyxwiz::ui::LinkButton("Open Data Studio")) open_data_studio = true;
                    }
                    if (!f.code.empty()) {
                        meta.Next(ImGui::CalcTextSize(f.code.c_str()).x);
                        ImGui::TextColored(t.text_faint, "%s", f.code.c_str());
                    }
                    ImGui::EndGroup();
                    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
                    ImGui::PopID();
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Where each layer runs###placement")) {
                ImGui::BeginChild("##placement", ImVec2(0.0f, body_h));
                if (r.placements.empty()) {
                    ImGui::TextColored(t.text_dim, "No layer placement was reported.");
                } else {
                    Wrapped(t.text, r.placement_sentence);
                    ChipRow(r.placement_chips);
                    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
                    if (ImGui::BeginTable("##placement_rows", 3,
                                          ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthStretch, 1.0f);
                        ImGui::TableSetupColumn("Runs on", ImGuiTableColumnFlags_WidthStretch, 0.8f);
                        ImGui::TableSetupColumn("Why", ImGuiTableColumnFlags_WidthStretch, 1.6f);
                        ImGui::TableHeadersRow();
                        for (size_t i = 0; i < r.placements.size(); ++i) {
                            const auto& p = r.placements[i];
                            ImGui::PushID(static_cast<int>(i));
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            Wrapped(t.text, p.layer);
                            ImGui::TableNextColumn();
                            ImGui::TextColored(ToneColour(p.tone), "%s", p.runs_on.c_str());
                            if (!p.engine.empty()) Wrapped(t.text_dim, p.engine);
                            ImGui::TableNextColumn();
                            Wrapped(t.text, p.why);
                            if (!p.details.empty() && ImGui::TreeNodeEx("Details", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                                for (const auto& [key, value] : p.details)
                                    Wrapped(t.text_dim, key + ": " + value);
                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Configuration###configuration")) {
                ImGui::BeginChild("##configuration", ImVec2(0.0f, body_h));
                if (r.config.empty()) {
                    ImGui::TextColored(t.text_dim, "The compile stopped before it produced a configuration.");
                } else {
                    cyxwiz::ui::FlowRow tiles;
                    for (const auto& tile : r.tiles) {
                        float w = 0.0f;
                        {
                            cyxwiz::ui::FontScope medium(cyxwiz::ui::Font::Medium);
                            w = ImGui::CalcTextSize(tile.value.c_str()).x;
                        }
                        w = (std::max)(w, ImGui::CalcTextSize(tile.label.c_str()).x) + t.space_lg;
                        tiles.Next(w);
                        ImGui::BeginGroup();
                        {
                            cyxwiz::ui::FontScope medium(cyxwiz::ui::Font::Medium);
                            ImGui::TextUnformatted(tile.value.c_str());
                        }
                        ImGui::TextColored(t.text_dim, "%s", tile.label.c_str());
                        ImGui::EndGroup();
                        ImGui::SameLine(0.0f, 0.0f);
                        ImGui::Dummy(ImVec2(t.space_lg, 0.0f));
                    }
                    ImGui::NewLine();
                    for (size_t g = 0; g < r.config.size(); ++g) {
                        const auto& group = r.config[g];
                        ImGui::Dummy(ImVec2(0.0f, t.space_sm));
                        ImGui::TextColored(t.text, "%s", group.title.c_str());
                        const std::string id = "##config_" + std::to_string(g);
                        if (ImGui::BeginTable(id.c_str(), 2, ImGuiTableFlags_SizingStretchProp)) {
                            ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.35f);
                            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
                            for (const auto& [key, value] : group.rows) {
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                Wrapped(t.text_dim, key);
                                ImGui::TableNextColumn();
                                Wrapped(t.text, value);
                            }
                            ImGui::EndTable();
                        }
                    }
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        // Footer: Copy report at the left; Close (and Train when ready) at the right.
        ImGui::Dummy(ImVec2(0.0f, 2.0f));
        if (cyxwiz::ui::SecondaryButton("Copy report", true, nullptr, cyxwiz::ui::ButtonSize::Small))
            ImGui::SetClipboardText(r.copy_text.c_str());
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        float right = cyxwiz::ui::ButtonWidth("Close", cyxwiz::ui::ButtonSize::Regular);
        if (r.offer_train) right += spacing + cyxwiz::ui::ButtonWidth("Train", cyxwiz::ui::ButtonSize::Regular);
        cyxwiz::ui::SameLineRight(right);
        bool close = false;
        if (r.offer_train) {
            close = cyxwiz::ui::SecondaryButton("Close", true, nullptr, cyxwiz::ui::ButtonSize::Regular);
            ImGui::SameLine();
            if (cyxwiz::ui::PrimaryButton("Train")) train = true;
            ImGui::SetItemDefaultFocus();
        } else {
            close = cyxwiz::ui::PrimaryButton("Close");
            ImGui::SetItemDefaultFocus();
        }
        if (close || train || EscapePressedHere() || focus_node >= 0 || open_node >= 0 || open_data_studio ||
            add_tokenizer_after >= 0 || add_after_input >= 0)
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // Actions run after the modal closes.
    if (focus_node >= 0 && node_editor_) node_editor_->FocusNode(focus_node);
    if (open_node >= 0 && node_editor_) node_editor_->OpenNodeConfiguration(open_node);
    if (add_tokenizer_after >= 0 && node_editor_) node_editor_->AddTextTokenizerAfter(add_tokenizer_after);
    if (add_after_input >= 0 && node_editor_) node_editor_->AddModelInputStepAfter(add_after_input, add_type);
    if (open_data_studio && data_studio_panel_) {
        data_studio_panel_->SetVisible(true);
        if (compile_result_config_.backing != report::Backing::NotLoaded && !compile_result_config_.dataset.empty())
            data_studio_panel_->ShowProfile(compile_result_config_.dataset);
    }
    if (train && node_editor_) {
        StartTrainingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
        if (training_plot_panel_) training_plot_panel_->SetVisible(true);
    }
}

void MainWindow::ShowOperationError(std::string title, std::string message) {
    spdlog::error("{}: {}", title, message);
    operation_error_title_ = std::move(title);
    operation_error_message_ = std::move(message);
    show_operation_error_popup_ = true;
}

void MainWindow::RenderOperationErrorPopup() {
    static constexpr const char* kId = "###operation_error";
    if (show_operation_error_popup_) {
        ImGui::OpenPopup(kId);
        show_operation_error_popup_ = false;
    }
    const std::string window = operation_error_title_ + kId;
    NextWindowFits(520.0f, 240.0f);
    if (ImGui::BeginPopupModal(window.c_str(), nullptr)) {
        const auto& t = cyxwiz::ui::CurrentTokens();
        {
            cyxwiz::ui::FontScope medium(cyxwiz::ui::Font::Medium);
            ImGui::TextColored(t.error, "%s", ICON_FA_CIRCLE_XMARK);
            ImGui::SameLine();
            Wrapped(t.error, operation_error_title_);
        }
        ImGui::BeginChild("##operation_error_body", ImVec2(0.0f, -FooterHeight()));
        Wrapped(t.text, operation_error_message_);
        ImGui::EndChild();
        if (cyxwiz::ui::SecondaryButton("Copy", true, nullptr, cyxwiz::ui::ButtonSize::Small))
            ImGui::SetClipboardText((operation_error_title_ + "\n" + operation_error_message_).c_str());
        cyxwiz::ui::SameLineRight(cyxwiz::ui::ButtonWidth("Close", cyxwiz::ui::ButtonSize::Regular));
        const bool close = cyxwiz::ui::PrimaryButton("Close");
        ImGui::SetItemDefaultFocus();
        if (close || EscapePressedHere()) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void MainWindow::RenderMaterializationMemoryConfirmationPopup() {
    static constexpr const char* kId = "Memory risk before training###memory_risk";
    if (show_materialization_memory_confirmation_popup_) {
        ImGui::OpenPopup(kId);
        show_materialization_memory_confirmation_popup_ = false;
    }
    report::MemoryRiskInput in;
    in.node_name = materialization_memory_confirmation_node_;
    in.node_id = materialization_memory_confirmation_node_id_;
    in.risk = materialization_memory_confirmation_risk_;
    in.estimated_bytes = materialization_memory_confirmation_estimated_bytes_;
    in.available_bytes = materialization_memory_confirmation_available_bytes_;
    in.budget_bytes = materialization_memory_confirmation_budget_bytes_;
    in.detail = materialization_memory_confirmation_detail_;

    NextWindowFits(560.0f, 400.0f);
    bool continue_training = false;
    bool cancel_training = false;
    if (ImGui::BeginPopupModal(kId, nullptr)) {
        const report::MemoryRiskView v = report::BuildMemoryRiskView(in);
        const auto& t = cyxwiz::ui::CurrentTokens();
        {
            cyxwiz::ui::FontScope medium(cyxwiz::ui::Font::Medium);
            ImGui::TextColored(ToneColour(v.tone), "%s", ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::SameLine();
            Wrapped(ToneColour(v.tone), v.verdict);
        }
        Wrapped(t.text_dim, v.subtitle);
        ImGui::BeginChild("##memory_risk_body", ImVec2(0.0f, -FooterHeight()));
        if (v.estimate_share > 0.0) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ToneColour(v.tone));
            ImGui::ProgressBar(static_cast<float>(v.estimate_share), ImVec2(-FLT_MIN, 8.0f), "");
            ImGui::PopStyleColor();
            ImGui::TextColored(t.text_dim, "The estimate is %.0f%% of the free memory.", v.estimate_share * 100.0);
        }
        if (ImGui::BeginTable("##memory_rows", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.4f);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            for (const auto& [key, value] : v.rows) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                Wrapped(t.text_dim, key);
                ImGui::TableNextColumn();
                Wrapped(t.text, value);
            }
            ImGui::EndTable();
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        Wrapped(t.text, v.advice);
        if (!v.detail.empty()) {
            ImGui::Dummy(ImVec2(0.0f, t.space_xs));
            Wrapped(t.text_dim, v.detail);
        }
        ImGui::EndChild();
        // The risky choice reads as risky; Cancel is the default.
        if (cyxwiz::ui::DangerButton("Continue anyway")) continue_training = true;
        cyxwiz::ui::SameLineRight(cyxwiz::ui::ButtonWidth("Cancel", cyxwiz::ui::ButtonSize::Regular));
        if (cyxwiz::ui::PrimaryButton("Cancel")) cancel_training = true;
        ImGui::SetItemDefaultFocus();
        if (EscapePressedHere()) cancel_training = true;
        if (continue_training || cancel_training) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if (cancel_training) {
        pending_memory_confirmation_nodes_.clear();
        pending_memory_confirmation_links_.clear();
        materialization_memory_confirmation_detail_.clear();
        materialization_memory_confirmation_risk_.clear();
        materialization_memory_confirmation_node_.clear();
        materialization_memory_confirmation_node_id_ = -1;
        materialization_memory_confirmation_estimated_bytes_ = 0;
        if (training_plot_panel_)
            training_plot_panel_->SetPreparationFailed("Training not started: cancelled at the memory-risk question.");
        spdlog::info("StartTrainingFromGraph: materialization memory confirmation cancelled by user");
    } else if (continue_training) {
        auto nodes = std::move(pending_memory_confirmation_nodes_);
        auto links = std::move(pending_memory_confirmation_links_);
        accepted_materialization_memory_node_id_ = materialization_memory_confirmation_node_id_;
        accepted_materialization_memory_estimated_bytes_ = materialization_memory_confirmation_estimated_bytes_;
        accepted_materialization_memory_risk_ = materialization_memory_confirmation_risk_;
        materialization_memory_confirmation_detail_.clear();
        materialization_memory_confirmation_risk_.clear();
        materialization_memory_confirmation_node_.clear();
        materialization_memory_confirmation_node_id_ = -1;
        materialization_memory_confirmation_estimated_bytes_ = 0;
        materialization_memory_confirmation_accepted_once_ = true;
        StartTrainingFromGraph(nodes, links);
    }
}

}  // namespace gui
