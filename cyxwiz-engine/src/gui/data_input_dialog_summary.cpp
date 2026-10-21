// DataInputDialog data strip: what the node loads and how far it got, above
// the tabs (TOFIX129 B-2; words from core/data_input_presentation).

#include "node_config_dialog.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/data_input_presentation.h"

#include <chrono>
#include <string>

namespace gui {

void DataInputDialog::RenderDataStrip() {
    namespace view = cyxwiz::data_input_view;
    const auto& t = cyxwiz::ui::CurrentTokens();

    view::StripInput in;
    in.source = CurrentSourcePath();
    const bool table = file_category_ == FileCategory::Tabular || file_category_ == FileCategory::TimeSeries;
    in.kind = table && in.source == file_path_ ? std::string(GetFileTypeName()) + " table" : CurrentSourceLabel();
    in.file_size = file_size_;
    in.loading = is_loading_async_;
    if (in.loading) {
        in.loading_message = apply_status_message_;
        while (!in.loading_message.empty() && in.loading_message.back() == '.') in.loading_message.pop_back();
        const auto running = std::chrono::steady_clock::now() - apply_started_at_;
        in.loading_seconds = std::chrono::duration<double>(running).count();
    }
    in.loaded = data_load_state_ == DataLoadState::InMemory && !loaded_dataset_name_.empty();
    in.dataset_name = loaded_dataset_name_;
    in.rows = loaded_rows_;
    in.cols = loaded_cols_;
    in.classes = loaded_classes_;
    in.backend = static_cast<view::Backend>(loaded_backend_);
    if (file_category_ == FileCategory::TimeSeries && node_) {
        const auto time = node_->parameters.find("time_column");
        if (time != node_->parameters.end()) in.time_column = time->second;
    }
    in.bytes = loaded_memory_bytes_;
    in.bytes_estimate = loaded_memory_is_estimate_;
    in.load_seconds = last_load_elapsed_ms_ >= 0.0f ? last_load_elapsed_ms_ / 1000.0 : -1.0;
    if (node_) {
        const auto count = [this](const char* key) {
            const auto it = node_->parameters.find(key);
            try { return it == node_->parameters.end() ? -1 : std::stoi(it->second); } catch (...) { return -1; }
        };
        in.audit_errors = count("audit_errors");
        in.audit_warnings = count("audit_warnings");
    }
    if (!in.loading && !apply_success_ && !apply_status_message_.empty()) {
        in.failed = true;
        in.failure = apply_status_message_;
    }
    const view::Strip strip = view::BuildStrip(in);

    const ImVec4 tone = strip.tone == view::LoadTone::Loaded    ? t.success
                        : strip.tone == view::LoadTone::Loading ? t.accent_text
                        : strip.tone == view::LoadTone::Failed  ? t.error
                                                                : t.warning;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(strip.title.c_str());
    ImGui::SameLine(0.0f, t.space_lg);
    ImGui::TextColored(tone, "%s", strip.state.c_str());
    if (!strip.detail.empty()) {
        ImGui::SameLine(0.0f, t.space_md);
        ImGui::TextColored(t.text_dim, "%s", strip.detail.c_str());
    }
    if (!strip.facts.empty()) {
        cyxwiz::ui::FlowRow row;
        for (const auto& fact : strip.facts) {
            const std::string text = fact.value + " " + fact.label;
            row.Next(ImGui::CalcTextSize(text.c_str()).x);
            ImGui::TextUnformatted(fact.value.c_str());
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::TextColored(t.text_dim, "%s", fact.label.c_str());
        }
    }
    if (!strip.note.empty()) ImGui::TextColored(t.text_faint, "%s", strip.note.c_str());
    ImGui::PopTextWrapPos();
}

} // namespace gui
