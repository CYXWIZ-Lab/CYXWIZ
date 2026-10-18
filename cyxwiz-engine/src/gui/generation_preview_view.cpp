#include "generation_preview_view.h"

#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace cyxwiz {
namespace {

std::string OneDecimal(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f", value);
    return buffer;
}

std::string FourDecimals(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4f", value);
    return buffer;
}

std::string ShortPoint(const GenerationPreviewReport& report) {
    if (report.point == "baseline") return "Before";
    if (report.point == "step") return "Step " + std::to_string(report.step);
    return std::to_string(report.epoch);
}

std::string FirstLine(const std::string& text) { return text.substr(0, text.find('\n')); }

std::string FitText(const std::string& text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    std::string cut = text;
    while (!cut.empty() && ImGui::CalcTextSize((cut + "...").c_str()).x > width) {
        cut.pop_back();
        while (!cut.empty() && (static_cast<unsigned char>(cut.back()) & 0xC0) == 0x80) cut.pop_back();
    }
    return cut + "...";
}

std::string Shown(const std::string& text) {
    std::string shown;
    for (char c : text) shown += c == '\n' ? std::string("\\n") : std::string(1, c);
    return shown;
}

ImVec4 StopColour(const GenerationPreviewSample& sample) {
    const auto& t = ui::CurrentTokens();
    if (sample.stop_reason == "eos" || sample.stop_reason == "stop_text") return t.success;
    if (sample.stop_reason == "max_tokens") return t.text_dim;
    return t.warning;
}

// A muted label over its figure, as the dashboard's tiles.
void Tile(const char* label, const std::string& value) {
    const auto& t = ui::CurrentTokens();
    ui::BeginCard(label);
    ImGui::TextColored(t.text_dim, "%s", label);
    ImGui::TextColored(t.text_bright, "%s", value.c_str());
    ui::EndCard();
}

}  // namespace

void GenerationPreviewView::SetFolder(const std::filesystem::path& folder) {
    if (folder == folder_) return;
    folder_ = folder;
    reports_.clear();
    skipped_.clear();
    stamp_.clear();
    shown_ = left_ = right_ = -1;
    prompt_ = 0;
    open_prompts_.clear();
    export_note_.clear();
    Refresh(true);
}

void GenerationPreviewView::Refresh(bool force) {
    const double now = ImGui::GetCurrentContext() ? ImGui::GetTime() : 0.0;
    if (!force && now - polled_ < 2.0) return;
    polled_ = now;
    if (folder_.empty()) return;
    const auto stamp = GenerationPreviewFolderStamp(folder_);
    if (!force && stamp == stamp_) return;
    stamp_ = stamp;
    skipped_.clear();
    reports_ = LoadGenerationPreviewReports(folder_, &skipped_);
}

void GenerationPreviewView::Render() {
    const auto& t = ui::CurrentTokens();
    Refresh(false);
    if (folder_.empty()) return;
    if (reports_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "No generation previews yet. They appear here as the run writes them.");
        for (const auto& note : skipped_) ImGui::TextColored(t.text_faint, "Not shown: %s", note.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    shown_ = std::clamp(shown_, -1, static_cast<int>(reports_.size()) - 1);
    const auto& report = reports_[static_cast<size_t>(shown_ < 0 ? reports_.size() - 1 : shown_)];

    const bool wide = ImGui::GetContentRegionAvail().x >= 1000.0f;
    if (wide && ImGui::BeginTable("##gen_view", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("main", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("rail", ImGuiTableColumnFlags_WidthFixed, 300.0f);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }
    RenderToolbar();
    if (view_ == 0) {
        RenderSummary(report);
        RenderLatest(report);
    } else if (view_ == 1) {
        RenderCompare();
    } else {
        RenderHistory();
    }
    if (wide) {
        ImGui::TableSetColumnIndex(1);
        RenderRail(report);
        ImGui::EndTable();
    } else {
        ImGui::Spacing();
        RenderRail(report);
    }
}

void GenerationPreviewView::RenderToolbar() {
    const auto& t = ui::CurrentTokens();
    const char* views[] = {"Latest", "Compare two points", "One prompt over time"};
    ui::SegmentedControl("##gen_views", views, 3, &view_);
    if (view_ != 0) return;
    // Latest: which pass to show.
    ImGui::SameLine(0.0f, 16.0f);
    float chips = 0.0f;
    for (const auto& r : reports_) chips += ui::FilterChipWidth(ShortPoint(r).c_str(), "") + 6.0f;
    if (ImGui::GetContentRegionAvail().x < chips + 60.0f) ImGui::NewLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Show");
    const int newest = static_cast<int>(reports_.size()) - 1;
    for (int i = 0; i < static_cast<int>(reports_.size()); ++i) {
        const std::string label = ShortPoint(reports_[static_cast<size_t>(i)]);
        ImGui::SameLine(0.0f, 6.0f);
        if (ImGui::GetContentRegionAvail().x < ui::FilterChipWidth(label.c_str(), "")) ImGui::NewLine();
        ImGui::PushID(i);
        const bool chosen = (shown_ < 0 ? newest : shown_) == i;
        if (ui::FilterChip("##point", label.c_str(), "", chosen)) shown_ = i == newest ? -1 : i;
        ui::Tooltip(GenerationPreviewPointLabel(reports_[static_cast<size_t>(i)]).c_str());
        ImGui::PopID();
    }
}

void GenerationPreviewView::RenderSummary(const GenerationPreviewReport& report) {
    const auto summary = SummarizeGenerationPreview(report);
    const auto of = [&](int n) { return std::to_string(n) + " of " + std::to_string(summary.samples); };
    ImGui::Spacing();
    if (ImGui::BeginTable("##gen_summary", 5, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        Tile(GenerationPreviewPointLabel(report).c_str(), std::to_string(summary.samples) + " samples");
        ImGui::TableSetColumnIndex(1);
        Tile("Checks passed",
             summary.checks == 0 ? std::string("no checks set")
                                 : std::to_string(summary.checks_passed) + " of " + std::to_string(summary.checks));
        ImGui::TableSetColumnIndex(2);
        Tile("Stopped by itself", of(summary.stopped_by_itself));
        ImGui::TableSetColumnIndex(3);
        Tile("Repeating itself", of(summary.repeating));
        ImGui::TableSetColumnIndex(4);
        Tile("Validation loss", report.validation_loss ? FourDecimals(*report.validation_loss) : std::string("not run"));
        ImGui::EndTable();
    }
}

void GenerationPreviewView::RenderOutput(const GenerationPreviewSample& sample, const GenerationPreviewReport& report,
                                         float height) {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##output", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PushTextWrapPos(0.0f);
    if (sample.text.empty()) ImGui::TextColored(t.text_faint, "(no text)");
    else ImGui::TextUnformatted(sample.text.c_str());
    const std::string end = sample.stop_reason == "eos"         ? "[end of sequence]"
                            : sample.stop_reason == "stop_text" ? "[stopped at \"" + Shown(sample.stop_text) + "\"]"
                            : sample.stop_reason == "max_tokens"
                                ? "[cut at " + std::to_string(report.max_new_tokens) + " tokens]"
                                : "[" + sample.stop_reason + "]";
    ImGui::TextColored(StopColour(sample), "%s", end.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void GenerationPreviewView::RenderLatest(const GenerationPreviewReport& report) {
    const auto& t = ui::CurrentTokens();
    const int columns = ImGui::GetContentRegionAvail().x >= 820.0f ? 2 : 1;
    if (!ImGui::BeginTable("##gen_cards", columns, ImGuiTableFlags_SizingStretchSame)) return;
    for (size_t i = 0; i < report.samples.size(); ++i) {
        const auto& sample = report.samples[i];
        if (i % static_cast<size_t>(columns) == 0) ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(static_cast<int>(i % static_cast<size_t>(columns)));
        const std::string id = std::to_string(sample.prompt_index) + "." + std::to_string(sample.sample_index);
        ImGui::PushID(id.c_str());
        ui::BeginCard("##sample");
        std::string number = std::to_string(sample.prompt_index + 1);
        if (sample.sample_index > 0 || (i + 1 < report.samples.size() &&
                                        report.samples[i + 1].prompt_index == sample.prompt_index))
            number += "." + std::to_string(sample.sample_index + 1);
        const std::string check = GenerationPreviewCheckLabel(sample);
        const float check_width = check.empty() ? 0.0f : ui::ChipWidth(check.c_str()) + 8.0f;
        ImGui::TextColored(t.text_faint, "%s", number.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_bright, "%s",
                           FitText(FirstLine(sample.prompt), ImGui::GetContentRegionAvail().x - check_width).c_str());
        if (!check.empty()) {
            ui::SameLineRight(check_width - 8.0f);
            ui::Chip(check.c_str(), sample.check_result == GenerationPreviewCheckResult::Passed ? t.success : t.caution);
        }
        RenderOutput(sample, report, ImGui::GetTextLineHeightWithSpacing() * 6.0f);
        ui::Chip(GenerationPreviewStopLabel(sample, report.max_new_tokens).c_str(), StopColour(sample));
        if (!sample.repeated_phrase.empty()) {
            const std::string repeats = "Repeats \"" + sample.repeated_phrase + "\"";
            ImGui::SameLine();
            if (ImGui::GetContentRegionAvail().x < ui::ChipWidth(repeats.c_str())) ImGui::NewLine();
            ui::Chip(repeats.c_str(), t.warning);
        }
        const bool open = open_prompts_.count(id) > 0;
        if (ui::LinkButton(open ? "Hide what the model saw" : "What the model saw")) {
            if (open) open_prompts_.erase(id);
            else open_prompts_.insert(id);
        }
        if (open) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", sample.rendered_prompt.c_str());
            ImGui::TextColored(t.text_faint, "%d prompt tokens, %d new tokens", sample.prompt_tokens, sample.new_tokens);
            ImGui::PopTextWrapPos();
        }
        ui::EndCard();
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void GenerationPreviewView::RenderCompare() {
    const auto& t = ui::CurrentTokens();
    const int n = static_cast<int>(reports_.size());
    if (n < 2) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "Comparing needs two passes; this run has one so far.");
        return;
    }
    int left = left_ < 0 ? n - 2 : std::min(left_, n - 1);
    int right = right_ < 0 ? n - 1 : std::min(right_, n - 1);
    const auto pick = [&](const char* id, int& index, int& stored) {
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::BeginCombo(id, GenerationPreviewPointLabel(reports_[static_cast<size_t>(index)]).c_str())) {
            for (int i = 0; i < n; ++i)
                if (ImGui::Selectable(GenerationPreviewPointLabel(reports_[static_cast<size_t>(i)]).c_str(), i == index))
                    stored = index = i;
            ImGui::EndCombo();
        }
    };
    ImGui::Spacing();
    pick("##left", left, left_);
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "against");
    ImGui::SameLine();
    pick("##right", right, right_);
    const auto& a = reports_[static_cast<size_t>(left)];
    const auto& b = reports_[static_cast<size_t>(right)];
    if (!ImGui::BeginTable("##compare", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) return;
    ImGui::TableSetupColumn("Prompt", ImGuiTableColumnFlags_WidthStretch, 0.6f);
    ImGui::TableSetupColumn(GenerationPreviewPointLabel(a).c_str(), ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn(GenerationPreviewPointLabel(b).c_str(), ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableHeadersRow();
    const auto find = [](const GenerationPreviewReport& r, const GenerationPreviewSample& s) -> const GenerationPreviewSample* {
        for (const auto& x : r.samples)
            if (x.prompt_index == s.prompt_index && x.sample_index == s.sample_index) return &x;
        return nullptr;
    };
    for (const auto& s : b.samples) {
        ImGui::PushID(s.prompt_index * 16 + s.sample_index);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_bright, "%d  %s", s.prompt_index + 1, FirstLine(s.prompt).c_str());
        const std::string check = GenerationPreviewCheckLabel(s);
        if (!check.empty()) ImGui::TextColored(t.text_dim, "check: %s", GenerationPreviewCheckKindName(s.check.kind));
        ImGui::PopTextWrapPos();
        const float height = ImGui::GetTextLineHeightWithSpacing() * 6.0f;
        ImGui::TableSetColumnIndex(1);
        ImGui::PushID(0);
        if (const auto* old = find(a, s)) RenderOutput(*old, a, height);
        else ImGui::TextColored(t.text_faint, "not in this pass");
        ImGui::PopID();
        ImGui::TableSetColumnIndex(2);
        ImGui::PushID(1);
        RenderOutput(s, b, height);
        ImGui::PopID();
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void GenerationPreviewView::RenderHistory() {
    const auto& t = ui::CurrentTokens();
    const auto& newest = reports_.back();
    int prompts = 0;
    for (const auto& s : newest.samples) prompts = std::max(prompts, s.prompt_index + 1);
    prompt_ = std::clamp(prompt_, 0, std::max(0, prompts - 1));
    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Prompt");
    ImGui::SameLine();
    std::string current;
    for (const auto& s : newest.samples)
        if (s.prompt_index == prompt_) current = std::to_string(prompt_ + 1) + "  " + FirstLine(s.prompt);
    ImGui::SetNextItemWidth(std::min(520.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::BeginCombo("##history_prompt", FitText(current, 480.0f).c_str())) {
        for (const auto& s : newest.samples) {
            if (s.sample_index != 0) continue;
            const std::string label = std::to_string(s.prompt_index + 1) + "  " + FirstLine(s.prompt);
            if (ImGui::Selectable(FitText(label, 500.0f).c_str(), s.prompt_index == prompt_)) prompt_ = s.prompt_index;
        }
        ImGui::EndCombo();
    }
    if (!ImGui::BeginTable("##history", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) return;
    ImGui::TableSetupColumn("When", ImGuiTableColumnFlags_WidthFixed, 150.0f);
    ImGui::TableSetupColumn("Output", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Result", ImGuiTableColumnFlags_WidthFixed, 200.0f);
    ImGui::TableHeadersRow();
    for (size_t r = 0; r < reports_.size(); ++r) {
        const auto& report = reports_[r];
        for (const auto& s : report.samples) {
            if (s.prompt_index != prompt_) continue;
            ImGui::PushID(static_cast<int>(r * 16 + static_cast<size_t>(s.sample_index)));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(t.text_bright, "%s", GenerationPreviewPointLabel(report).c_str());
            if (report.validation_loss)
                ImGui::TextColored(t.text_dim, "val loss %s", FourDecimals(*report.validation_loss).c_str());
            ImGui::TableSetColumnIndex(1);
            RenderOutput(s, report, ImGui::GetTextLineHeightWithSpacing() * 4.0f);
            ImGui::TableSetColumnIndex(2);
            const std::string check = GenerationPreviewCheckLabel(s);
            if (!check.empty())
                ImGui::TextColored(s.check_result == GenerationPreviewCheckResult::Passed ? t.success : t.caution, "%s",
                                   check.c_str());
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", GenerationPreviewStopLabel(s, report.max_new_tokens).c_str());
            if (!s.repeated_phrase.empty()) ImGui::TextColored(t.warning, "repeats \"%s\"", s.repeated_phrase.c_str());
            ImGui::PopTextWrapPos();
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
}

void GenerationPreviewView::RenderRail(const GenerationPreviewReport& report) {
    const auto& t = ui::CurrentTokens();
    ui::BeginCard("##gen_how");
    ImGui::TextColored(t.text_bright, "How these were made");
    std::string stops;
    for (const auto& stop : report.stop_texts) stops += (stops.empty() ? "" : " · ") + Shown(stop);
    std::string decoding = report.decoding == "sampling"
                               ? "sampling, temperature " + OneDecimal(report.temperature) +
                                     (report.top_k > 0 ? ", top-k " + std::to_string(report.top_k) : "") +
                                     (report.top_p < 1.0f ? ", top-p " + OneDecimal(report.top_p) : "") +
                                     ", seed " + std::to_string(report.seed)
                               : "greedy";
    if (report.repetition_penalty > 1.0f) decoding += ", repetition penalty " + OneDecimal(report.repetition_penalty);
    std::string mode = report.mode == "chat" ? "chat / instruction" : "continuation";
    for (auto value : {GenerationPreviewTemplate::RoleLines, GenerationPreviewTemplate::ChatML,
                       GenerationPreviewTemplate::Llama3})
        if (report.mode == "chat" && report.chat_template == GenerationPreviewTemplateName(value))
            mode += std::string(" (") + GenerationPreviewTemplateLabel(value) + ")";
    ui::FactsTable("##how", {{"Mode", mode},
                             {"Decoding", decoding},
                             {"Max new tokens", std::to_string(report.max_new_tokens)},
                             {"Stop texts", stops.empty() ? std::string("end of sequence only") : "EOS · " + stops},
                             {"Weights", report.point == "try"
                                             ? "checkpoint " + report.weights + " (epoch " + std::to_string(report.epoch) + ")"
                                             : report.weights == "best" ? "best so far" : "current"},
                             {"Context", report.context > 0 ? std::to_string(report.context) + " tokens" : ""},
                             {"Time", OneDecimal(report.elapsed_ms / 1000.0) + " s"}});
    ui::EndCard();
    ui::BeginCard("##gen_details");
    ImGui::TextColored(t.text_bright, "Details");
    ui::FactsTable("##details", {{"Run", report.run_id},
                                 {"Passes", std::to_string(reports_.size())},
                                 {"File", report.file.filename().string()},
                                 {"Folder", folder_.string()}});
    for (const auto& note : skipped_) ImGui::TextColored(t.text_faint, "Not shown: %s", note.c_str());
    ui::EndCard();
    if (ui::SecondaryButton("Export samples (JSONL)")) {
        const auto path = folder_ / "samples.jsonl";
        std::string error;
        export_failed_ = !ExportGenerationPreviewSamples(reports_, path, error);
        export_note_ = export_failed_ ? "Export failed: " + error : "Saved " + path.string();
    }
    if (!export_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(export_failed_ ? t.error : t.success, "%s", export_note_.c_str());
        ImGui::PopTextWrapPos();
    }
}

}  // namespace cyxwiz
