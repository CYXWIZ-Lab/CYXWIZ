#include "quality_analyzer_dialog.h"

#include "icons.h"
#include "ui_buttons.h"
#include "ui_fonts.h"
#include "ui_tokens.h"

#include "../core/async_task_manager.h"
#include "../core/graph_compiler.h"
#include "../core/image_quality_analysis.h"
#include "../core/image_utils.h"

#include <cyxwiz/device.h>

#include <glad/glad.h>
#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

// windows.h (through glad) renames ImageUtils::LoadImage.
#ifdef LoadImage
#undef LoadImage
#endif

namespace gui {

namespace {

namespace ui = cyxwiz::ui;

constexpr float kChecksWidth = 280.0f;
constexpr float kThumb = 72.0f;
constexpr float kHistogramHeight = 44.0f;

const char* const kStoredNote = "Rejected images stay on disk; only training skips them.";

// A tinted card: one step off the window, no border, as tall as its content.
bool BeginCard(const char* id) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::CurrentTokens().bg_panel);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, ui::CurrentTokens().rounding_button);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    const bool open = ImGui::BeginChild(id, ImVec2(0, 0),
                                        ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    return open;
}

void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void Dim(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void Coloured(const ImVec4& colour, const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

// A label on the left and a number field on the right of the card.
bool NumberRow(const char* label, const char* id, float* value, float width) {
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::SameLine(ImGui::GetContentRegionMax().x - width);
    ImGui::SetNextItemWidth(width);
    return ImGui::InputFloat(id, value, 0.0f, 0.0f, "%g");
}

std::string Count(size_t value) { return std::to_string(value); }

}  // namespace

QualityAnalyzerDialog::QualityAnalyzerDialog(MLNode* node)
    : NodeConfigDialog("Quality Analyzer###quality_analyzer_" + std::to_string(node ? node->id : 0), node) {
    if (node_) checks_ = cyxwiz::ImageQualityChecksFromParameters(node_->parameters);
}

QualityAnalyzerDialog::~QualityAnalyzerDialog() {
    ReleaseThumbnails();
}

bool QualityAnalyzerDialog::IsBusy() const {
    return job_ && !job_->finished.load();
}

void QualityAnalyzerDialog::Apply() {
    if (!node_) return;
    cyxwiz::WriteImageQualityChecks(checks_, node_->parameters);
    original_params_ = node_->parameters;
    has_changes_ = false;
}

void QualityAnalyzerDialog::Reset() {
    if (!node_) return;
    // The analysis made in this session stays: it exists whatever the checks.
    const auto key = node_->parameters.find("analysis_key");
    const std::string analysis_key = key != node_->parameters.end() ? key->second : std::string();
    node_->parameters = original_params_;
    if (!analysis_key.empty()) node_->parameters["analysis_key"] = analysis_key;
    checks_ = cyxwiz::ImageQualityChecksFromParameters(node_->parameters);
    has_changes_ = false;
    Rejudge();
}

// The dataset and the Resize size, exactly as training compiles them.
void QualityAnalyzerDialog::ResolveContext() {
    context_resolved_ = true;
    context_error_.clear();
    entry_.reset();
    analysis_.reset();
    stale_ = false;
    ReleaseThumbnails();
    if (!node_editor_) {
        context_error_ = "Open the Quality Analyzer from its node in the graph.";
        return;
    }
    const auto& nodes = node_editor_->GetNodes();
    const auto config = cyxwiz::GraphCompiler{}.Compile(nodes, node_editor_->GetLinks(), true);
    dataset_name_ = config.dataset_name;
    const auto* entry = dataset_name_.empty() ? nullptr
                                              : cyxwiz::DataRegistry::Instance().GetImageDatasetEntry(dataset_name_);
    if (!entry) {
        context_error_ = "Connect a Data Input with images and load them (Apply in its dialog), then check again.";
        return;
    }
    entry_ = *entry;
    title_ = "Quality Analyzer - " + dataset_name_ + "###quality_analyzer_" + std::to_string(node_->id);
    const bool has_resize = std::any_of(nodes.begin(), nodes.end(),
                                        [](const MLNode& n) { return n.type == NodeType::Resize; });
    if (!has_resize) {
        context_error_ = "Add a Resize node: the images are measured at its size, as the model sees them.";
        return;
    }
    const auto shape = cyxwiz::DecodedImageShape(config);
    width_ = static_cast<int>(shape.width);
    height_ = static_cast<int>(shape.height);
    if (const std::string why = cyxwiz::image::ValidateImageQualityShape(shape); !why.empty()) {
        context_error_ = why;
        return;
    }
    try {
        key_ = cyxwiz::ImageQualityKey(*entry_, width_, height_);
    } catch (const std::exception& e) {
        context_error_ = std::string("The images cannot be listed: ") + e.what();
        return;
    }
    analysis_ = cyxwiz::LoadImageQualityAnalysis(key_);
    if (!analysis_ && node_) {
        // The last analysis, for other images or another size: shown, marked stale.
        const auto previous = node_->parameters.find("analysis_key");
        if (previous != node_->parameters.end() && !previous->second.empty() && previous->second != key_) {
            analysis_ = cyxwiz::LoadImageQualityAnalysis(previous->second);
            stale_ = analysis_ != nullptr;
        }
    }
    Rejudge();
}

void QualityAnalyzerDialog::Rejudge() {
    view_.reset();
    checks_error_ = cyxwiz::image::ValidateImageQualityChecks(checks_);
    if (!analysis_ || !checks_error_.empty()) return;
    try {
        const auto verdict = cyxwiz::JudgeImageQualityAnalysis(*analysis_, checks_);
        view_ = cyxwiz::BuildQualityAnalyzerView(*analysis_, verdict, checks_, dataset_name_);
    } catch (const std::exception& e) {
        checks_error_ = e.what();
    }
}

void QualityAnalyzerDialog::StartAnalysis() {
    if (!entry_ || IsBusy()) return;
    job_ = std::make_shared<Job>();
    job_error_.clear();
    device_label_ = cyxwiz::ImageQualityDeviceLabel();
    const auto job = job_;
    const auto entry = *entry_;
    const int width = width_, height = height_;
    task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Quality Analyzer: " + dataset_name_,
        [job, entry, width, height](cyxwiz::LambdaTask& task) {
            try {
                // Measure where training runs: a worker starts on ArrayFire's
                // default backend, not the selected device.
                if (const auto selected = cyxwiz::Device::GetProcessDevice()) {
                    cyxwiz::Device(selected->type, selected->device_id).ActivateExact(false);
                }
                const auto analysis = cyxwiz::AnalyzeImageQuality(
                    entry, width, height,
                    [&](size_t done, size_t total) {
                        job->done = done;
                        job->total = total;
                        task.ReportProgress(static_cast<float>(done) / static_cast<float>(total),
                                            std::to_string(done) + " of " + std::to_string(total) + " images");
                        if (task.ShouldStop()) job->cancel = true;
                    },
                    &job->cancel);
                std::lock_guard<std::mutex> lock(job->mutex);
                if (!analysis) {
                    job->cancelled = true;
                    task.MarkCancelled("Stopped");
                } else {
                    std::string error;
                    if (!cyxwiz::SaveImageQualityAnalysis(*analysis, &error)) throw std::runtime_error(error);
                    job->key = analysis->key;
                }
            } catch (const std::exception& e) {
                {
                    std::lock_guard<std::mutex> lock(job->mutex);
                    job->error = e.what();
                }
                job->finished = true;
                throw;
            }
            job->finished = true;
        });
}

void QualityAnalyzerDialog::PollJob() {
    if (!job_ || !job_->finished.load()) return;
    std::string key;
    {
        std::lock_guard<std::mutex> lock(job_->mutex);
        job_error_ = job_->error;
        key = job_->cancelled ? std::string() : job_->key;
    }
    job_.reset();
    if (!key.empty() && node_) {
        node_->parameters["analysis_key"] = key;
        ResolveContext();
    }
}

void QualityAnalyzerDialog::RenderContent() {
    PollJob();
    if (!context_resolved_) ResolveContext();
    const bool analyzing = IsBusy();

    ImGui::BeginChild("##checks", ImVec2(kChecksWidth, 0), ImGuiChildFlags_None);
    RenderChecks(analyzing);
    ImGui::EndChild();
    ImGui::SameLine(0.0f, 20.0f);
    ImGui::BeginChild("##results", ImVec2(0, 0), ImGuiChildFlags_None);
    if (!context_error_.empty()) {
        ImGui::Dummy(ImVec2(0, 40.0f));
        Coloured(ui::CurrentTokens().text, context_error_);
        ImGui::Dummy(ImVec2(0, 8.0f));
        if (ui::SecondaryButton("Check again", true, nullptr, ui::ButtonSize::Regular)) ResolveContext();
    } else if (analyzing) {
        RenderAnalyzing();
    } else if (analysis_) {
        RenderResults();
    } else {
        RenderNotAnalyzed();
    }
    ImGui::EndChild();
}

void QualityAnalyzerDialog::RenderChecks(bool locked) {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted("CHECKS");
    ImGui::PopStyleColor();
    ImGui::Dummy(ImVec2(0, 2.0f));

    bool changed = false;
    ImGui::BeginDisabled(locked);
    if (BeginCard("##blur")) {
        changed |= ImGui::Checkbox("Blur", &checks_.blur);
        changed |= NumberRow("Reject below", "##blur_min", &checks_.blur_min, 64.0f);
        Dim("Variance of the Laplacian; higher is sharper.");
    }
    EndCard();
    if (BeginCard("##brightness")) {
        changed |= ImGui::Checkbox("Brightness", &checks_.brightness);
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted("Reject below / above");
        ImGui::PopStyleColor();
        const float field = 44.0f;
        ImGui::SameLine(ImGui::GetContentRegionMax().x - field * 2 - ImGui::GetStyle().ItemSpacing.x);
        ImGui::SetNextItemWidth(field);
        changed |= ImGui::InputFloat("##brightness_min", &checks_.brightness_min, 0.0f, 0.0f, "%g");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(field);
        changed |= ImGui::InputFloat("##brightness_max", &checks_.brightness_max, 0.0f, 0.0f, "%g");
        Dim("Mean luminance, 0 to 255.");
    }
    EndCard();
    if (BeginCard("##contrast")) {
        changed |= ImGui::Checkbox("Contrast", &checks_.contrast);
        changed |= NumberRow("Reject below", "##contrast_min", &checks_.contrast_min, 64.0f);
        Dim("Spread of luminance (standard deviation / 255).");
    }
    EndCard();
    if (BeginCard("##duplicates")) {
        changed |= ImGui::Checkbox("Near-duplicates", &checks_.duplicates);
        Dim("Keeps the first image of each look-alike group.");
    }
    EndCard();
    ImGui::EndDisabled();

    if (changed) {
        has_changes_ = true;
        Rejudge();
    }

    if (locked) {
        Dim("The checks are locked while the images are measured.");
    } else if (width_ > 0 && context_error_.empty()) {
        const std::string size = std::to_string(width_) + " x " + std::to_string(height_);
        Dim(("Measured at the Resize size, " + size + ", as the model sees the images.").c_str());
        if (analysis_) {
            ImGui::Dummy(ImVec2(0, 2.0f));
            if (ui::SecondaryButton("Analyze again", true, nullptr, ui::ButtonSize::Regular)) StartAnalysis();
        }
    }
}

void QualityAnalyzerDialog::RenderNotAnalyzed() {
    const auto& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;
    const float column = (std::min)(380.0f, width);
    const auto centred = [&](float item) { ImGui::SetCursorPosX((std::max)(0.0f, (width - item) * 0.5f)); };

    ImGui::Dummy(ImVec2(0, (std::max)(20.0f, ImGui::GetContentRegionAvail().y * 0.25f)));
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        centred(ImGui::CalcTextSize(ICON_FA_IMAGE).x);
        ImGui::TextUnformatted(ICON_FA_IMAGE);
        ImGui::PopStyleColor();
        centred(ImGui::CalcTextSize("Not analyzed yet").x);
        ImGui::TextUnformatted("Not analyzed yet");
    }
    const std::string count = entry_ ? Count(entry_->num_images) + " images" : "the images";
    const std::string text = "Analyze measures all " + count +
                             " once on the GPU and shows which ones the checks would leave out, with examples. "
                             "Training refuses to start until this node has a current analysis.";
    centred(column);
    ImGui::BeginGroup();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + column);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    ImGui::EndGroup();
    if (!job_error_.empty()) {
        centred(column);
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + column);
        ImGui::PushStyleColor(ImGuiCol_Text, t.caution);
        ImGui::TextUnformatted(("The analysis failed: " + job_error_).c_str());
        ImGui::PopStyleColor();
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    }
    ImGui::Dummy(ImVec2(0, 6.0f));
    centred(ui::ButtonWidth("Analyze", ui::ButtonSize::Regular));
    if (ui::PrimaryButton("Analyze")) StartAnalysis();
}

void QualityAnalyzerDialog::RenderAnalyzing() {
    const auto& t = ui::CurrentTokens();
    const size_t done = job_->done.load();
    const size_t total = job_->total.load() ? job_->total.load() : (entry_ ? entry_->num_images : 0);
    const float pad = 40.0f;
    ImGui::Dummy(ImVec2(0, (std::max)(20.0f, ImGui::GetContentRegionAvail().y * 0.3f)));
    ImGui::Indent(pad);
    const float width = ImGui::GetContentRegionAvail().x - pad;
    {
        ui::FontScope bold(ui::Font::Medium);
        Coloured(t.accent_text, std::string(ICON_FA_SPINNER) + "  Measuring images...");
    }
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_raised);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, t.accent);
    ImGui::ProgressBar(total ? static_cast<float>(done) / static_cast<float>(total) : 0.0f, ImVec2(width, 8.0f), "");
    ImGui::PopStyleColor(2);
    const std::string progress = Count(done) + " of " + Count(total) + " images";
    ImGui::TextUnformatted(progress.c_str());
    ImGui::SameLine(ImGui::GetCursorPosX() + width - ImGui::CalcTextSize(device_label_.c_str()).x - pad);
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted(device_label_.c_str());
    ImGui::PopStyleColor();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_faint);
    ImGui::TextUnformatted("Images are decoded on the CPU in batches and measured on the GPU. This runs in Task "
                           "View too, so you can close the dialog and keep working.");
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();
    ImGui::Unindent(pad);
}

void QualityAnalyzerDialog::RenderResults() {
    const auto& t = ui::CurrentTokens();
    if (stale_) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.warning, 0.12f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_button);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 8.0f));
        ImGui::BeginChild("##stale", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        Coloured(t.warning, std::string(ICON_FA_TRIANGLE_EXCLAMATION) +
                                "  The images or the Resize size changed since this analysis. Analyze again before "
                                "training.");
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
    if (!checks_error_.empty() || !view_) {
        Coloured(t.caution, checks_error_.empty() ? std::string("No result.") : checks_error_);
        return;
    }
    const auto& view = *view_;
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::TextUnformatted(view.headline.c_str());
    }
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted(view.subline.c_str());
    ImGui::PopStyleColor();
    if (!view.imbalance.empty()) {
        Coloured(t.warning, std::string(ICON_FA_TRIANGLE_EXCLAMATION) + "  " + view.imbalance);
    }
    ImGui::Dummy(ImVec2(0, 4.0f));
    for (const auto& row : view.reasons) RenderReason(row);
    if (!view.footnote.empty()) Coloured(t.text_faint, view.footnote);
    ImGui::Dummy(ImVec2(0, 6.0f));
    RenderHistograms();
}

void QualityAnalyzerDialog::RenderReason(const cyxwiz::QualityReasonRow& row) {
    const auto& t = ui::CurrentTokens();
    ImGui::PushID(row.title.c_str());
    if (BeginCard("##reason")) {
        {
            ui::FontScope bold(ui::Font::Medium);
            ImGui::TextUnformatted(row.title.c_str());
        }
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted(Count(row.count).c_str());
        ImGui::PopStyleColor();
        const float rule = ImGui::CalcTextSize(row.rule.c_str()).x;
        ImGui::SameLine((std::max)(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - rule));
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_faint);
        ImGui::TextUnformatted(row.rule.c_str());
        ImGui::PopStyleColor();

        const float step = kThumb + ImGui::GetStyle().ItemSpacing.x;
        const float right = ImGui::GetContentRegionMax().x;
        for (size_t e = 0; e < row.examples.size(); ++e) {
            const auto& example = row.examples[e];
            // Wrap to the next line instead of running past the card.
            if (e > 0) {
                if (ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + step <= right) ImGui::SameLine();
            }
            ImGui::BeginGroup();
            const unsigned int texture = Thumbnail(example.index);
            if (texture != 0) {
                ImGui::Image(static_cast<ImTextureID>(texture), ImVec2(kThumb, kThumb));
            } else {
                const ImVec2 at = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddRectFilled(at, ImVec2(at.x + kThumb, at.y + kThumb),
                                                          ui::ToU32(t.bg_raised), 4.0f);
                ImGui::Dummy(ImVec2(kThumb, kThumb));
            }
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
            ImGui::TextUnformatted(example.value.c_str());
            ImGui::PopStyleColor();
            ImGui::EndGroup();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", example.tooltip.c_str());
        }
    }
    EndCard();
    ImGui::PopID();
}

void QualityAnalyzerDialog::RenderHistograms() {
    const auto& t = ui::CurrentTokens();
    const auto& histograms = view_->histograms;
    const float gap = 14.0f;
    const float width = (ImGui::GetContentRegionAvail().x - gap * (histograms.size() - 1)) / histograms.size();
    for (size_t i = 0; i < histograms.size(); ++i) {
        const auto& h = histograms[i];
        if (i > 0) ImGui::SameLine(0.0f, gap);
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted(h.title.c_str());
        ImGui::PopStyleColor();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        const ImVec2 at = ImGui::GetCursorScreenPos();
        draw->AddRectFilled(at, ImVec2(at.x + width, at.y + kHistogramHeight), ui::ToU32(t.plot_bg), 4.0f);
        const float inner = width - 8.0f;
        const float bar = inner / static_cast<float>(h.heights.size());
        for (size_t b = 0; b < h.heights.size(); ++b) {
            const float height = (std::max)(2.0f, (kHistogramHeight - 4.0f) * h.heights[b]);
            const float x = at.x + 4.0f + bar * b;
            draw->AddRectFilled(ImVec2(x + 1.0f, at.y + kHistogramHeight - height),
                                ImVec2(x + bar - 1.0f, at.y + kHistogramHeight),
                                ui::ToU32(h.cut[b] ? t.warning : t.text_faint), 1.0f);
        }
        ImGui::Dummy(ImVec2(width, kHistogramHeight));
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_faint);
        const float start = ImGui::GetCursorPosX();
        ImGui::TextUnformatted(h.low.c_str());
        ImGui::SameLine(start + (width - ImGui::CalcTextSize(h.cut_label.c_str()).x) * 0.5f);
        ImGui::TextUnformatted(h.cut_label.c_str());
        ImGui::SameLine(start + width - ImGui::CalcTextSize(h.high.c_str()).x);
        ImGui::TextUnformatted(h.high.c_str());
        ImGui::PopStyleColor();
        ImGui::EndGroup();
    }
}

void QualityAnalyzerDialog::RenderDetails() {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_bar);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_button);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 8.0f));
    ImGui::BeginChild("##details", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    if (ImGui::BeginTable("##details_table", 4, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("k1", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("v1", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("k2", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("v2", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        const auto& details = view_->details;
        for (size_t i = 0; i < details.size(); ++i) {
            if (i % 2 == 0) ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_faint);
            ImGui::TextUnformatted(details[i].first.c_str());
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(details[i].second.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

float QualityAnalyzerDialog::FooterHeight() const {
    const auto& t = ui::CurrentTokens();
    const ImGuiStyle& style = ImGui::GetStyle();
    float height = ImGui::GetFontSize() + t.button_padding.y * 2.0f + style.ItemSpacing.y * 3.0f;
    if (show_details_ && view_) {
        const size_t rows = (view_->details.size() + 1) / 2;
        height += rows * ImGui::GetTextLineHeightWithSpacing() + 16.0f + style.ItemSpacing.y * 2.0f + 18.0f;
    }
    return height;
}

void QualityAnalyzerDialog::RenderFooter(bool& should_close) {
    const bool analyzing = IsBusy();
    const bool results = view_.has_value() && !analyzing && context_error_.empty();
    if (show_details_ && results) RenderDetails();

    ImGui::Dummy(ImVec2(0, 2.0f));
    if (results) {
        if (ui::LinkButton(show_details_ ? "Hide details" : "Details")) show_details_ = !show_details_;
        ImGui::SameLine(0.0f, 10.0f);
    }
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_faint);
    ImGui::TextUnformatted(kStoredNote);
    ImGui::PopStyleColor();

    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float ok = ui::ButtonWidth("OK", ui::ButtonSize::Regular);
    const float cancel = ui::ButtonWidth("Cancel", ui::ButtonSize::Regular);
    const float stop = analyzing ? ui::ButtonWidth("Stop", ui::ButtonSize::Regular) + spacing : 0.0f;
    ImGui::SameLine((std::max)(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - ok - cancel - spacing - stop));
    if (analyzing) {
        if (ui::DangerButton("Stop", true, nullptr, ui::ButtonSize::Regular)) {
            job_->cancel = true;
            cyxwiz::AsyncTaskManager::Instance().Cancel(task_id_);
        }
        ImGui::SameLine();
    }
    const bool clicked_ok = results
        ? ui::PrimaryButton("OK", !analyzing, "Wait for the analysis, or Stop it")
        : ui::SecondaryButton("OK", !analyzing, "Wait for the analysis, or Stop it", ui::ButtonSize::Regular);
    if (clicked_ok) {
        Apply();
        should_close = true;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) {
        Reset();
        should_close = true;
    }
}

// The image as the model sees it: decoded and shrunk to the Resize size.
unsigned int QualityAnalyzerDialog::Thumbnail(size_t index) {
    const auto it = thumbnails_.find(index);
    if (it != thumbnails_.end()) return it->second;
    unsigned int texture = 0;
    std::vector<float> pixels;
    int width = 0, height = 0, channels = 0;
    if (analysis_ && index < analysis_->files.size() &&
        cyxwiz::ImageUtils::LoadImage(analysis_->files[index], pixels, width, height, channels) &&
        cyxwiz::ImageUtils::ResizeImage(pixels, width, height, channels, analysis_->width, analysis_->height,
                                        cyxwiz::ImageUtils::ResizeMethod::Area)) {
        width = analysis_->width;
        height = analysis_->height;
        std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4, 255);
        for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
            for (int c = 0; c < 3; ++c) {
                const float v = pixels[p * channels + (channels >= 3 ? c : 0)];
                rgba[p * 4 + c] = static_cast<unsigned char>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        GLuint id = 0;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        texture = id;
    }
    thumbnails_[index] = texture;
    return texture;
}

void QualityAnalyzerDialog::ReleaseThumbnails() {
    for (const auto& [index, texture] : thumbnails_) {
        if (texture != 0) {
            const GLuint id = texture;
            glDeleteTextures(1, &id);
        }
    }
    thumbnails_.clear();
}

}  // namespace gui
