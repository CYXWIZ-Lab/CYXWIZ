#include "image_profile_view.h"

#include "explore_plots.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/image_channel_stats.h"
#include "../../core/image_index_service.h"
#include "../../core/image_quality_analysis.h"
#include "../../core/image_pipeline_preview.h"
#include "../../core/live_graph_compile.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <vector>

namespace cyxwiz {

namespace {

namespace ip = image_profile;

const char* ModeName(ResizeMode mode) {
    switch (mode) {
        case ResizeMode::AspectFit: return "fit";
        case ResizeMode::AspectFill: return "fill";
        case ResizeMode::Center: return "center";
        default: return "exact";
    }
}

// The int32 / int64 column `name` of a one-chunk table, -1 for nulls.
template <typename ArrayType, typename T>
std::vector<T> Column(const std::shared_ptr<arrow::Table>& table, const char* name) {
    std::vector<T> out;
    const auto column = table->GetColumnByName(name);
    if (!column || column->num_chunks() == 0) return out;
    const auto a = std::static_pointer_cast<ArrayType>(column->chunk(0));
    out.reserve(static_cast<size_t>(a->length()));
    for (int64_t i = 0; i < a->length(); ++i) out.push_back(a->IsNull(i) ? T(-1) : static_cast<T>(a->Value(i)));
    return out;
}

void Tile(const ip::Tile& tile) {
    ImGui::BeginGroup();
    {
        ui::FontScope bold(ui::Font::Bold);
        ImGui::TextUnformatted(tile.value.c_str());
    }
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", tile.label.c_str());
    ImGui::EndGroup();
}

// A share bar: the track in the plot tone, the share in the accent.
void ShareBar(float share, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight() * 0.6f;
    const float y = at.y + (ImGui::GetTextLineHeight() - h) * 0.5f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + width, y + h), ui::ToU32(t.plot_bg), 3.0f);
    if (share > 0.0f)
        dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + std::max(2.0f, width * share), y + h), ui::ToU32(t.accent_text), 3.0f);
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
}

void Histogram(const char* title, const ip::Histogram& h, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextUnformatted(title);
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "px");
    if (h.counts.empty()) {
        ImGui::TextColored(t.text_dim, "No readable images.");
        return;
    }
    const std::vector<int64_t> bins(h.counts.begin(), h.counts.end());
    explore_plot::Bars(bins, h.edges.front(), h.edges.back(), width, 90.0f);
}

}  // namespace

ImageProfileView::ImageProfileView() = default;

ImageProfileView::~ImageProfileView() {
    if (sets_task_) AsyncTaskManager::Instance().Cancel(sets_task_);
    if (stats_task_) AsyncTaskManager::Instance().Cancel(stats_task_);
    if (quality_task_) AsyncTaskManager::Instance().Cancel(quality_task_);
}

// The verdict's counts into the model, and the quality columns onto the index.
void ImageProfileView::ShowQuality(const std::shared_ptr<const ImageQualityAnalysis>& analysis,
                                   const ImageQualityVerdict& verdict) {
    input_.quality_step = ip::Input::QualityStep::Measured;
    input_.quality_device = analysis->device;
    input_.rejected = verdict.rejected;
    input_.blurry = verdict.blurry;
    input_.dark = verdict.dark;
    input_.bright = verdict.bright;
    input_.low_contrast = verdict.low_contrast;
    input_.duplicates = verdict.duplicates;
    input_.multiple = verdict.multiple;
    input_.class_total = verdict.class_total;
    input_.class_rejected = verdict.class_rejected;
    AttachImageQuality(dataset_, analysis, verdict);
}

void ImageProfileView::SetDataset(const std::string& dataset) {
    if (dataset == dataset_) return;
    dataset_ = dataset;
    index_seen_ = nullptr;
    graph_seen_.clear();
    input_ = {};
    view_ = ip::Build(input_);
}

std::string ImageProfileView::GraphIdentity() const {
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    if (!config) return "no graph";
    return ImageStepsKey(PreparedImageSpecFrom(config->image_preprocessing), config->image_augmentation,
                         config->image_cv_steps);
}

void ImageProfileView::Refresh() {
    auto& registry = DataRegistry::Instance();
    const DataRegistry::ImageDatasetEntry* entry = registry.GetImageDatasetEntry(dataset_);
    const auto index = registry.GetArrowDataset(DataRegistry::ImageIndexName(dataset_));
    const auto table = index ? index->GetArrowTable() : nullptr;
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    const int resize_width = config ? config->image_preprocessing.target_width : 0;
    const int resize_height = config ? config->image_preprocessing.target_height : 0;
    const std::string mode = config ? ModeName(config->image_preprocessing.resize_mode) : "";
    if (table.get() == index_seen_ && resize_width == input_.resize_width && resize_height == input_.resize_height &&
        mode == input_.resize_mode)
        return;
    index_seen_ = table.get();
    input_.dataset = dataset_;
    input_.normalize = config && config->preprocessing.has_normalization;
    input_.norm_mean = config ? config->preprocessing.norm_mean : 0.0f;
    input_.norm_std = config ? config->preprocessing.norm_std : 1.0f;
    input_.resize_width = resize_width;
    input_.resize_height = resize_height;
    input_.resize_mode = mode;
    input_.labels_source =
        entry && !entry->labels_csv.empty() ? std::filesystem::path(entry->labels_csv).filename().string() : std::string();
    input_.class_names = entry ? entry->ShownClassNames() : std::vector<std::string>{};
    input_.labels.clear();
    input_.widths.clear();
    input_.heights.clear();
    input_.channels.clear();
    input_.bytes.clear();
    if (table) {
        const auto combined = table->CombineChunks().ValueOrDie();
        input_.labels = Column<arrow::Int32Array, int>(combined, "label");
        input_.widths = Column<arrow::Int32Array, int>(combined, "width");
        input_.heights = Column<arrow::Int32Array, int>(combined, "height");
        input_.channels = Column<arrow::Int32Array, int>(combined, "channels");
        input_.bytes = Column<arrow::Int64Array, int64_t>(combined, "bytes");
    }
    view_ = ip::Build(input_);
}

void ImageProfileView::ListSets() {
    const DataRegistry::ImageDatasetEntry* found = DataRegistry::Instance().GetImageDatasetEntry(dataset_);
    if (!found) return;
    graph_seen_ = GraphIdentity();
    const DataRegistry::ImageDatasetEntry entry = *found;
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    const PreparedImageSpec spec = config ? PreparedImageSpecFrom(config->image_preprocessing) : PreparedImageSpec{};
    const std::vector<FixedImageStep> fixed =
        config ? SplitFixedImageTransforms(config->image_augmentation, config->image_cv_steps).fixed
               : std::vector<FixedImageStep>{};
    if (sets_task_) AsyncTaskManager::Instance().Cancel(sets_task_);
    input_.sets_listed = false;
    view_ = ip::Build(input_);
    auto sets = std::make_shared<std::vector<ip::PreparedSet>>();
    auto quality = std::make_shared<std::shared_ptr<const ImageQualityAnalysis>>();
    auto verdict = std::make_shared<ImageQualityVerdict>();
    std::weak_ptr<int> alive = alive_;
    const std::string dataset = dataset_;
    if (input_.quality_step != ip::Input::QualityStep::Measuring) input_.quality_step = ip::Input::QualityStep::Looking;
    sets_task_ = AsyncTaskManager::Instance().RunAsync(
        "Prepared image sets " + dataset,
        [entry, spec, fixed, sets, quality, verdict](LambdaTask&) {
            const std::string in_use = spec.width > 0 ? PreparedImageSetKeyFor(entry, spec, fixed) : std::string();
            for (const auto& s : ListPreparedImageSets(entry))
                sets->push_back({s.width, s.height, s.channels, s.mode, s.fixed, s.bytes, s.key == in_use});
            if (spec.width <= 0) return;
            *quality = LoadImageQualityAnalysis(ImageQualityKey(entry, spec.width, spec.height));
            if (*quality) *verdict = JudgeImageQualityAnalysis(**quality, image::ImageQualityChecks{});
        },
        nullptr,
        [this, alive, sets, quality, verdict, dataset](bool success, const std::string&) {
            if (alive.expired() || dataset != dataset_) return;
            sets_task_ = 0;
            if (!success) return;
            input_.sets = *sets;
            input_.sets_listed = true;
            if (input_.quality_step == ip::Input::QualityStep::Looking) {
                if (*quality) ShowQuality(*quality, *verdict);
                else input_.quality_step = ip::Input::QualityStep::NotMeasured;
            }
            view_ = ip::Build(input_);
        },
        alive_);
}

void ImageProfileView::MeasureStats() {
    const DataRegistry::ImageDatasetEntry* found = DataRegistry::Instance().GetImageDatasetEntry(dataset_);
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    if (!found || !config) return;
    const DataRegistry::ImageDatasetEntry entry = *found;
    const PreparedImageSpec spec = PreparedImageSpecFrom(config->image_preprocessing);
    const std::vector<FixedImageStep> fixed =
        SplitFixedImageTransforms(config->image_augmentation, config->image_cv_steps).fixed;
    stats_for_ = GraphIdentity();
    input_.channel_mean.clear();
    input_.channel_std.clear();
    input_.stats_error.clear();
    using Step = ip::Input::StatsStep;
    auto progress = stats_progress_;
    {
        std::lock_guard<std::mutex> lock(progress->mutex);
        progress->step = Step::Preparing;
        progress->done = progress->total = 0;
        progress->device.clear();
    }
    auto result = std::make_shared<ImageChannelStats>();
    auto error = std::make_shared<std::string>();
    auto device = std::make_shared<std::string>();
    auto size = std::make_shared<std::pair<int, int>>();
    std::weak_ptr<int> alive = alive_;
    stats_task_ = AsyncTaskManager::Instance().RunAsync(
        "Channel statistics " + dataset_,
        [progress, entry, spec, fixed, result, error, device, size](LambdaTask& task) {
            const PreparedImageResult prepared = PrepareImageSet(entry, spec, fixed, 0, [&](size_t done, size_t total) {
                std::lock_guard<std::mutex> lock(progress->mutex);
                progress->done = done;
                progress->total = total;
                return !task.ShouldStop();
            });
            if (!prepared.set) {
                *error = "The images could not be prepared: " + prepared.error;
                return;
            }
            *device = ImageQualityDeviceLabel();
            {
                std::lock_guard<std::mutex> lock(progress->mutex);
                progress->step = Step::Measuring;
                progress->device = *device;
            }
            std::atomic<bool> stop{false};
            *result = MeasureImageChannelStats(*prepared.set, [&](size_t, size_t) { stop = task.ShouldStop(); }, &stop);
            *size = {static_cast<int>(prepared.set->Width()), static_cast<int>(prepared.set->Height())};
        },
        nullptr,
        [this, alive, result, error, device, size](bool success, const std::string& message) {
            if (alive.expired()) return;
            stats_task_ = 0;
            {
                std::lock_guard<std::mutex> lock(stats_progress_->mutex);
                stats_progress_->step = ip::Input::StatsStep::Idle;
            }
            if (!error->empty() || !success) input_.stats_error = !error->empty() ? *error : message;
            input_.channel_mean = result->mean;
            input_.channel_std = result->std_dev;
            input_.stats_images = result->images;
            input_.stats_device = *device;
            input_.stats_width = size->first;
            input_.stats_height = size->second;
            input_.sets_listed = false;  // the set is in cache/images now
            graph_seen_.clear();
            view_ = ip::Build(input_);
        },
        alive_);
}

void ImageProfileView::MeasureQuality() {
    const DataRegistry::ImageDatasetEntry* found = DataRegistry::Instance().GetImageDatasetEntry(dataset_);
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    if (!found || !config) return;
    const DataRegistry::ImageDatasetEntry entry = *found;
    const PreparedImageSpec spec = PreparedImageSpecFrom(config->image_preprocessing);
    auto progress = quality_progress_;
    {
        std::lock_guard<std::mutex> lock(progress->mutex);
        progress->done = progress->total = 0;
    }
    input_.quality_step = ip::Input::QualityStep::Measuring;
    input_.quality_error.clear();
    view_ = ip::Build(input_);
    auto analysis = std::make_shared<std::shared_ptr<const ImageQualityAnalysis>>();
    auto verdict = std::make_shared<ImageQualityVerdict>();
    auto error = std::make_shared<std::string>();
    std::weak_ptr<int> alive = alive_;
    quality_task_ = AsyncTaskManager::Instance().RunAsync(
        "Image quality " + dataset_,
        [entry, spec, progress, analysis, verdict, error](LambdaTask& task) {
            std::atomic<bool> cancel{false};
            auto measured = AnalyzeImageQuality(entry, spec.width, spec.height, [&](size_t done, size_t total) {
                cancel = task.ShouldStop();
                std::lock_guard<std::mutex> lock(progress->mutex);
                progress->done = done;
                progress->total = total;
            }, &cancel);
            if (!measured) return;  // cancelled
            std::string why;
            if (!SaveImageQualityAnalysis(*measured, &why)) *error = "Measured, but not saved: " + why;
            *analysis = std::make_shared<const ImageQualityAnalysis>(std::move(*measured));
            *verdict = JudgeImageQualityAnalysis(**analysis, image::ImageQualityChecks{});
        },
        nullptr,
        [this, alive, analysis, verdict, error](bool success, const std::string& message) {
            if (alive.expired()) return;
            quality_task_ = 0;
            if (*analysis) ShowQuality(*analysis, *verdict);
            else input_.quality_step = ip::Input::QualityStep::NotMeasured;
            if (!error->empty()) input_.quality_error = *error;
            else if (!success) input_.quality_error = "The images could not be measured: " + message;
            view_ = ip::Build(input_);
        },
        alive_);
}

void ImageProfileView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (dataset_.empty()) return;
    if (!DataRegistry::Instance().GetImageDatasetEntry(dataset_)) {
        ImGui::TextColored(t.text_dim, "'%s' is no longer loaded.", dataset_.c_str());
        return;
    }
    Refresh();
    if (!stats_for_.empty() && !stats_task_ && GraphIdentity() != stats_for_) {
        stats_for_.clear();  // the graph's image steps changed: the statistics were of other images
        input_.channel_mean.clear();
        input_.channel_std.clear();
        input_.stats_error.clear();
        view_ = ip::Build(input_);
    }
    if (GraphIdentity() != graph_seen_ && !sets_task_) ListSets();
    if (quality_task_) {
        std::lock_guard<std::mutex> lock(quality_progress_->mutex);
        if (quality_progress_->done != input_.quality_done || quality_progress_->total != input_.quality_total) {
            input_.quality_done = quality_progress_->done;
            input_.quality_total = quality_progress_->total;
            view_ = ip::Build(input_);
        }
    }
    {
        std::lock_guard<std::mutex> lock(stats_progress_->mutex);
        const Progress& p = *stats_progress_;
        if (p.step != input_.stats_step || p.done != input_.stats_done || p.total != input_.stats_total) {
            input_.stats_step = p.step;
            input_.stats_done = p.done;
            input_.stats_total = p.total;
            if (p.step == ip::Input::StatsStep::Measuring) input_.stats_device = p.device;
            view_ = ip::Build(input_);
        }
    }

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(dataset_.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "images \xC2\xB7 index %s", DataRegistry::ImageIndexName(dataset_).c_str());
    if (!index_seen_) {
        const ImageIndexStatus status = GetImageIndexStatus(dataset_);
        if (status.state == ImageIndexStatus::State::Failed) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.error, "The index could not be built: %s", status.error.c_str());
            ImGui::PopTextWrapPos();
        } else {
            ImGui::TextColored(t.running, ICON_FA_SPINNER " Indexing images: %zu of %zu read", status.done, status.total);
            ImGui::ProgressBar(status.total ? static_cast<float>(status.done) / static_cast<float>(status.total) : 0.0f,
                               ImVec2(std::min(420.0f, ImGui::GetContentRegionAvail().x), 4.0f), "");
        }
        return;
    }

    // Tiles.
    ImGui::Spacing();
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    for (size_t i = 0; i < view_.tiles.size(); ++i) {
        if (i > 0) {
            const float w = std::max(ImGui::CalcTextSize(view_.tiles[i].value.c_str()).x,
                                     ImGui::CalcTextSize(view_.tiles[i].label.c_str()).x);
            ImGui::SameLine(0.0f, 32.0f);
            if (ImGui::GetCursorScreenPos().x + w > right_edge) ImGui::NewLine();
        }
        Tile(view_.tiles[i]);
    }
    ImGui::Spacing();

    // Classes and sizes, then the two histograms: side by side when there is room.
    const bool two = ImGui::GetContentRegionAvail().x >= 760.0f;
    const auto pair = [two](const char* id, const std::function<void()>& left, const std::function<void()>& right) {
        if (!two) {
            left();
            right();
            return;
        }
        if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchSame)) return;
        ImGui::TableNextColumn();
        left();
        ImGui::TableNextColumn();
        right();
        ImGui::EndTable();
    };
    pair("##classes_sizes", [&] {
        ui::BeginCard("image_classes");
        ImGui::TextUnformatted(view_.classes_title.c_str());
        if (ImGui::BeginTable("##classes", 3, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("class", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("bar", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("count", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000,000 \xC2\xB7 100%").x);
            for (const auto& row : view_.classes) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (row.name == std::to_string(row.label)) ImGui::TextUnformatted(row.name.c_str());
                else ImGui::Text("%d \xC2\xB7 %s", row.label, row.name.c_str());
                ImGui::TableNextColumn();
                ShareBar(row.share, ImGui::GetContentRegionAvail().x);
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", row.text.c_str());
            }
            ImGui::EndTable();
        }
        if (!view_.balance.empty()) ImGui::TextColored(t.text_dim, "%s", view_.balance.c_str());
        ui::EndCard();
    }, [&] {
        ui::BeginCard("image_sizes");
        ImGui::TextUnformatted("Original size");
        if (ImGui::BeginTable("##sizes", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Resize 0000 \xC3\x97 0000").x);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            for (const auto& [what, value] : view_.sizes) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", what.c_str());
                ImGui::TableNextColumn();
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(value.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::EndTable();
        }
        ui::EndCard();
    });
    pair("##histograms", [&] {
        ui::BeginCard("image_widths");
        Histogram("Width", view_.widths, ImGui::GetContentRegionAvail().x);
        ui::EndCard();
    }, [&] {
        ui::BeginCard("image_heights");
        Histogram("Height", view_.heights, ImGui::GetContentRegionAvail().x);
        ui::EndCard();
    });

    // Channel statistics.
    ui::BeginCard("image_stats");
    ImGui::TextUnformatted(view_.stats_title.c_str());
    if (!view_.stats.empty() && ImGui::BeginTable("##stats", 1 + static_cast<int>(view_.channels.size()),
                                                   ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("");
        for (const auto& c : view_.channels) ImGui::TableSetupColumn(c.c_str(), ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();
        for (const auto& row : view_.stats) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", row.what.c_str());
            for (const auto& value : row.values) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(value.c_str());
            }
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    const bool measuring = input_.stats_step != ip::Input::StatsStep::Idle;
    ImGui::TextColored(measuring ? t.running : t.text_dim, "%s", view_.stats_note.c_str());
    ImGui::PopTextWrapPos();
    if (!measuring &&
        ui::SecondaryButton(view_.stats.empty() ? "Measure" : "Measure again", view_.can_measure,
                            "The graph needs a Resize node"))
        MeasureStats();
    ui::EndCard();

    // Quality.
    ui::BeginCard("image_quality");
    ImGui::TextUnformatted(view_.quality_title.c_str());
    if (!view_.quality_summary.empty()) ImGui::TextUnformatted(view_.quality_summary.c_str());
    if (!view_.quality.empty() && ImGui::BeginTable("##quality", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("reason", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("low contrast").x + 12.0f);
        ImGui::TableSetupColumn("bar", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("count", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000,000 \xC2\xB7 100.0%").x);
        for (const auto& row : view_.quality) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.reason.c_str());
            ImGui::TableNextColumn();
            ShareBar(row.of_rejected, ImGui::GetContentRegionAvail().x);
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", row.text.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    const bool quality_running = input_.quality_step == ip::Input::QualityStep::Measuring;
    ImGui::TextColored(quality_running ? t.running : t.text_dim, "%s", view_.quality_note.c_str());
    ImGui::PopTextWrapPos();
    if (input_.quality_step == ip::Input::QualityStep::NotMeasured || input_.quality_step == ip::Input::QualityStep::Measured) {
        if (ui::SecondaryButton(view_.quality_button.c_str(), view_.can_measure_quality, "The graph needs a Resize node"))
            MeasureQuality();
    }
    ui::EndCard();

    // Prepared sets.
    ImGui::Spacing();
    ui::BeginCard("image_sets");
    ImGui::TextUnformatted("Prepared sets");
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "\xC2\xB7 cache/images, decoded once at a Resize size");
    if (!view_.sets.empty() &&
        ImGui::BeginTable("##sets", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0000 \xC3\x97 0000 \xC3\x97 0").x + 12.0f);
        ImGui::TableSetupColumn("resize", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("center").x + 12.0f);
        ImGui::TableSetupColumn("fixed transforms", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("on disk", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000.0 GB").x + 12.0f);
        ImGui::TableSetupColumn("##use", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("used by this graph").x + 6.0f);
        ImGui::TableHeadersRow();
        for (const auto& row : view_.sets) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.size.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.resize.c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            if (row.fixed.empty()) ImGui::TextColored(t.text_faint, "none");
            else ImGui::TextUnformatted(row.fixed.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.bytes.c_str());
            ImGui::TableNextColumn();
            if (row.in_use) ImGui::TextColored(t.success, "used by this graph");
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", view_.sets_note.c_str());
    ImGui::PopTextWrapPos();
    ui::EndCard();
}

}  // namespace cyxwiz
