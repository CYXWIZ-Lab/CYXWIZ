#include "image_gallery_view.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_platform.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/live_graph_compile.h"
#include "../../core/texture_manager.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <functional>

namespace cyxwiz {

namespace {

namespace ig = image_gallery;

constexpr float kTile = 96.0f;
constexpr float kDetailWidth = 280.0f;
constexpr int kTileRows = 4;

std::shared_ptr<arrow::Array> Chunk(const std::shared_ptr<arrow::Table>& t, const char* name) {
    const auto column = t->GetColumnByName(name);
    return column && column->num_chunks() > 0 ? column->chunk(0) : nullptr;
}

// A tile: the thumbnail (or its placeholder), the class tag, hover and
// selection washes. Returns true when clicked.
bool Tile(gui::ImageThumbnails& thumbs, const ig::Row& row, const std::string& tag, const std::string& flag, bool selected) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##tile", ImVec2(kTile, kTile));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 end(at.x + kTile, at.y + kTile);
    const auto thumb = thumbs.Get(row.path);
    if (thumb.texture) {
        dl->AddImageRounded(thumb.texture, at, end, ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 4.0f);
    } else {
        dl->AddRectFilled(at, end, ui::ToU32(t.bg_raised), 4.0f);
        if (thumb.failed || !row.readable) {
            const ImVec2 icon = ImGui::CalcTextSize(ICON_FA_TRIANGLE_EXCLAMATION);
            dl->AddText(ImVec2(at.x + (kTile - icon.x) * 0.5f, at.y + (kTile - icon.y) * 0.5f), ui::ToU32(t.caution),
                        ICON_FA_TRIANGLE_EXCLAMATION);
        }
    }
    // The class, bottom left, on a wash of the window tone.
    const ImVec2 size = ImGui::CalcTextSize(tag.c_str());
    const ImVec2 tag_at(at.x + 4.0f, end.y - size.y - 6.0f);
    ImVec4 wash = t.bg_window;
    wash.w = 0.8f;
    dl->AddRectFilled(ImVec2(tag_at.x - 3.0f, tag_at.y - 1.0f), ImVec2(tag_at.x + size.x + 3.0f, tag_at.y + size.y + 1.0f),
                      ui::ToU32(wash), 3.0f);
    dl->AddText(tag_at, ui::ToU32(t.text), tag.c_str());
    // A failed check, top right.
    if (!flag.empty()) {
        const ImVec2 fsize = ImGui::CalcTextSize(flag.c_str());
        const ImVec2 flag_at(end.x - fsize.x - 7.0f, at.y + 5.0f);
        dl->AddRectFilled(ImVec2(flag_at.x - 3.0f, flag_at.y - 1.0f), ImVec2(flag_at.x + fsize.x + 3.0f, flag_at.y + fsize.y + 1.0f),
                          ui::ToU32(wash), 3.0f);
        dl->AddText(flag_at, ui::ToU32(t.caution), flag.c_str());
    }
    if (selected) dl->AddRectFilled(at, end, ui::ToU32(t.selection), 4.0f);
    else if (hovered) dl->AddRectFilled(at, end, ui::ToU32(t.hover), 4.0f);
    if (hovered) ImGui::SetTooltip("%s", row.file.c_str());
    return clicked;
}

}  // namespace

ImageGalleryView::ImageGalleryView()
    : tiles_(static_cast<int>(kTile), gui::ImageThumbnails::Shape::Square, 240),
      preview_(256, gui::ImageThumbnails::Shape::Fit, 24) {}

ImageGalleryView::~ImageGalleryView() {
    if (pipeline_task_) AsyncTaskManager::Instance().Cancel(pipeline_task_);
    ReleasePipeline();
}

void ImageGalleryView::ReleasePipeline() {
    for (uint32_t id : pipeline_textures_)
        if (id) TextureManager::Instance().DeleteTexture(id);
    pipeline_textures_.clear();
}

// The picked image through the graph's Resize and image steps (training's own
// code, on a task), shown below the grid.
void ImageGalleryView::RenderPipeline(const ig::Row& row) {
    const ui::Tokens& t = ui::CurrentTokens();
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    ui::BeginCard("model_sees");
    ImGui::TextUnformatted("What the model sees");
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "\xC2\xB7 %s", row.file.c_str());
    if (!config) {
        ImGui::TextColored(t.text_dim, "The graph is compiling; its image steps show here once it has.");
        ui::EndCard();
        return;
    }
    const PreparedImageSpec spec = PreparedImageSpecFrom(config->image_preprocessing);
    const image::ImageAugmentation steps = config->image_augmentation;
    const std::vector<CvImageStep> cv_steps = config->image_cv_steps;
    const std::string key = row.path + "|" + ImageStepsKey(spec, steps, cv_steps);
    if (key != pipeline_key_ && !pipeline_task_) {
        pipeline_key_ = key;
        auto result = std::make_shared<ImagePipelinePreview>();
        const std::string path = row.path;
        const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(path));
        std::weak_ptr<int> alive = alive_;
        pipeline_task_ = AsyncTaskManager::Instance().RunAsync(
            "What the model sees: " + row.file,
            [path, spec, steps, cv_steps, seed, result](LambdaTask&) {
                *result = PreviewImagePipeline(path, spec, steps, cv_steps, 4, seed);
            },
            nullptr,
            [this, alive, result](bool, const std::string&) {
                if (alive.expired()) return;
                pipeline_task_ = 0;
                ReleasePipeline();
                pipeline_ = std::move(*result);
                auto& textures = TextureManager::Instance();
                const auto upload = [&](const PreviewPicture& p) {
                    pipeline_textures_.push_back(p.rgba.empty() ? 0u : textures.CreateTextureFromUint8Data(p.rgba.data(), p.width, p.height, 4));
                };
                upload(pipeline_.decoded);
                for (const auto& p : pipeline_.fixed) upload(p);
                for (const auto& p : pipeline_.draws) upload(p);
            },
            alive_);
    }
    if (pipeline_task_ || (pipeline_textures_.empty() && pipeline_.error.empty())) {
        ImGui::TextColored(t.running, ICON_FA_SPINNER " Running the graph's image steps on this image");
        ui::EndCard();
        return;
    }
    if (!pipeline_.error.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.caution, "%s", pipeline_.error.c_str());
        ImGui::PopTextWrapPos();
        ui::EndCard();
        return;
    }
    // A picture, 2x (at most 160 px), its title and what was done.
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    bool first = true;
    const auto picture = [&](const PreviewPicture& p, uint32_t texture) {
        const float scale = std::min(2.0f, 160.0f / static_cast<float>(std::max(p.width, p.height)));
        const ImVec2 size(p.width * scale, p.height * scale);
        const float w = std::max(size.x, std::max(ImGui::CalcTextSize(p.title.c_str()).x, ImGui::CalcTextSize(p.detail.c_str()).x));
        if (!first) {
            ImGui::SameLine(0.0f, 16.0f);
            if (ImGui::GetCursorScreenPos().x + std::min(w, 220.0f) > right_edge) ImGui::NewLine();
        }
        first = false;
        ImGui::BeginGroup();
        if (texture) ImGui::Image(static_cast<ImTextureID>(texture), size);
        ImGui::TextUnformatted(p.title.c_str());
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + std::max(size.x, 160.0f));
        ImGui::TextColored(t.text_dim, "%s", p.detail.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    };
    size_t k = 0;
    picture(pipeline_.decoded, pipeline_textures_[k++]);
    for (const auto& p : pipeline_.fixed) picture(p, pipeline_textures_[k++]);
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Random steps \xC2\xB7 per batch");
    if (pipeline_.draws.empty()) {
        ImGui::TextColored(t.text_dim, "None in this graph: every epoch sees the same images.");
    } else {
        std::string ops;
        for (const auto& op : pipeline_.random_ops) ops += (ops.empty() ? "" : ", ") + op;
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "(%s)", ops.c_str());
        first = true;
        for (const auto& p : pipeline_.draws) picture(p, pipeline_textures_[k++]);
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Run with the code training runs, on the ArrayFire device; Normalize is left out "
                                     "so the pictures stay viewable.");
    ImGui::PopTextWrapPos();
    ui::EndCard();
}

void ImageGalleryView::SetDataset(const std::string& dataset) {
    if (dataset == dataset_) return;
    dataset_ = dataset;
    index_seen_ = nullptr;
    input_ = {};
}

void ImageGalleryView::Refresh() {
    auto& registry = DataRegistry::Instance();
    const auto* entry = registry.GetImageDatasetEntry(dataset_);
    const auto index = registry.GetArrowDataset(DataRegistry::ImageIndexName(dataset_));
    const auto table = index ? index->GetArrowTable() : nullptr;
    if (table.get() == index_seen_) return;
    index_seen_ = table.get();
    input_.class_names = entry ? entry->ShownClassNames() : std::vector<std::string>{};
    input_.rows.clear();
    if (!table) return;
    const auto combined = table->CombineChunks().ValueOrDie();
    const auto path = std::static_pointer_cast<arrow::StringArray>(Chunk(combined, "path"));
    const auto file = std::static_pointer_cast<arrow::StringArray>(Chunk(combined, "file"));
    const auto label = std::static_pointer_cast<arrow::Int32Array>(Chunk(combined, "label"));
    const auto width = std::static_pointer_cast<arrow::Int32Array>(Chunk(combined, "width"));
    const auto height = std::static_pointer_cast<arrow::Int32Array>(Chunk(combined, "height"));
    const auto channels = std::static_pointer_cast<arrow::Int32Array>(Chunk(combined, "channels"));
    const auto bytes = std::static_pointer_cast<arrow::Int64Array>(Chunk(combined, "bytes"));
    const auto readable = std::static_pointer_cast<arrow::BooleanArray>(Chunk(combined, "readable"));
    if (!path || !file || !label || !width || !height || !channels || !bytes || !readable) return;
    // Quality columns, when measured.
    const auto blur = std::static_pointer_cast<arrow::FloatArray>(Chunk(combined, "blur"));
    const auto brightness = std::static_pointer_cast<arrow::FloatArray>(Chunk(combined, "brightness"));
    const auto contrast = std::static_pointer_cast<arrow::FloatArray>(Chunk(combined, "contrast"));
    const auto verdict = std::static_pointer_cast<arrow::StringArray>(Chunk(combined, "verdict"));
    const auto duplicate_of = std::static_pointer_cast<arrow::StringArray>(Chunk(combined, "duplicate_of"));
    const bool quality = blur && brightness && contrast && verdict && duplicate_of;
    input_.rows.reserve(static_cast<size_t>(path->length()));
    for (int64_t i = 0; i < path->length(); ++i) {
        ig::Row row;
        row.path = path->GetString(i);
        row.file = file->GetString(i);
        row.label = label->IsNull(i) ? -1 : label->Value(i);
        row.width = width->IsNull(i) ? -1 : width->Value(i);
        row.height = height->IsNull(i) ? -1 : height->Value(i);
        row.channels = channels->IsNull(i) ? -1 : channels->Value(i);
        row.bytes = bytes->Value(i);
        row.readable = readable->Value(i);
        if (quality && !verdict->IsNull(i)) {
            row.measured = true;
            row.blur = blur->Value(i);
            row.brightness = brightness->Value(i);
            row.contrast = contrast->Value(i);
            row.verdict = verdict->GetString(i);
            if (!duplicate_of->IsNull(i)) row.duplicate_of = duplicate_of->GetString(i);
        }
        input_.rows.push_back(std::move(row));
    }
}

void ImageGalleryView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    tiles_.Pump();
    preview_.Pump();
    if (dataset_.empty()) return;
    Refresh();
    if (!index_seen_) {
        ImGui::TextColored(t.text_dim, "The images show once their index is built (Bring in lists it under the Data Input).");
        return;
    }

    const bool side = ImGui::GetContentRegionAvail().x >= 760.0f;
    const float grid_w = side ? ImGui::GetContentRegionAvail().x - kDetailWidth - ImGui::GetStyle().ItemSpacing.x * 2.0f
                              : ImGui::GetContentRegionAvail().x;
    const float step = kTile + ImGui::GetStyle().ItemSpacing.x;
    const int columns = std::max(1, static_cast<int>((grid_w + ImGui::GetStyle().ItemSpacing.x) / step));
    input_.per_page = static_cast<size_t>(columns * kTileRows);
    ig::View view = ig::Build(input_);
    input_.filter = "all";
    for (const auto& chip : view.chips)
        if (chip.on) input_.filter = chip.key;
    input_.page = view.page;

    // Filters, sort, pages.
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    bool first = true;
    for (const auto& chip : view.chips) {
        const std::string& count = chip.count_text;
        if (!first) {
            ImGui::SameLine();
            if (ImGui::GetCursorScreenPos().x + ui::FilterChipWidth(chip.label.c_str(), count.c_str()) > right_edge) ImGui::NewLine();
        }
        first = false;
        if (ui::FilterChip(chip.key.c_str(), chip.label.c_str(), count.c_str(), chip.on)) {
            input_.filter = chip.key;
            input_.page = 0;
        }
    }
    static const char* kSorts[] = {"Dataset order", "Smallest", "Largest", "Blurriest"};
    if (!view.quality && input_.sort == ig::Sort::Blurriest) input_.sort = ig::Sort::DatasetOrder;
    int sort = static_cast<int>(input_.sort);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Sort");
    ImGui::SameLine();
    if (ui::SegmentedControl("##gallery_sort", kSorts, view.quality ? 4 : 3, &sort)) {
        input_.sort = static_cast<ig::Sort>(sort);
        input_.page = 0;
    }
    ImGui::SameLine(0.0f, 16.0f);
    if (ui::SecondaryButton(ICON_FA_CHEVRON_LEFT "##previous", view.has_previous, "This is the first page",
                            ui::ButtonSize::Small))
        input_.page = view.page - 1;
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_CHEVRON_RIGHT "##next", view.has_next, "This is the last page", ui::ButtonSize::Small))
        input_.page = view.page + 1;
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", view.showing.c_str());
    ImGui::Spacing();

    // The grid.
    ImGui::BeginGroup();
    for (size_t k = 0; k < view.shown.size(); ++k) {
        const size_t i = view.shown[k];
        if (k % static_cast<size_t>(columns) != 0) ImGui::SameLine();
        ImGui::PushID(static_cast<int>(i));
        if (Tile(tiles_, input_.rows[i], view.tags[k], view.flags[k], static_cast<long long>(i) == view.selected))
            input_.selected = static_cast<long long>(i);
        ImGui::PopID();
    }
    if (view.shown.empty()) ImGui::Dummy(ImVec2(grid_w, kTile));
    ImGui::EndGroup();

    // The picked image.
    if (side) ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x * 2.0f);
    ImGui::BeginGroup();
    view = ig::Build(input_);  // a click this frame
    if (view.selected < 0) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + kDetailWidth);
        ImGui::TextColored(t.text_dim, "Click an image for its details.");
        ImGui::PopTextWrapPos();
    } else {
        const ig::Row& row = input_.rows[static_cast<size_t>(view.selected)];
        const auto big = preview_.Get(row.path);
        if (big.texture) {
            const float scale = std::min(1.0f, kDetailWidth / static_cast<float>(std::max(big.width, big.height)));
            ImGui::Image(big.texture, ImVec2(big.width * scale, big.height * scale));
        } else {
            ImGui::Dummy(ImVec2(kDetailWidth, kDetailWidth * 0.6f));
        }
        ImGui::TextUnformatted(view.detail_title.c_str());
        if (ImGui::BeginTable("##image_detail", 2, ImGuiTableFlags_SizingStretchProp, ImVec2(kDetailWidth, 0.0f))) {
            float label_w = 0.0f;  // the longest label ("Brightness")
            for (const auto& item : view.detail) label_w = std::max(label_w, ImGui::CalcTextSize(item.first.c_str()).x);
            ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthFixed, label_w + 8.0f);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            for (const auto& [what, value] : view.detail) {
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
        if (ui::LinkButton("Show in folder")) ui::ShowInFileManager(row.path);
        if (on_open_index) {
            ImGui::SameLine(0.0f, 12.0f);
            if (ui::LinkButton("Open the index table")) on_open_index(DataRegistry::ImageIndexName(dataset_));
        }
    }
    ImGui::EndGroup();
    if (view.selected >= 0) {
        ImGui::Spacing();
        RenderPipeline(input_.rows[static_cast<size_t>(view.selected)]);
    }
}

void ImageGalleryView::RenderStrip(size_t count) {
    tiles_.Pump();
    if (dataset_.empty()) return;
    Refresh();
    if (!index_seen_) {
        ImGui::TextColored(ui::CurrentTokens().text_dim, "Indexing the images...");
        return;
    }
    const float step = kTile + ImGui::GetStyle().ItemSpacing.x;
    const int columns = std::max(1, static_cast<int>((ImGui::GetContentRegionAvail().x + ImGui::GetStyle().ItemSpacing.x) / step));
    ig::Input first = input_;
    first.filter = "all";
    first.sort = ig::Sort::DatasetOrder;
    first.page = 0;
    first.per_page = count;
    const ig::View view = ig::Build(first);
    for (size_t k = 0; k < view.shown.size(); ++k) {
        if (k % static_cast<size_t>(columns) != 0) ImGui::SameLine();
        ImGui::PushID(static_cast<int>(view.shown[k]));
        Tile(tiles_, input_.rows[view.shown[k]], view.tags[k], view.flags[k], false);
        ImGui::PopID();
    }
}

}  // namespace cyxwiz
