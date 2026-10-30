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
#include <imgui_internal.h>  // SetKeyOwner

#include <algorithm>
#include <cmath>
#include <functional>

namespace cyxwiz {

namespace {

namespace ig = image_gallery;

constexpr int kThumbSize = 192;       // decoded thumbnails; tiles up to kMaxTile scale them
constexpr int kMinTile = 64, kMaxTile = 256, kTileStep = 16;
constexpr float kStripTile = 96.0f;   // the Data Input dialog's strip
constexpr float kDetailWidth = 300.0f;
constexpr float kListThumb = 44.0f;
constexpr size_t kMaxPerPage = 200;   // within the thumbnail cache

std::shared_ptr<arrow::Array> Chunk(const std::shared_ptr<arrow::Table>& t, const char* name) {
    const auto column = t->GetColumnByName(name);
    return column && column->num_chunks() > 0 ? column->chunk(0) : nullptr;
}

// What a tile shows over its picture.
struct TileText {
    const std::string* tag = nullptr;   // the class, bottom left
    const std::string* name = nullptr;  // the file name, bottom right
    const std::string* size = nullptr;  // "1023 x 768", top left
    const std::string* flag = nullptr;  // a failed check, top right
};

// The texture coordinates that crop a thumbnail to the tile's square
// (fill) or keep it whole (fit), and where it goes in the tile.
void Placement(const gui::ImageThumbnails::Thumb& thumb, bool fill, const ImVec2& at, float size, ImVec2* p0, ImVec2* p1,
               ImVec2* uv0, ImVec2* uv1) {
    *uv0 = ImVec2(0, 0);
    *uv1 = ImVec2(1, 1);
    *p0 = at;
    *p1 = ImVec2(at.x + size, at.y + size);
    if (thumb.width <= 0 || thumb.height <= 0) return;
    const float aspect = static_cast<float>(thumb.width) / static_cast<float>(thumb.height);
    if (fill) {
        if (aspect > 1.0f) *uv0 = ImVec2((1.0f - 1.0f / aspect) * 0.5f, 0.0f), *uv1 = ImVec2(1.0f - uv0->x, 1.0f);
        else *uv0 = ImVec2(0.0f, (1.0f - aspect) * 0.5f), *uv1 = ImVec2(1.0f, 1.0f - uv0->y);
    } else {
        const float w = aspect >= 1.0f ? size : size * aspect;
        const float h = aspect >= 1.0f ? size / aspect : size;
        *p0 = ImVec2(at.x + (size - w) * 0.5f, at.y + (size - h) * 0.5f);
        *p1 = ImVec2(p0->x + w, p0->y + h);
    }
}

// A label on a wash of the window tone.
void Label(ImDrawList* dl, const ImVec2& at, const std::string& text, const ImVec4& colour) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    dl->AddRectFilled(ImVec2(at.x - 3.0f, at.y - 1.0f), ImVec2(at.x + size.x + 3.0f, at.y + size.y + 1.0f),
                      ui::ToU32(ui::WithAlpha(t.bg_window, 0.8f)), 3.0f);
    dl->AddText(at, ui::ToU32(colour), text.c_str());
}

// A tile: the thumbnail (or its placeholder), its labels, hover and
// selection washes. Returns true when clicked; `opened` on a double-click.
bool Tile(gui::ImageThumbnails& thumbs, const ig::Row& row, float size, bool fill, const TileText& text, bool selected,
          bool* opened = nullptr) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##tile", ImVec2(size, size));
    const bool hovered = ImGui::IsItemHovered();
    if (opened && hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) *opened = true;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 end(at.x + size, at.y + size);
    const auto thumb = thumbs.Get(row.path);
    dl->AddRectFilled(at, end, ui::ToU32(t.bg_input), 4.0f);
    if (thumb.texture) {
        ImVec2 p0, p1, uv0, uv1;
        Placement(thumb, fill, at, size, &p0, &p1, &uv0, &uv1);
        dl->AddImageRounded(thumb.texture, p0, p1, uv0, uv1, IM_COL32_WHITE, fill ? 4.0f : 0.0f);
    } else if (thumb.failed || !row.readable) {
        const ImVec2 icon = ImGui::CalcTextSize(ICON_FA_TRIANGLE_EXCLAMATION);
        dl->AddText(ImVec2(at.x + (size - icon.x) * 0.5f, at.y + (size - icon.y) * 0.5f), ui::ToU32(t.caution),
                    ICON_FA_TRIANGLE_EXCLAMATION);
    }
    dl->PushClipRect(at, end, true);
    const float line = ImGui::GetTextLineHeight();
    if (text.size && !text.size->empty()) Label(dl, ImVec2(at.x + 5.0f, at.y + 4.0f), *text.size, t.text_dim);
    if (text.flag && !text.flag->empty()) {
        const float w = ImGui::CalcTextSize(text.flag->c_str()).x;
        Label(dl, ImVec2(end.x - w - 7.0f, at.y + 4.0f), *text.flag, t.caution);
    }
    float tag_w = 0.0f;
    if (text.tag && !text.tag->empty()) {
        Label(dl, ImVec2(at.x + 5.0f, end.y - line - 5.0f), *text.tag, t.text);
        tag_w = ImGui::CalcTextSize(text.tag->c_str()).x + 12.0f;
    }
    if (text.name && !text.name->empty()) {
        const float w = ImGui::CalcTextSize(text.name->c_str()).x;
        Label(dl, ImVec2(std::max(at.x + 5.0f + tag_w, end.x - w - 7.0f), end.y - line - 5.0f), *text.name, t.text_dim);
    }
    dl->PopClipRect();
    if (selected) {
        dl->AddRectFilled(at, end, ui::ToU32(t.selection), 4.0f);
        dl->AddRect(ImVec2(at.x + 1.0f, at.y + 1.0f), ImVec2(end.x - 1.0f, end.y - 1.0f), ui::ToU32(t.accent), 4.0f, 0, 2.0f);
    }
    else if (hovered) dl->AddRectFilled(at, end, ui::ToU32(t.hover), 4.0f);
    if (hovered) ImGui::SetTooltip("%s", row.file.c_str());
    return clicked;
}

}  // namespace

ImageGalleryView::ImageGalleryView()
    : tiles_(kThumbSize, gui::ImageThumbnails::Shape::Fit, 240),
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
// code, on a task).
void ImageGalleryView::UpdatePipeline(const ig::Row& row) {
    const TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
    if (!config) return;
    const PreparedImageSpec spec = PreparedImageSpecFrom(config->image_preprocessing);
    const image::ImageAugmentation steps = config->image_augmentation;
    const std::vector<CvImageStep> cv_steps = config->image_cv_steps;
    const std::string key = row.path + "|" + ImageStepsKey(spec, steps, cv_steps);
    if (key == pipeline_key_ || pipeline_task_) return;
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

ImageViewerView::ModelPicture ImageGalleryView::ModelPicture() const {
    ImageViewerView::ModelPicture out;
    // The last fixed step is what every epoch feeds the model (before the
    // random steps and Normalize).
    if (pipeline_task_ || pipeline_textures_.empty() || !pipeline_.error.empty()) return out;
    const size_t k = pipeline_.fixed.size();  // decoded is 0, the fixed steps follow
    const PreviewPicture& p = k == 0 ? pipeline_.decoded : pipeline_.fixed.back();
    if (k >= pipeline_textures_.size() || !pipeline_textures_[k]) return out;
    out.texture = static_cast<ImTextureID>(pipeline_textures_[k]);
    out.width = p.width;
    out.height = p.height;
    out.title = p.title;
    return out;
}

// The graph's steps on the picked image, shown below the grid.
void ImageGalleryView::RenderPipeline(const ig::Row& row) {
    const ui::Tokens& t = ui::CurrentTokens();
    UpdatePipeline(row);
    ui::BeginCard("model_sees");
    if (!live_compile_ || !live_compile_->Config()) {
        ImGui::TextColored(t.text_dim, "The graph is compiling; its image steps show here once it has.");
        ui::EndCard();
        return;
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
    ++rows_generation_;
    layout_ = Layout::Grid;
}

void ImageGalleryView::Refresh() {
    auto& registry = DataRegistry::Instance();
    const auto* entry = registry.GetImageDatasetEntry(dataset_);
    const auto index = registry.GetArrowDataset(DataRegistry::ImageIndexName(dataset_));
    const auto table = index ? index->GetArrowTable() : nullptr;
    if (table.get() == index_seen_) return;
    index_seen_ = table.get();
    ++rows_generation_;
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

const ig::View& ImageGalleryView::CurrentView() {
    const std::string key = input_.filter + '|' + std::to_string(static_cast<int>(input_.sort)) + '|' +
                            std::to_string(input_.page) + '|' + std::to_string(input_.per_page) + '|' +
                            std::to_string(input_.selected) + '|' + std::to_string(input_.strip_radius) + '|' +
                            std::to_string(rows_generation_);
    if (key != view_key_) {
        view_ = ig::Build(input_);
        view_key_ = key;
    }
    return view_;
}

// Picks `row` and turns to the page that shows it.
void ImageGalleryView::Pick(const ig::View& view, long long row) {
    if (row < 0) return;
    input_.selected = row;
    const auto it = std::find(view.order.begin(), view.order.end(), static_cast<size_t>(row));
    if (it != view.order.end()) input_.page = static_cast<size_t>(it - view.order.begin()) / std::max<size_t>(1, input_.per_page);
}

// Arrow keys move the pick (Up / Down a row in the grid), Enter opens it in
// the viewer, Esc goes back; only while Data Studio has focus and no field
// is being typed in. The gallery owns those keys then, so keyboard
// navigation (on in the Engine) does not also move its focus or press the
// widget it is on: ImGui's navigation reads only keys nobody owns, and an
// owner set this frame holds for the next.
void ImageGalleryView::HandleKeys(const ig::View& view) {
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) || ImGui::GetIO().WantTextInput) return;
    if (view.order.empty()) return;
    const ImGuiID owner = ImGui::GetID("##gallery_keys");
    for (const ImGuiKey key : {ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_UpArrow, ImGuiKey_DownArrow,
                               ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Escape})
        ImGui::SetKeyOwner(key, owner);
    const auto pressed = [&](ImGuiKey key) { return ImGui::IsKeyPressed(key, ImGuiInputFlags_Repeat, owner); };
    if (layout_ == Layout::Viewer && pressed(ImGuiKey_Escape)) {
        layout_ = before_viewer_;
        return;
    }
    if (view.selected >= 0 && layout_ != Layout::Viewer && (pressed(ImGuiKey_Enter) || pressed(ImGuiKey_KeypadEnter))) {
        before_viewer_ = layout_;
        layout_ = Layout::Viewer;
        return;
    }
    const auto at = view.selected < 0 ? view.order.end()
                                      : std::find(view.order.begin(), view.order.end(), static_cast<size_t>(view.selected));
    const long long index = at == view.order.end() ? -1 : static_cast<long long>(at - view.order.begin());
    const long long row_step = layout_ == Layout::Grid ? std::max(1, columns_) : 1;
    long long move = 0;
    if (pressed(ImGuiKey_RightArrow)) move = 1;
    if (pressed(ImGuiKey_LeftArrow)) move = -1;
    if (layout_ != Layout::Viewer && pressed(ImGuiKey_DownArrow)) move = row_step;
    if (layout_ != Layout::Viewer && pressed(ImGuiKey_UpArrow)) move = -row_step;
    if (move == 0) return;
    const long long last = static_cast<long long>(view.order.size()) - 1;
    const long long next = index < 0 ? 0 : std::clamp(index + move, 0LL, last);
    Pick(view, static_cast<long long>(view.order[static_cast<size_t>(next)]));
}

void ImageGalleryView::RenderToolbar(const ig::View& view) {
    const ui::Tokens& t = ui::CurrentTokens();
    static const char* kLayouts[] = {ICON_FA_TH " Grid", ICON_FA_LIST " List", ICON_FA_IMAGE " Viewer"};
    int layout = static_cast<int>(layout_);
    if (ui::SegmentedControl("##gallery_layout", kLayouts, 3, &layout) && layout != static_cast<int>(layout_)) {
        if (static_cast<Layout>(layout) == Layout::Viewer) {
            before_viewer_ = layout_;
            if (view.selected < 0 && !view.order.empty()) Pick(view, static_cast<long long>(view.order.front()));
        }
        layout_ = static_cast<Layout>(layout);
    }
    ImGui::SameLine(0.0f, 16.0f);
    if (layout_ == Layout::Grid) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Tile");
        ImGui::SameLine();
        ui::NumberField("##tile_size", &tile_, kMinTile, kMaxTile, kTileStep, 52.0f, "Tile size in pixels");
        ImGui::SameLine(0.0f, 12.0f);
        static const char* kFits[] = {"Fill", "Fit"};
        int fit = fill_ ? 0 : 1;
        if (ui::SegmentedControl("##tile_fit", kFits, 2, &fit)) fill_ = fit == 0;
        ImGui::SameLine(0.0f, 16.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Show");
        struct Toggle {
            const char* label;
            bool* on;
        };
        for (const Toggle& toggle : {Toggle{"Class##show_class", &show_class_}, Toggle{"Name##show_name", &show_name_},
                                     Toggle{"Size##show_size", &show_size_}, Toggle{"Flags##show_flags", &show_flags_}}) {
            ImGui::SameLine();
            const std::string id = toggle.label;
            const std::string label = id.substr(0, id.find("##"));
            if (ui::FilterChip(id.c_str(), label.c_str(), "", *toggle.on)) *toggle.on = !*toggle.on;
        }
    } else if (layout_ == Layout::Viewer) {
        if (ui::SecondaryButton(ICON_FA_CHEVRON_LEFT "##viewer_previous", view.previous >= 0, "This is the first image"))
            Pick(view, view.previous);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(view.detail_title.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "\xC2\xB7 %s", view.position.c_str());
        ImGui::SameLine();
        if (ui::SecondaryButton(ICON_FA_CHEVRON_RIGHT "##viewer_next", view.next >= 0, "This is the last image"))
            Pick(view, view.next);
        // The zoom controls, right aligned (they wrap below when narrow).
        const float zoom_w = 330.0f;
        if (ui::SameLineRight(zoom_w)) viewer_.RenderZoomControls();
    } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Click a column to sort by it; double-click a row to open it in the viewer.");
    }
}

void ImageGalleryView::RenderFilters(const ig::View& view, bool sort_buttons) {
    const ui::Tokens& t = ui::CurrentTokens();
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
    if (sort_buttons) {
        static const char* kSorts[] = {"Dataset order", "Smallest", "Largest", "Blurriest"};
        if (!view.quality && input_.sort == ig::Sort::Blurriest) input_.sort = ig::Sort::DatasetOrder;
        int sort = static_cast<int>(input_.sort);
        const int shown = view.quality ? 4 : 3;
        ImGui::SameLine(0.0f, 16.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Sort");
        ImGui::SameLine();
        int chosen = sort < shown ? sort : -1;  // a list column's sort shows as none here
        if (ui::SegmentedControl("##gallery_sort", kSorts, shown, &chosen)) {
            input_.sort = static_cast<ig::Sort>(chosen);
            input_.page = 0;
        }
    }
    const float pager_w = ImGui::CalcTextSize(view.showing.c_str()).x + 2.0f * ui::ButtonWidth(ICON_FA_CHEVRON_LEFT, ui::ButtonSize::Small) +
                          2.0f * ImGui::GetStyle().ItemSpacing.x;
    if (!ui::SameLineRight(pager_w)) ImGui::SameLine(0.0f, 16.0f);
    if (ui::SecondaryButton(ICON_FA_CHEVRON_LEFT "##previous", view.has_previous, "This is the first page", ui::ButtonSize::Small))
        input_.page = view.page - 1;
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_CHEVRON_RIGHT "##next", view.has_next, "This is the last page", ui::ButtonSize::Small))
        input_.page = view.page + 1;
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", view.showing.c_str());
}

void ImageGalleryView::RenderGrid(const ig::View& view, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::BeginChild("##gallery_grid", ImVec2(width, height), ImGuiChildFlags_None);
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float step = static_cast<float>(tile_) + gap;
    const float avail = ImGui::GetContentRegionAvail().x;
    columns_ = std::max(1, static_cast<int>((avail + gap) / step));
    // The tiles stretch to fill the row exactly.
    // At least a pixel: a zero-size button asserts in a very narrow panel.
    const float size = std::max(1.0f, std::floor((avail - gap * static_cast<float>(columns_ - 1)) / static_cast<float>(columns_)));
    for (size_t k = 0; k < view.shown.size(); ++k) {
        const size_t i = view.shown[k];
        if (k % static_cast<size_t>(columns_) != 0) ImGui::SameLine();
        ImGui::PushID(static_cast<int>(i));
        TileText text;
        text.tag = show_class_ ? &view.tags[k] : nullptr;
        text.name = show_name_ ? &input_.rows[i].file : nullptr;
        text.size = show_size_ ? &view.sizes[k] : nullptr;
        text.flag = show_flags_ ? &view.flags[k] : nullptr;
        bool opened = false;
        if (Tile(tiles_, input_.rows[i], size, fill_, text, static_cast<long long>(i) == view.selected, &opened))
            input_.selected = static_cast<long long>(i);
        if (opened) {
            input_.selected = static_cast<long long>(i);
            before_viewer_ = Layout::Grid;
            layout_ = Layout::Viewer;
        }
        ImGui::PopID();
    }
    if (view.shown.empty()) ImGui::TextColored(t.text_dim, "No image matches.");

    // What the model sees: folded under the grid.
    if (view.selected >= 0) {
        ImGui::Spacing();
        const ig::Row& row = input_.rows[static_cast<size_t>(view.selected)];
        const std::string fold = std::string(model_sees_open_ ? ICON_FA_CHEVRON_DOWN : ICON_FA_CHEVRON_RIGHT) +
                                 " What the model sees \xC2\xB7 " + row.file + "##model_sees_fold";
        if (ui::GhostButton(fold.c_str())) model_sees_open_ = !model_sees_open_;
        if (model_sees_open_) RenderPipeline(row);
    }
    ImGui::EndChild();
}

void ImageGalleryView::RenderList(const ig::View& view, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable;
    if (!ImGui::BeginTable("##gallery_list", 7, flags, ImVec2(width, height))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("##thumb", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, kListThumb);
    ImGui::TableSetupColumn("File", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_WidthStretch, 0.7f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_PreferSortDescending, 1.0f);
    ImGui::TableSetupColumn("Channels", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoSort, 0.8f);
    ImGui::TableSetupColumn("On disk", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_PreferSortDescending, 0.8f);
    ImGui::TableSetupColumn("Quality", ImGuiTableColumnFlags_WidthStretch, 1.2f);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsDirty) {
        const bool none = specs->SpecsCount == 0;
        const int column = none ? -1 : specs->Specs[0].ColumnIndex;
        const bool down = !none && specs->Specs[0].SortDirection == ImGuiSortDirection_Descending;
        if (none) input_.sort = ig::Sort::DatasetOrder;
        switch (column) {
            case 1: input_.sort = ig::Sort::Name; break;
            case 2: input_.sort = ig::Sort::Class; break;
            case 3: input_.sort = down ? ig::Sort::Largest : ig::Sort::Smallest; break;
            case 5: input_.sort = ig::Sort::LargestFile; break;
            case 6: input_.sort = view.quality ? ig::Sort::Blurriest : ig::Sort::DatasetOrder; break;
            default: break;
        }
        input_.page = 0;
        specs->SpecsDirty = false;
    }
    for (size_t k = 0; k < view.shown.size(); ++k) {
        const size_t i = view.shown[k];
        const ig::Row& row = input_.rows[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextRow(ImGuiTableRowFlags_None, kListThumb + 4.0f);
        ImGui::TableNextColumn();
        const bool selected = static_cast<long long>(i) == view.selected;
        if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick,
                              ImVec2(0.0f, kListThumb))) {
            input_.selected = static_cast<long long>(i);
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                before_viewer_ = Layout::List;
                layout_ = Layout::Viewer;
            }
        }
        ImGui::SameLine(0.0f, 0.0f);
        const auto thumb = tiles_.Get(row.path);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(kListThumb, kListThumb));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(at, ImVec2(at.x + kListThumb, at.y + kListThumb), ui::ToU32(t.bg_input), 4.0f);
        if (thumb.texture) {
            ImVec2 p0, p1, uv0, uv1;
            Placement(thumb, true, at, kListThumb, &p0, &p1, &uv0, &uv1);
            dl->AddImageRounded(thumb.texture, p0, p1, uv0, uv1, IM_COL32_WHITE, 4.0f);
        }
        const auto cell = [&](const std::string& text, const ImVec4& colour) {
            ImGui::TableNextColumn();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (kListThumb - ImGui::GetTextLineHeight()) * 0.5f);
            ImGui::TextColored(colour, "%s", text.c_str());
        };
        cell(row.file, t.text);
        cell(view.tags[k], t.text);
        cell(view.sizes[k], t.text);
        cell(view.channels[k], t.text_dim);
        cell(view.on_disk[k], t.text);
        cell(view.verdicts[k], view.verdicts[k] == "passes" || view.verdicts[k] == "not measured" ? t.text_dim : t.caution);
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void ImageGalleryView::RenderDetails(const ig::View& view, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::BeginGroup();
    if (view.selected < 0) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
        ImGui::TextColored(t.text_dim, "Click an image for its details. Arrow keys move the pick; Enter opens it in the viewer.");
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        return;
    }
    const ig::Row& row = input_.rows[static_cast<size_t>(view.selected)];
    const auto big = preview_.Get(row.path);
    if (big.texture) {
        const float scale = std::min(width / static_cast<float>(big.width), (width * 0.75f) / static_cast<float>(big.height));
        ImGui::Image(big.texture, ImVec2(big.width * scale, big.height * scale));
    } else {
        ImGui::Dummy(ImVec2(width, width * 0.6f));
    }
    ImGui::TextUnformatted(view.detail_title.c_str());
    if (ui::SameLineRight(ui::ButtonWidth("Open in viewer", ui::ButtonSize::Small)) &&
        ui::SecondaryButton("Open in viewer")) {
        before_viewer_ = layout_;
        layout_ = Layout::Viewer;
    }
    if (ImGui::BeginTable("##image_detail", 2, ImGuiTableFlags_SizingStretchProp, ImVec2(width, 0.0f))) {
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
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
    ImGui::TextColored(t.text_faint, "Arrow keys move the pick; Enter opens it in the viewer.");
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
}

void ImageGalleryView::RenderViewer(const ig::View& view) {
    if (view.selected < 0) {
        ImGui::TextColored(ui::CurrentTokens().text_dim, "No image matches the filter.");
        return;
    }
    const ig::Row& row = input_.rows[static_cast<size_t>(view.selected)];
    UpdatePipeline(row);
    const float strip_tile = 56.0f;
    const auto strip = [&] {
        for (size_t k = 0; k < view.strip.size(); ++k) {
            if (k > 0) ImGui::SameLine();
            const size_t i = view.strip[k];
            ImGui::PushID(static_cast<int>(i));
            if (Tile(tiles_, input_.rows[i], strip_tile, true, {}, static_cast<long long>(i) == view.selected))
                Pick(view, static_cast<long long>(i));
            ImGui::PopID();
        }
    };
    viewer_.Render(row.path, view.detail, ModelPicture(), strip_tile + 8.0f, strip);
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

    // The page fills the grid's height twice over (it scrolls), within the
    // thumbnail cache.
    const ImGuiStyle& style = ImGui::GetStyle();
    const bool side = ImGui::GetContentRegionAvail().x >= 760.0f;
    const float grid_w = side ? ImGui::GetContentRegionAvail().x - kDetailWidth - style.ItemSpacing.x * 2.0f
                              : ImGui::GetContentRegionAvail().x;
    if (layout_ == Layout::Grid) {
        const float step = static_cast<float>(tile_) + style.ItemSpacing.x;
        const size_t columns = static_cast<size_t>(std::max(1, static_cast<int>((grid_w + style.ItemSpacing.x) / step)));
        const size_t rows = static_cast<size_t>(std::max(1.0f, std::ceil(ImGui::GetContentRegionAvail().y / step))) * 2;
        input_.per_page = std::max(columns, std::min(kMaxPerPage / columns * columns, columns * rows));
    } else if (layout_ == Layout::List) {
        input_.per_page = 100;
    }
    const ig::View* view = &CurrentView();
    input_.filter = "all";
    for (const auto& chip : view->chips)
        if (chip.on) input_.filter = chip.key;
    input_.page = view->page;
    HandleKeys(*view);

    ui::BeginCard("gallery_toolbar");
    RenderToolbar(*view);
    ui::EndCard();
    view = &CurrentView();  // a pick or layout change this frame
    if (layout_ == Layout::Viewer) {
        RenderViewer(*view);
        return;
    }
    RenderFilters(*view, layout_ == Layout::Grid);
    ImGui::Spacing();

    const float height = ImGui::GetContentRegionAvail().y;
    if (layout_ == Layout::Grid) RenderGrid(*view, grid_w, height);
    else RenderList(*view, grid_w, height);
    view = &CurrentView();  // a click this frame
    if (side) {
        ImGui::SameLine(0.0f, style.ItemSpacing.x * 2.0f);
        ImGui::BeginChild("##gallery_details", ImVec2(kDetailWidth, height), ImGuiChildFlags_None);
        RenderDetails(*view, kDetailWidth - style.ScrollbarSize);
        ImGui::EndChild();
    } else {
        RenderDetails(*view, ImGui::GetContentRegionAvail().x);
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
    const float step = kStripTile + ImGui::GetStyle().ItemSpacing.x;
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
        TileText text;
        text.tag = &view.tags[k];
        text.flag = &view.flags[k];
        Tile(tiles_, input_.rows[view.shown[k]], kStripTile, true, text, false);
        ImGui::PopID();
    }
}

}  // namespace cyxwiz
