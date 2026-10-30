#include "image_viewer_view.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/async_task_manager.h"
#include "../../core/texture_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace {

namespace iv = image_viewer;

constexpr float kSideWidth = 290.0f;
constexpr int kAroundColumns = 12, kAroundRows = 8;
const char* kChannels[] = {"RGB", "R", "G", "B", "Luminance", "Alpha"};
const char* kCompare[] = {"Original", "Model sees", "Side by side"};

ImTextureID Texture(uint32_t id) { return static_cast<ImTextureID>(id); }

// The picture in `size` at (at), `texture` drawn with `uv` corners, masked by
// `mask` (a channel view keeps one of R, G, B).
void DrawQuad(ImDrawList* dl, ImTextureID texture, const ImVec2& min, const ImVec2& max,
              const std::array<iv::UV, 4>& uv, ImU32 mask) {
    dl->AddImageQuad(texture, min, ImVec2(max.x, min.y), max, ImVec2(min.x, max.y), ImVec2(uv[0].u, uv[0].v),
                     ImVec2(uv[1].u, uv[1].v), ImVec2(uv[2].u, uv[2].v), ImVec2(uv[3].u, uv[3].v), mask);
}

// A grey and lighter grey checkerboard in the window's tones, behind
// transparent pixels.
void DrawCheckerboard(ImDrawList* dl, const ImVec2& min, const ImVec2& max) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float cell = 8.0f;
    dl->AddRectFilled(min, max, ui::ToU32(t.bg_input));
    const ImVec2 clip_min = dl->GetClipRectMin(), clip_max = dl->GetClipRectMax();
    const float x0 = std::max(min.x, clip_min.x), y0 = std::max(min.y, clip_min.y);
    const float x1 = std::min(max.x, clip_max.x), y1 = std::min(max.y, clip_max.y);
    for (float y = min.y + std::floor((y0 - min.y) / cell) * cell; y < y1; y += cell)
        for (float x = min.x + std::floor((x0 - min.x) / cell) * cell; x < x1; x += cell) {
            const int k = static_cast<int>(std::floor((x - min.x) / cell) + std::floor((y - min.y) / cell));
            if (k % 2 == 0) continue;
            dl->AddRectFilled(ImVec2(x, y), ImVec2(std::min(x + cell, max.x), std::min(y + cell, max.y)), ui::ToU32(t.bg_raised));
        }
}

// The next `width` px on this row when it fits before `right_edge`, else
// on the next row: nothing runs off a narrow panel.
void Flow(float width, float right_edge, float gap) {
    ImGui::SameLine(0.0f, gap);
    if (ImGui::GetCursorScreenPos().x + width > right_edge) ImGui::NewLine();
}

std::string Stat(const ChannelHistogram& h) {
    char text[48];
    std::snprintf(text, sizeof(text), "mean %.1f \xC2\xB7 std %.1f", h.mean, h.std_dev);
    return text;
}

}  // namespace

ImageViewerView::ImageViewerView() = default;

ImageViewerView::~ImageViewerView() {
    if (task_) AsyncTaskManager::Instance().Cancel(task_);
    Release();
}

void ImageViewerView::Release() {
    auto& textures = TextureManager::Instance();
    for (uint32_t* id : {&colour_, &luminance_, &alpha_}) {
        if (*id) textures.DeleteTexture(*id);
        *id = 0;
    }
}

void ImageViewerView::Load(const std::string& path) {
    if (path == path_ || task_) return;
    path_ = path;
    loading_ = true;
    fitted_ = true;
    hovering_ = false;
    struct Result {
        ImagePixels pixels;
        ImageHistograms histograms;
        ImageDisplay display;
    };
    auto result = std::make_shared<Result>();
    std::weak_ptr<int> alive = alive_;
    const int max_side = TextureManager::Instance().MaxTextureSize();  // UI thread: needs the GL context
    task_ = AsyncTaskManager::Instance().RunAsync(
        "Viewer: " + path,
        [path, result, max_side](LambdaTask&) {
            result->pixels = DecodeImagePixels(path);
            result->histograms = ComputeHistograms(result->pixels);
            result->display = MakeImageDisplay(result->pixels, max_side);
        },
        nullptr,
        [this, alive, result](bool, const std::string&) {
            if (alive.expired()) return;
            task_ = 0;
            Release();
            pixels_ = std::move(result->pixels);
            histograms_ = result->histograms;
            display_ = std::move(result->display);
            loading_ = false;
            if (!pixels_.error.empty() || display_.colour.empty()) return;
            colour_ = TextureManager::Instance().CreateTextureFromUint8Data(display_.colour.data(), display_.width,
                                                                            display_.height, 4);
            display_.colour = {};
        },
        alive_);
}

void ImageViewerView::RenderZoomControls() {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "Zoom");
    ImGui::SameLine();
    if (ui::GhostButton("Fit##zoom_fit", true, nullptr, fitted_)) fitted_ = true;
    ImGui::SameLine();
    const bool actual = !fitted_ && std::fabs(view_.zoom - 1.0f) < 1e-4f;
    if (ui::GhostButton("100 %##zoom_actual", true, nullptr, actual)) {
        fitted_ = false;
        view_ = {1.0f, 0.0f, 0.0f};
    }
    ImGui::SameLine();
    zoom_percent_ = std::round(view_.zoom * 100.0f);
    ui::NumberFieldOptions options;
    options.min = iv::kMinZoom * 100.0;
    options.max = iv::kMaxZoom * 100.0;
    options.format = "%.0f %%";
    options.width = 70.0f;
    options.hint = "Zoom in percent";
    if (ui::NumberField("##zoom_percent", &zoom_percent_, options)) {
        fitted_ = false;
        view_ = iv::ZoomAt(view_, static_cast<float>(zoom_percent_ / 100.0), 0.0f, 0.0f);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("-##zoom_out")) {
        fitted_ = false;
        view_ = iv::ZoomAt(view_, iv::StepZoom(view_.zoom, -1), 0.0f, 0.0f);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("+##zoom_in")) {
        fitted_ = false;
        view_ = iv::ZoomAt(view_, iv::StepZoom(view_.zoom, 1), 0.0f, 0.0f);
    }
}

void ImageViewerView::RenderControls(bool has_model) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Channel");
    ImGui::SameLine();
    const int channels = pixels_.HasAlpha() ? 6 : 5;
    if (channel_ >= channels) channel_ = 0;
    ui::SegmentedControl("##viewer_channel", kChannels, channels, &channel_);
    Flow(ImGui::CalcTextSize("View").x + gap + 2.0f * ui::ButtonWidth(ICON_FA_ROTATE_LEFT, ui::ButtonSize::Small) + gap,
         right_edge, 16.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "View");
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_ROTATE_LEFT "##rotate_left")) {
        orientation_.quarter_turns = (orientation_.quarter_turns + 3) % 4;
        fitted_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotate the view left (the file is not changed)");
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_ROTATE_RIGHT "##rotate_right")) {
        orientation_.quarter_turns = (orientation_.quarter_turns + 1) % 4;
        fitted_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotate the view right (the file is not changed)");
    Flow(ui::ButtonWidth(ICON_FA_ARROWS_LEFT_RIGHT " Flip", ui::ButtonSize::Small), right_edge, gap);
    if (ui::GhostButton(ICON_FA_ARROWS_LEFT_RIGHT " Flip##flip", true, nullptr, orientation_.flip))
        orientation_.flip = !orientation_.flip;
    Flow(ui::ButtonWidth(ICON_FA_BORDER_ALL " Pixel grid", ui::ButtonSize::Small), right_edge, gap);
    if (ui::GhostButton(ICON_FA_BORDER_ALL " Pixel grid##grid", true, nullptr, pixel_grid_)) pixel_grid_ = !pixel_grid_;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Lines between pixels from 800 %%");
    Flow(ui::ButtonWidth("Checkerboard", ui::ButtonSize::Small), right_edge, gap);
    if (ui::GhostButton("Checkerboard##checker", true, nullptr, checkerboard_)) checkerboard_ = !checkerboard_;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shows behind transparent pixels");

    const float compare_w = ImGui::CalcTextSize("Compare").x + ImGui::GetStyle().ItemSpacing.x +
                            ui::SegmentedControlWidth(kCompare, 3);
    // Right aligned; on the next row when there is no room.
    ui::SameLineRight(compare_w);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Compare");
    ImGui::SameLine();
    if (!has_model) compare_ = 0;
    ImGui::BeginDisabled(!has_model);
    ui::SegmentedControl("##viewer_compare", kCompare, 3, &compare_);
    ImGui::EndDisabled();
    if (!has_model && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("The graph's image steps show here once it has compiled");
}

void ImageViewerView::RenderCanvas(const ImVec2& size, const ModelPicture& model) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    // At least a pixel: a zero-size button asserts in a very narrow panel.
    ImGui::InvisibleButton("##viewer_canvas", ImVec2(std::max(size.x, 1.0f), std::max(size.y, 1.0f)),
                           ImGuiButtonFlags_MouseButtonLeft);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 end(at.x + size.x, at.y + size.y);
    dl->AddRectFilled(at, end, ui::ToU32(t.bg_input), 6.0f);
    dl->PushClipRect(at, end, true);
    hovering_ = false;

    if (loading_ || !pixels_.error.empty() || !colour_) {
        const char* text = loading_ ? ICON_FA_SPINNER " Reading the image"
                           : !pixels_.error.empty() ? pixels_.error.c_str()
                                                    : "This image cannot be shown.";
        const ImVec2 ts = ImGui::CalcTextSize(text);
        dl->AddText(ImVec2(at.x + (size.x - ts.x) * 0.5f, at.y + (size.y - ts.y) * 0.5f),
                    ui::ToU32(loading_ ? t.running : t.caution), text);
        dl->PopClipRect();
        return;
    }

    // Model sees / side by side: each picture fitted in its half.
    const auto fitted_picture = [&](ImTextureID texture, int w, int h, ImVec2 min, ImVec2 max, const std::string& title) {
        const float area_w = max.x - min.x - 24.0f, area_h = max.y - min.y - 40.0f;
        const float zoom = iv::FitZoom({}, w, h, area_w, area_h);
        const ImVec2 shown(w * zoom, h * zoom);
        const ImVec2 p0(min.x + (max.x - min.x - shown.x) * 0.5f, min.y + 12.0f + (area_h - shown.y) * 0.5f);
        dl->AddImage(texture, p0, ImVec2(p0.x + shown.x, p0.y + shown.y));
        const std::string label = title + " \xC2\xB7 " + iv::ZoomText(zoom);
        const ImVec2 ls = ImGui::CalcTextSize(label.c_str());
        dl->AddText(ImVec2(min.x + (max.x - min.x - ls.x) * 0.5f, max.y - ls.y - 10.0f), ui::ToU32(t.text_dim), label.c_str());
    };
    if (compare_ != 0 && model.texture) {
        if (compare_ == 1) {
            fitted_picture(model.texture, model.width, model.height, at, end, model.title);
        } else {
            const float mid = at.x + size.x * 0.5f;
            fitted_picture(Texture(colour_), pixels_.width, pixels_.height, at, ImVec2(mid, end.y), "Original");
            fitted_picture(model.texture, model.width, model.height, ImVec2(mid, at.y), end, model.title);
        }
        dl->PopClipRect();
        return;
    }

    // Zoom and pan: the wheel zooms around the cursor, a drag pans, a
    // double-click goes between Fit and 100 %.
    const ImVec2 centre(at.x + size.x * 0.5f, at.y + size.y * 0.5f);
    const ImGuiIO& io = ImGui::GetIO();
    if (fitted_) view_ = {iv::FitZoom(orientation_, pixels_.width, pixels_.height, size.x - 24.0f, size.y - 24.0f), 0.0f, 0.0f};
    if (hovered && io.MouseWheel != 0.0f) {
        fitted_ = false;
        view_ = iv::ZoomAt(view_, iv::StepZoom(view_.zoom, io.MouseWheel > 0.0f ? 1 : -1), io.MousePos.x - centre.x,
                           io.MousePos.y - centre.y);
    }
    if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0f)) {
        fitted_ = false;
        view_.pan_x += io.MouseDelta.x;
        view_.pan_y += io.MouseDelta.y;
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        if (fitted_) {
            view_ = iv::ZoomAt(view_, 1.0f, io.MousePos.x - centre.x, io.MousePos.y - centre.y);
            fitted_ = false;
        } else {
            fitted_ = true;
        }
    }
    if (hovered) ImGui::SetMouseCursor(active ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_Hand);

    float shown_w = 0.0f, shown_h = 0.0f;
    iv::ShownSize(orientation_, pixels_.width, pixels_.height, &shown_w, &shown_h);
    shown_w *= view_.zoom;
    shown_h *= view_.zoom;
    const ImVec2 min(centre.x + view_.pan_x - shown_w * 0.5f, centre.y + view_.pan_y - shown_h * 0.5f);
    const ImVec2 max(min.x + shown_w, min.y + shown_h);
    if (checkerboard_) DrawCheckerboard(dl, min, max);

    // The luminance and alpha views are uploaded the first time they show.
    auto& textures = TextureManager::Instance();
    if (channel_ == 4 && !luminance_ && !display_.luminance.empty()) {
        luminance_ = textures.CreateTextureFromUint8Data(display_.luminance.data(), display_.width, display_.height, 4);
        display_.luminance = {};
    }
    if (channel_ == 5 && !alpha_ && !display_.alpha.empty()) {
        alpha_ = textures.CreateTextureFromUint8Data(display_.alpha.data(), display_.width, display_.height, 4);
        display_.alpha = {};
    }
    uint32_t texture = colour_;
    ImU32 mask = IM_COL32_WHITE;
    // A channel mask (not a theme colour): R, G or B alone.
    if (channel_ == 1) mask = IM_COL32(255, 0, 0, 255);
    else if (channel_ == 2) mask = IM_COL32(0, 255, 0, 255);
    else if (channel_ == 3) mask = IM_COL32(0, 0, 255, 255);
    else if (channel_ == 4 && luminance_) texture = luminance_;
    else if (channel_ == 5 && alpha_) texture = alpha_;
    DrawQuad(dl, Texture(texture), min, max, iv::CornerUVs(orientation_), mask);

    // The pixel grid, from 800 %, over the visible part only.
    if (pixel_grid_ && view_.zoom >= iv::kPixelGridZoom) {
        const ImU32 line = ui::ToU32(ui::WithAlpha(t.bg_window, 0.55f));
        const float step = view_.zoom;
        const float x0 = std::max(min.x, at.x), x1 = std::min(max.x, end.x);
        const float y0 = std::max(min.y, at.y), y1 = std::min(max.y, end.y);
        for (float x = min.x + std::ceil((x0 - min.x) / step) * step; x <= x1; x += step)
            dl->AddLine(ImVec2(x, y0), ImVec2(x, y1), line);
        for (float y = min.y + std::ceil((y0 - min.y) / step) * step; y <= y1; y += step)
            dl->AddLine(ImVec2(x0, y), ImVec2(x1, y), line);
    }

    // The pixel under the cursor, outlined once a pixel is big enough to see.
    if (hovered) {
        int px = 0, py = 0;
        if (iv::SourcePixel(orientation_, pixels_.width, pixels_.height, (io.MousePos.x - min.x) / view_.zoom,
                            (io.MousePos.y - min.y) / view_.zoom, &px, &py)) {
            hovering_ = true;
            hover_x_ = px;
            hover_y_ = py;
            if (view_.zoom >= 4.0f) {
                const float cx = min.x + std::floor((io.MousePos.x - min.x) / view_.zoom) * view_.zoom;
                const float cy = min.y + std::floor((io.MousePos.y - min.y) / view_.zoom) * view_.zoom;
                dl->AddRect(ImVec2(cx, cy), ImVec2(cx + view_.zoom, cy + view_.zoom), ui::ToU32(t.text_bright), 0.0f, 0, 2.0f);
            }
        }
    }
    const char* hint = "Drag to pan \xC2\xB7 wheel to zoom \xC2\xB7 double-click: Fit / 100 %";
    const ImVec2 hint_size = ImGui::CalcTextSize(hint);
    dl->AddRectFilled(ImVec2(at.x + 6.0f, at.y + 5.0f), ImVec2(at.x + 14.0f + hint_size.x, at.y + 11.0f + hint_size.y),
                      ui::ToU32(ui::WithAlpha(t.bg_window, 0.8f)), 4.0f);
    dl->AddText(ImVec2(at.x + 10.0f, at.y + 8.0f), ui::ToU32(t.text_dim), hint);
    dl->PopClipRect();
}

void ImageViewerView::RenderStatus() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_input);
    ImGui::BeginChild("##viewer_status", ImVec2(0.0f, ImGui::GetFrameHeight() + 4.0f), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleColor();
    if (hovering_ && !pixels_.rgba.empty()) {
        const uint8_t* p = pixels_.rgba.data() + (static_cast<size_t>(hover_y_) * pixels_.width + hover_x_) * 4;
        ImGui::Text("x %d  y %d", hover_x_, hover_y_);
        ImGui::SameLine(0.0f, 18.0f);
        if (pixels_.channels == 1 || pixels_.channels == 2) {
            ImGui::Text("Grey %d", p[0]);
        } else {
            ImGui::TextColored(t.channel[0], "R %d", p[0]);
            ImGui::SameLine();
            ImGui::TextColored(t.channel[1], "G %d", p[1]);
            ImGui::SameLine();
            ImGui::TextColored(t.channel[2], "B %d", p[2]);
        }
        if (pixels_.HasAlpha()) {
            ImGui::SameLine();
            ImGui::Text("A %d", p[3]);
        }
        ImGui::SameLine(0.0f, 18.0f);
        ImGui::TextUnformatted(iv::HexColour(p[0], p[1], p[2]).c_str());
    } else {
        ImGui::TextColored(t.text_dim, "Point at the image for a pixel's values.");
    }
    if (!pixels_.rgba.empty()) {
        char right[96];
        const char* kind = pixels_.channels == 1 ? "grey" : pixels_.channels == 2 ? "grey + alpha"
                         : pixels_.channels == 4 ? "RGBA" : "RGB";
        std::snprintf(right, sizeof(right), "%d \xC3\x97 %d \xC2\xB7 %s \xC2\xB7 %s \xC2\xB7 %s", pixels_.width, pixels_.height,
                      kind, pixels_.bits == 32 ? "32-bit float" : pixels_.bits == 16 ? "16 bit" : "8 bit",
                      iv::ZoomText(view_.zoom).c_str());
        ui::SameLineRight(ImGui::CalcTextSize(right).x);
        ImGui::TextColored(t.text_dim, "%s", right);
    }
    ImGui::EndChild();
}

void ImageViewerView::RenderSide(const std::vector<std::pair<std::string, std::string>>& detail) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;

    // Histograms: red, green, blue, luminance (and alpha); the chosen
    // channel's stays bright.
    ui::BeginCard("viewer_histogram");
    ImGui::TextUnformatted("Histogram");
    const char* range = "0 to 255, 32 bins";
    ui::SameLineRight(ImGui::CalcTextSize(range).x);
    ImGui::TextColored(t.text_dim, "%s", range);
    struct Row {
        const char* name;
        const ChannelHistogram* h;
        ImVec4 colour;
        int channel;  // the channel view it belongs to
    };
    std::vector<Row> rows = {{"Red", &histograms_.red, t.channel[0], 1},
                             {"Green", &histograms_.green, t.channel[1], 2},
                             {"Blue", &histograms_.blue, t.channel[2], 3},
                             {"Luminance", &histograms_.luminance, t.text, 4}};
    if (pixels_.channels == 1 || pixels_.channels == 2) rows = {{"Grey", &histograms_.luminance, t.text, 4}};
    if (pixels_.HasAlpha()) rows.push_back({"Alpha", &histograms_.alpha, t.text_dim, 5});
    ImDrawList* dl = ImGui::GetWindowDrawList();  // the card's own
    const float inner = ImGui::GetContentRegionAvail().x;
    for (const auto& row : rows) {
        const bool lit = channel_ == 0 || channel_ == row.channel;
        ImVec4 colour = row.colour;
        if (!lit) colour.w *= 0.35f;
        ImGui::TextColored(colour, "%s", row.name);
        const std::string stat = Stat(*row.h);
        ui::SameLineRight(ImGui::CalcTextSize(stat.c_str()).x);
        ImGui::TextColored(t.text_dim, "%s", stat.c_str());
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float h = 36.0f;
        ImGui::Dummy(ImVec2(inner, h));
        dl->AddRectFilled(at, ImVec2(at.x + inner, at.y + h), ui::ToU32(t.bg_input), 4.0f);
        const uint64_t peak = std::max<uint64_t>(1, *std::max_element(row.h->bins.begin(), row.h->bins.end()));
        const float bar = (inner - 6.0f) / kHistogramBins;
        for (int b = 0; b < kHistogramBins; ++b) {
            const float fill = (h - 6.0f) * static_cast<float>(row.h->bins[b]) / static_cast<float>(peak);
            const float x = at.x + 3.0f + b * bar;
            dl->AddRectFilled(ImVec2(x, at.y + h - 3.0f - std::max(fill, 1.0f)), ImVec2(x + bar - 1.0f, at.y + h - 3.0f),
                              ui::ToU32(colour));
        }
    }
    ui::EndCard();

    // The pixels around the cursor (the image centre when not pointing), one
    // cell each; the pointed pixel outlined.
    ui::BeginCard("viewer_around");
    ImGui::TextUnformatted("Around the cursor");
    if (!pixels_.rgba.empty()) {
        const int cx = hovering_ ? hover_x_ : pixels_.width / 2;
        const int cy = hovering_ ? hover_y_ : pixels_.height / 2;
        ImDrawList* cells = ImGui::GetWindowDrawList();  // the card's own
        const float cell = std::floor((ImGui::GetContentRegionAvail().x - 2.0f) / kAroundColumns);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(cell * kAroundColumns + 2.0f, cell * kAroundRows + 2.0f));
        cells->AddRectFilled(at, ImVec2(at.x + cell * kAroundColumns + 2.0f, at.y + cell * kAroundRows + 2.0f), ui::ToU32(t.bg_input));
        for (int r = 0; r < kAroundRows; ++r)
            for (int c = 0; c < kAroundColumns; ++c) {
                const int x = cx - kAroundColumns / 2 + c, y = cy - kAroundRows / 2 + r;
                if (x < 0 || y < 0 || x >= pixels_.width || y >= pixels_.height) continue;
                const uint8_t* p = pixels_.rgba.data() + (static_cast<size_t>(y) * pixels_.width + x) * 4;
                const ImVec2 p0(at.x + 1.0f + c * cell, at.y + 1.0f + r * cell);
                cells->AddRectFilled(p0, ImVec2(p0.x + cell - 1.0f, p0.y + cell - 1.0f), IM_COL32(p[0], p[1], p[2], 255));
                if (x == cx && y == cy)
                    cells->AddRect(p0, ImVec2(p0.x + cell - 1.0f, p0.y + cell - 1.0f), ui::ToU32(t.text_bright), 0.0f, 0, 2.0f);
            }
    }
    ui::EndCard();

    // The gallery's facts about the image.
    ui::BeginCard("viewer_detail");
    if (ImGui::BeginTable("##viewer_detail", 2, ImGuiTableFlags_SizingStretchProp, ImVec2(width - 2.0f * ImGui::GetStyle().WindowPadding.x, 0.0f))) {
        float label_w = 0.0f;
        for (const auto& item : detail) label_w = std::max(label_w, ImGui::CalcTextSize(item.first.c_str()).x);
        ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthFixed, label_w + 8.0f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
        for (const auto& [what, value] : detail) {
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
}

void ImageViewerView::Render(const std::string& path, const std::vector<std::pair<std::string, std::string>>& detail,
                             const ModelPicture& model, float strip_height, const std::function<void()>& strip) {
    Load(path);
    RenderControls(model.texture != 0);
    ImGui::Spacing();

    const ImGuiStyle& style = ImGui::GetStyle();
    const bool side = ImGui::GetContentRegionAvail().x >= 760.0f;
    const float left_w = std::max(1.0f, side ? ImGui::GetContentRegionAvail().x - kSideWidth - style.ItemSpacing.x * 2.0f
                                             : ImGui::GetContentRegionAvail().x);
    const float status_h = ImGui::GetFrameHeight() + 4.0f;
    // Beside the side panel when wide; above it when narrow, leaving it room.
    const float below = ImGui::GetContentRegionAvail().y - strip_height - status_h - style.ItemSpacing.y * 3.0f;
    const float canvas_h = std::max(200.0f, side ? below : below * 0.55f);

    ImGui::BeginGroup();
    RenderCanvas(ImVec2(left_w, canvas_h), model);
    if (strip) {
        ImGui::BeginChild("##viewer_strip", ImVec2(left_w, strip_height), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
        strip();
        ImGui::EndChild();
    }
    ImGui::BeginChild("##viewer_status_box", ImVec2(left_w, status_h), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    RenderStatus();
    ImGui::EndChild();
    ImGui::EndGroup();

    if (side) ImGui::SameLine(0.0f, style.ItemSpacing.x * 2.0f);
    // Below the picture when narrow: at least the histograms; the panel
    // scrolls to it when short.
    ImGui::BeginChild("##viewer_side", ImVec2(side ? kSideWidth : 0.0f,
                                              side ? canvas_h + strip_height + status_h + style.ItemSpacing.y * 2.0f
                                                   : std::max(ImGui::GetContentRegionAvail().y, 320.0f)),
                      ImGuiChildFlags_None);
    RenderSide(detail);
    ImGui::EndChild();
}

}  // namespace cyxwiz
