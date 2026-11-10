// Studio Debugger > Model > Layers > Unit weights (TOFIX178 6.4c, approved
// mockup board Dense): a Dense layer that reads the model's image flattened
// has one weight per pixel and colour, so each unit's weights are drawn as
// the image they read. The largest weight norm first (or unit order), 32 or
// all; a clicked unit shows large in the Inspector with its numbers and a
// way to its top images.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/dense_contributions.h"
#include "../../core/layer_tensor_reader.h"
#include "../../core/layer_view_presentation.h"
#include "../../core/texture_manager.h"
#include "../../core/unit_weights.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace lv = layer_view;
namespace uw = unit_weights;

namespace {

constexpr size_t kFirstUnits = 32;
constexpr float kTile = 56.0f;
constexpr float kTileGap = 10.0f;

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

std::string Fixed(double value, int digits) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", digits, value);
    return buffer;
}

// A drawn unit as RGBA: grey images on the diverging scale (blue lowers the
// unit, red raises it), colour images as their colours.
std::vector<unsigned char> Rgba(const uw::Picture& picture, const uw::ImageShape& image, const ui::Tokens& t) {
    const size_t pixels = image.height * image.width;
    std::vector<unsigned char> rgba(pixels * 4, 255);
    const auto byte = [](float v) { return static_cast<unsigned char>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    for (size_t p = 0; p < pixels; ++p) {
        if (image.channels == 3) {
            for (size_t ch = 0; ch < 3; ++ch) rgba[p * 4 + ch] = byte(picture.pixels[p * 3 + ch]);
        } else {
            const ImVec4 c = ui::DivergingAt(t, picture.pixels[p]);
            rgba[p * 4] = byte(c.x);
            rgba[p * 4 + 1] = byte(c.y);
            rgba[p * 4 + 2] = byte(c.z);
        }
    }
    return rgba;
}

}  // namespace

struct StudioDebuggerPanel::UnitWeightsState {
    bool by_norm = true;
    bool all = false;

    std::string key;  // capture / layer / order / all
    uw::ImageShape image;
    std::vector<float> norms;
    struct Tile {
        size_t unit = 0;
        float low = 0.0f;
        float high = 0.0f;
        uint32_t texture = 0;
    };
    std::vector<Tile> tiles;
    std::string error;
    std::optional<size_t> picked;  // a unit
    std::string picked_layer;      // the capture / layer it was picked in
    int shown_frame = -10;

    void DropTiles() {
        for (auto& tile : tiles)
            if (tile.texture) TextureManager::Instance().DeleteTexture(tile.texture);
        tiles.clear();
    }
    ~UnitWeightsState() { DropTiles(); }
};

std::vector<size_t> StudioDebuggerPanel::UnitWeightsImage(size_t layer_index) const {
    if (!layer_capture_) return {};
    const auto& layers = layer_capture_->Layers();
    if (layer_index >= layers.size() || layers.front().module_index != 0) return {};
    const CapturedTensor* model_input = layers.front().Find(CapturedTensorRole::Input);
    if (!model_input) return {};
    // A Flatten takes the image batch first: [N, H, W, C].
    const auto& shape = model_input->tensor.Shape();
    if (shape.size() != 4) return {};
    bool parameter_free = true;
    for (size_t i = 0; i < layer_index && parameter_free; ++i)
        for (const auto& tensor : layers[i].tensors)
            if (tensor.role == CapturedTensorRole::Parameter) parameter_free = false;
    const CapturedTensor* weight = nullptr;
    for (const auto& tensor : layers[layer_index].tensors)
        if (tensor.role == CapturedTensorRole::Parameter && tensor.tensor.Shape().size() == 2 && !weight)
            weight = &tensor;
    if (!weight) return {};
    const auto image = uw::ReadsImage(weight->tensor.Shape()[1], shape[1], shape[2], shape[3], parameter_free);
    if (!image) return {};
    return {image->height, image->width, image->channels};
}

void StudioDebuggerPanel::RenderUnitWeights(const std::string& weight_name) {
    if (!unit_weights_) unit_weights_ = std::make_shared<UnitWeightsState>();
    UnitWeightsState& s = *unit_weights_;
    s.shown_frame = ImGui::GetFrameCount();
    const CapturedLayer& layer = layer_capture_->Layers()[selected_layer_];
    const CapturedTensor* weight = layer.Find(CapturedTensorRole::Parameter, weight_name);
    const std::vector<size_t> image_shape = UnitWeightsImage(selected_layer_);
    if (!weight || image_shape.size() != 3) {
        Wrapped(DebuggerMuted(), "The run did not keep this layer's weights (Layer capture: keep parameters).");
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t units = weight->tensor.Shape()[0];
    const std::string layer_key =
        std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" + std::to_string(selected_layer_);
    if (s.picked_layer != layer_key) {
        s.picked.reset();
        s.picked_layer = layer_key;
    }
    const std::string key = layer_key + "/" + std::to_string(s.by_norm) + "/" + std::to_string(s.all);
    if (s.key != key) {
        s.key = key;
        s.error.clear();
        s.DropTiles();
        try {
            s.image = {image_shape[0], image_shape[1], image_shape[2]};
            s.norms = uw::RowNorms(weight->tensor);
            const auto order = uw::Order(s.norms, s.by_norm, s.all ? 0 : kFirstUnits);
            const auto rows = uw::Rows(weight->tensor, order);  // one read for every drawn unit
            const size_t per = s.image.Pixels();
            for (size_t i = 0; i < order.size(); ++i) {
                const uw::Picture picture = uw::Draw(rows.data() + i * per, s.image);
                const auto rgba = Rgba(picture, s.image, t);
                s.tiles.push_back({order[i], picture.low, picture.high,
                                   TextureManager::Instance().CreateTextureFromUint8Data(
                                       rgba.data(), static_cast<int>(s.image.width),
                                       static_cast<int>(s.image.height), 4)});
            }
        } catch (const std::exception& e) {
            s.error = e.what();
        }
    }
    if (!s.error.empty()) {
        Wrapped(DebuggerMuted(), "The unit weights could not be drawn: " + s.error + ".");
        return;
    }

    Wrapped(DebuggerMuted(), uw::Lead(s.image, units, s.tiles.size(), s.by_norm));
    ImGui::Spacing();

    // Order, how many, all or the first.
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), "Order");
    ImGui::SameLine();
    static const char* const kOrders[] = {"Largest weight norm first", "Unit order"};
    int order_choice = s.by_norm ? 0 : 1;
    if (ui::SegmentedControl("##UnitWeightsOrder", kOrders, 2, &order_choice)) s.by_norm = order_choice == 0;
    const std::string count = std::to_string(s.tiles.size()) + " of " + std::to_string(units) + " units";
    if (units > kFirstUnits) {
        const std::string toggle =
            s.all ? "First " + std::to_string(kFirstUnits) + " units" : "All " + std::to_string(units) + " units";
        SameLineIfFits(ImGui::CalcTextSize(count.c_str()).x + ui::ButtonWidth(toggle.c_str(), ui::ButtonSize::Small) +
                           24.0f,
                       18.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "%s", count.c_str());
        ImGui::SameLine();
        if (ui::SecondaryButton((toggle + "##UnitWeightsAll").c_str())) s.all = !s.all;
    } else {
        SameLineIfFits(ImGui::CalcTextSize(count.c_str()).x + 8.0f, 18.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "%s", count.c_str());
    }
    ImGui::Spacing();

    // The tiles, wrapped to the column; a unit's number under each.
    const float line = ImGui::GetTextLineHeight();
    const float avail = ImGui::GetContentRegionAvail().x;
    const size_t per_row = std::max<size_t>(1, static_cast<size_t>((avail + kTileGap) / (kTile + kTileGap)));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < s.tiles.size(); ++i) {
        const auto& tile = s.tiles[i];
        if (i % per_row != 0) ImGui::SameLine(0.0f, kTileGap);
        ImGui::PushID(static_cast<int>(tile.unit));
        ImGui::BeginGroup();
        const ImVec2 at = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##unit", ImVec2(kTile, kTile + 3.0f + line))) {
            s.picked = tile.unit;
            NoteClickedChannel(selected_layer_, tile.unit);
        }
        const bool hovered = ImGui::IsItemHovered();
        if (tile.texture)
            dl->AddImageRounded(static_cast<ImTextureID>(tile.texture), at, ImVec2(at.x + kTile, at.y + kTile),
                                ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 3.0f);
        // The picked unit keeps the accent ring (a line that means something).
        if (s.picked && *s.picked == tile.unit)
            dl->AddRect(ImVec2(at.x - 2.0f, at.y - 2.0f), ImVec2(at.x + kTile + 2.0f, at.y + kTile + 2.0f),
                        ui::ToU32(t.accent_text), 4.0f, 0, 2.0f);
        const std::string label = "unit " + std::to_string(tile.unit);
        const float label_w = ImGui::CalcTextSize(label.c_str()).x;
        dl->AddText(ImVec2(at.x + std::max(0.0f, (kTile - label_w) * 0.5f), at.y + kTile + 3.0f),
                    ui::ToU32(hovered ? DebuggerText() : DebuggerMuted()), label.c_str());
        ImGui::EndGroup();
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::BeginTooltip();
            ImGui::Text("Unit %zu", tile.unit);
            if (tile.unit < s.norms.size())
                ImGui::TextColored(DebuggerMuted(), "weight norm %s", Fixed(s.norms[tile.unit], 3).c_str());
            ImGui::TextColored(DebuggerMuted(), "weights %s to %s", contributions::Signed(tile.low).c_str(),
                               contributions::Signed(tile.high).c_str());
            ImGui::EndTooltip();
        }
        ImGui::PopID();
    }
    ImGui::Spacing();
    Wrapped(DebuggerFaint(), "Click a unit to see it large in the Inspector, with its numbers and its top images.");
}

bool StudioDebuggerPanel::RenderUnitWeightsPick() {
    if (!unit_weights_ || !unit_weights_->picked || ImGui::GetFrameCount() - unit_weights_->shown_frame > 1)
        return false;
    UnitWeightsState& s = *unit_weights_;
    const size_t unit = *s.picked;
    const auto tile = std::find_if(s.tiles.begin(), s.tiles.end(), [&](const auto& x) { return x.unit == unit; });
    const CapturedLayer& layer = layer_capture_->Layers()[selected_layer_];
    if (BeginCard("UnitWeightsPick")) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerMuted(), "Unit weights");
        if (tile != s.tiles.end() && tile->texture) {
            const float side = std::min(272.0f, ImGui::GetContentRegionAvail().x);
            ImGui::Image(static_cast<ImTextureID>(tile->texture),
                         ImVec2(side, side * static_cast<float>(s.image.height) /
                                          static_cast<float>(std::max<size_t>(1, s.image.width))));
        }
        ImGui::TextColored(DebuggerBright(), "Unit %zu of %s", unit, layer.node_name.c_str());
        if (unit < s.norms.size()) ImGui::Text("weight norm %s", Fixed(s.norms[unit], 3).c_str());
        if (tile != s.tiles.end())
            ImGui::TextColored(DebuggerMuted(), "weights %s to %s", contributions::Signed(tile->low).c_str(),
                               contributions::Signed(tile->high).c_str());
        try {
            if (const CapturedTensor* bias = layer.Find(CapturedTensorRole::Parameter, "bias")) {
                const auto b = lv::ReadValues(bias->tensor);
                if (unit < b.size()) ImGui::TextColored(DebuggerMuted(), "bias %s", contributions::Signed(b[unit]).c_str());
            }
            if (const CapturedTensor* output = layer.Find(CapturedTensorRole::Output)) {
                const auto y = lv::ReadValues(lv::SampleOf(output->tensor, selected_layer_sample_));
                if (unit < y.size()) ImGui::Text("this sample: %s", Fixed(y[unit], 4).c_str());
            }
        } catch (const std::exception&) {
            // The numbers are extra: the picture stands without them.
        }
        ImGui::Spacing();
        if (ui::SecondaryButton(("Top images of unit " + std::to_string(unit)).c_str())) {
            NoteClickedChannel(selected_layer_, unit);
            show_top_images_of_clicked_ = true;
        }
        ImGui::PopTextWrapPos();
    }
    EndCard();
    ImGui::Spacing();
    return true;
}

}  // namespace cyxwiz
