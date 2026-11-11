// Studio Debugger > Model > Layers (TOFIX178 6.2): every captured layer, its
// views picked by tensor shape (layer_view_presentation), the numbers read on
// the device and only what a view draws copied to the host
// (layer_tensor_reader), and the Inspector's sample and channel. The look
// follows the approved Stage 6 mockup (Main artboard).

#include "studio_debugger_panel.h"
#include "../../core/unit_weights.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/async_task_manager.h"
#include "../../core/debug_numerics.h"
#include "../../core/engine_config.h"
#include "../../core/layer_capture_settings.h"
#include "../../core/layer_capture_store.h"
#include "../../core/layer_tensor_reader.h"
#include "../../core/layer_view_presentation.h"
#include "../../core/texture_manager.h"
#include "../../core/training_snapshots_presentation.h"
#include "../../core/where_it_looks_presentation.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include "../ui_tokens.h"
#include "../icons.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>

namespace cyxwiz {

namespace lv = layer_view;

namespace {

constexpr size_t kStrongestShown = 16;
constexpr size_t kAllChannelsCap = 256;
constexpr size_t kHistogramBins = 32;
constexpr size_t kHeatmapCells = 262144;  // values per heatmap page
constexpr size_t kHeatmapRowsCap = 128;
constexpr float kLayerListWidth = 270.0f;

enum class Scale { Heat, Gray, Diverging };

// Normalised values ([0, 1], or [-1, 1] when diverging) as RGBA through the
// theme's scale tokens.
std::vector<unsigned char> Paint(const std::vector<float>& normalised, Scale scale) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<unsigned char> rgba(normalised.size() * 4);
    for (size_t i = 0; i < normalised.size(); ++i) {
        const ImVec4 c = scale == Scale::Heat ? ui::HeatAt(t, normalised[i])
                         : scale == Scale::Gray ? ui::GrayAt(t, normalised[i])
                                                : ui::DivergingAt(t, normalised[i]);
        rgba[i * 4 + 0] = static_cast<unsigned char>(std::lround(std::clamp(c.x, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 1] = static_cast<unsigned char>(std::lround(std::clamp(c.y, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 2] = static_cast<unsigned char>(std::lround(std::clamp(c.z, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

uint32_t Upload(const std::vector<unsigned char>& rgba, int width, int height) {
    if (rgba.empty() || width <= 0 || height <= 0) return 0;
    return TextureManager::Instance().CreateTextureFromUint8Data(rgba.data(), width, height, 4);
}

ImTextureID Texture(uint32_t id) { return static_cast<ImTextureID>(id); }

std::string Number(double value) {
    if (value == 0.0) value = 0.0;  // no "-0"
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4g", value);
    return buffer;
}

const CapturedTensor* FindTensor(const CapturedLayer& layer, CapturedTensorRole role, const std::string& name = {}) {
    return layer.Find(role, name);
}

// Histogram bars along the bottom of a box of `size`.
void DrawBars(const std::vector<float>& bars, ImVec2 size, const ImVec4& colour) {
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    if (bars.empty()) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float gap = 2.0f;
    const float width = std::max(1.0f, (size.x - gap * (bars.size() - 1)) / static_cast<float>(bars.size()));
    for (size_t i = 0; i < bars.size(); ++i) {
        const float h = bars[i] > 0.0f ? std::max(2.0f, bars[i] * size.y) : 0.0f;
        const float x = at.x + static_cast<float>(i) * (width + gap);
        dl->AddRectFilled(ImVec2(x, at.y + size.y - h), ImVec2(x + width, at.y + size.y), DebuggerU32(colour), 2.0f,
                          ImDrawFlags_RoundCornersTop);
    }
}

// A colour bar with its end labels.
void DrawLegend(Scale scale, const std::string& low, const std::string& high) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(DebuggerMuted(), "%s", low.c_str());
    ImGui::SameLine();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float width = 120.0f, height = 10.0f;
    const float y = at.y + (ImGui::GetTextLineHeight() - height) * 0.5f;
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    constexpr int kSteps = 48;
    for (int i = 0; i < kSteps; ++i) {
        const float v = static_cast<float>(i) / (kSteps - 1);
        const ImVec4 c = scale == Scale::Heat ? ui::HeatAt(t, v)
                         : scale == Scale::Gray ? ui::GrayAt(t, v)
                                                : ui::DivergingAt(t, v * 2.0f - 1.0f);
        const float x0 = at.x + width * i / kSteps, x1 = at.x + width * (i + 1) / kSteps;
        dl->AddRectFilled(ImVec2(x0, y), ImVec2(x1, y + height), ui::ToU32(c));
    }
    SameLineIfFits(ImGui::CalcTextSize(high.c_str()).x);
    ImGui::TextColored(DebuggerMuted(), "%s", high.c_str());
}

// What the capture ran on: the live run's input, else the part of a saved
// run's sample line after "mode |" ("images 3-10 of cat_dog, the first in train").
std::string SampleText(const StudioDebuggerSnapshot& session) {
    if (!session.debug_result.input_description.empty()) return session.debug_result.input_description;
    const size_t bar = session.sample_summary.rfind(" | ");
    return bar == std::string::npos ? std::string() : session.sample_summary.substr(bar + 3);
}

// "cat.10599.jpg · label cat · predicted cat, 99.0 %" for item `item` of the
// capture; the prediction only when `output` (the model's, for that item) is given.
std::string ItemHeadline(const StudioDebuggerSampleFacts& facts, size_t item, const std::vector<float>& output) {
    const std::string name = item < facts.item_names.size() ? facts.item_names[item] : std::string();
    const int label = item < facts.labels.size() ? facts.labels[item] : -1;
    std::optional<lv::Prediction> prediction;
    if (!output.empty()) prediction = lv::Predict(output, lv::OutputRuleFromKey(facts.output_rule), facts.class_names);
    return lv::SampleHeadline(name, label >= 0 ? lv::ClassName(label, facts.class_names) : std::string(), prediction);
}

void WrappedMuted(const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(DebuggerMuted(), "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

// What the shown view read, and its textures; rebuilt when the selection
// changes, never per frame.
struct StudioDebuggerPanel::LayerViewCache {
    const LayerCapture* capture = nullptr;
    std::vector<lv::LayerInfo> layers;
    std::vector<lv::LayerRow> rows;
    size_t first_spatial = std::numeric_limits<size_t>::max();
    size_t samples = 1;  // items in the captured batch

    std::string key;
    lv::LayerViews views;
    std::string error;
    DebugTensorNumericSummary numerics;
    std::vector<float> bars;
    std::vector<std::string> numbers;
    bool log_bars = true;

    // Feature maps.
    std::optional<lv::SpatialShape> spatial;
    lv::ChannelSummary channels;
    std::vector<size_t> order;
    float high = 0.0f;
    size_t dead = 0;
    std::vector<uint32_t> tiles;  // in `order`, the first tiles.size()

    // Units.
    std::vector<float> units;

    // The first layer's filters as pictures (their textures are `tiles`).
    lv::FilterPictures filters;

    // Weights and gradients.
    lv::TensorInfo matrix_info;
    std::optional<size_t> kernel_position;  // the shown slice of conv weights
    lv::MatrixLayout layout;
    size_t row_begin = 0;
    size_t row_count = 0;
    size_t shown_columns = 0;
    std::vector<float> matrix;
    uint32_t matrix_texture = 0;
    float limit = 0.0f;

    // The Inspector's sample picture and the model's output for it.
    std::string sample_key;
    uint32_t sample_texture = 0;
    int sample_width = 0;
    int sample_height = 0;
    // The whole sheet when the input is neither grey nor RGB: sample_texture
    // (its first tile) and the further channels' tiles, named.
    std::vector<ui::SheetTile> sample_sheet;
    std::vector<float> model_output;
    std::vector<lv::LayerSection> sections;  // the list's canvas-group sections

    void DropViewTextures() {
        auto& textures = TextureManager::Instance();
        for (uint32_t id : tiles)
            if (id) textures.DeleteTexture(id);
        tiles.clear();
        if (matrix_texture) textures.DeleteTexture(matrix_texture);
        matrix_texture = 0;
    }
    void DropSampleTexture() {
        if (sample_texture) TextureManager::Instance().DeleteTexture(sample_texture);
        for (size_t i = 1; i < sample_sheet.size(); ++i)  // tile 0 is sample_texture
            if (sample_sheet[i].texture) TextureManager::Instance().DeleteTexture(static_cast<uint32_t>(sample_sheet[i].texture));
        sample_sheet.clear();
        sample_texture = 0;
    }
    ~LayerViewCache() {
        DropViewTextures();
        DropSampleTexture();
    }
};

void StudioDebuggerPanel::RefreshLayerCaptureSource() {
    // A shown training run (6.3d) is read from the run store like a saved run.
    const std::string& run_id = shown_training_run_.empty() ? session_.run_id : shown_training_run_;
    const auto& live = session_.debug_result.layer_capture;
    if (live && shown_training_run_.empty()) {
        if (layer_capture_ != live) {
            layer_capture_ = live;
            layer_capture_run_id_ = run_id;
            layer_capture_label_ = kLocalDebugCaptureLabel;
            layer_capture_labels_ = {kLocalDebugCaptureLabel};
            layer_capture_listed_ = true;
            selected_layer_ = 0;
        }
        return;
    }
    if (layer_capture_run_id_ != run_id) {
        layer_capture_.reset();
        layer_capture_run_id_ = run_id;
        layer_capture_label_.clear();
        layer_capture_labels_.clear();
        layer_capture_listed_ = false;
        selected_layer_ = 0;
    }
    if (layer_capture_listed_ || layer_capture_loading_ || run_id.empty()) return;

    // A saved run keeps its capture in the run store: list it and read the
    // Local Debug one (else the latest snapshot) off the UI thread.
    layer_capture_listed_ = true;
    layer_capture_loading_ = true;
    struct Loaded {
        std::vector<std::string> labels;
        std::string label;
        std::shared_ptr<const LayerCapture> capture;
        std::string error;
    };
    auto loaded = std::make_shared<Loaded>();
    const std::string wanted = layer_capture_label_;
    AsyncTaskManager::Instance().RunAsync(
        "Studio Debugger: read layer capture",
        [loaded, run_id, wanted](LambdaTask&) {
            loaded->labels = ListLayerCaptures(run_id);
            if (loaded->labels.empty()) return;
            const auto has = [&](const std::string& l) {
                return std::find(loaded->labels.begin(), loaded->labels.end(), l) != loaded->labels.end();
            };
            loaded->label = !wanted.empty() && has(wanted) ? wanted
                            : has(kLocalDebugCaptureLabel) ? std::string(kLocalDebugCaptureLabel)
                                                           : loaded->labels.back();
            loaded->capture = LoadLayerCapture(run_id, loaded->label, &loaded->error);
        },
        nullptr,
        [this, loaded, run_id](bool, const std::string&) {
            layer_capture_loading_ = false;
            if (layer_capture_run_id_ != run_id) return;  // another run is shown now
            layer_capture_labels_ = loaded->labels;
            layer_capture_label_ = loaded->label;
            layer_capture_ = loaded->capture;
            selected_layer_ = 0;
            if (layer_capture_ && !layer_keep_node_.empty()) {
                const auto& layers = layer_capture_->Layers();
                for (size_t i = 0; i < layers.size(); ++i)
                    if (layers[i].node_name == layer_keep_node_) selected_layer_ = i;
            }
            layer_keep_node_.clear();
            if (!loaded->error.empty()) spdlog::warn("Studio Debugger: layer capture not read: {}", loaded->error);
        });
}

void StudioDebuggerPanel::ShowLayerSnapshot(const std::string& label) {
    if (label == layer_capture_label_ && layer_capture_) return;
    if (layer_capture_ && selected_layer_ < layer_capture_->Layers().size())
        layer_keep_node_ = layer_capture_->Layers()[selected_layer_].node_name;
    layer_capture_label_ = label;
    layer_capture_.reset();
    layer_capture_listed_ = false;
}

void StudioDebuggerPanel::RenderLayersView() {
    RefreshLayerCaptureSource();
    if (!layer_view_cache_) layer_view_cache_ = std::make_shared<LayerViewCache>();
    LayerViewCache& cache = *layer_view_cache_;

    if (!layer_capture_) {
        ImGui::PushTextWrapPos(0.0f);
        if (layer_capture_loading_) {
            ImGui::TextColored(DebuggerMuted(), ICON_FA_SPINNER "  Reading this run's layer capture...");
        } else {
            ImGui::TextColored(DebuggerBright(), "No layer capture for this run.");
            const bool capture_on = layer_capture::CapturesAnything(CaptureSettings(), false);
            ImGui::TextColored(DebuggerMuted(), "%s",
                               capture_on
                                   ? "Capture is on: this run did not reach Local Debug's forward and backward "
                                     "step (see Overview), or it ran before capture was turned on."
                                   : "Turn capture on with Layer capture in the toolbar (Local Debug runs, or "
                                     "Training every N epochs), then run Local Debug: every layer's input, "
                                     "output, weights and gradients for the picked sample show here.");
        }
        ImGui::PopTextWrapPos();
        return;
    }

    // The layer list follows the capture.
    if (cache.capture != layer_capture_.get()) {
        cache.DropViewTextures();
        cache.DropSampleTexture();
        cache.key.clear();
        cache.sample_key.clear();
        cache.capture = layer_capture_.get();
        cache.layers.clear();
        cache.rows.clear();
        cache.first_spatial = std::numeric_limits<size_t>::max();
        cache.samples = 1;
        for (const auto& layer : layer_capture_->Layers()) {
            std::string kind;
            for (const auto& trace : session_.traces) {
                if (trace.node_id == layer.node_id && !trace.node_type.empty()) {
                    kind = trace.node_type;
                    break;
                }
            }
            cache.layers.push_back(lv::DescribeCapturedLayer(layer, kind));
            cache.rows.push_back(lv::BuildLayerRow(cache.layers.back()));
            if (const auto* input = layer.Find(CapturedTensorRole::Input)) {
                const auto& s = input->tensor.Shape();
                if (cache.first_spatial == std::numeric_limits<size_t>::max() && lv::AsSpatial(s))
                    cache.first_spatial = cache.layers.size() - 1;
                if (cache.layers.size() == 1) cache.samples = s.size() == 4 ? s[3] : (s.empty() ? 1 : s[0]);
            }
        }
        std::map<int, std::string> group_of_node;
        for (const auto& group : session_.node_groups)
            for (int id : group.node_ids) group_of_node.emplace(id, group.name);
        cache.sections = lv::SectionLayers(cache.layers, group_of_node);
        selected_layer_ = std::min(selected_layer_, cache.layers.empty() ? 0 : cache.layers.size() - 1);
        selected_layer_sample_ = std::min(selected_layer_sample_, cache.samples - 1);
    }
    if (cache.layers.empty()) {
        ImGui::TextColored(DebuggerMuted(), "The capture holds no layers.");
        return;
    }

    // Source line: the picked item and the model's answer, what was
    // captured, from which step, and the batch item.
    ReadLayerSample();
    UpdateLayerModelSession();
    // The prediction comes from the evaluation pass (owner decision
    // 2026-11-09), the run's own forward only when the model cannot be rebuilt.
    const std::vector<float>* eval_output = EvalModelOutput();
    const bool eval_unavailable = !EvalModelUnavailable().empty();
    const std::vector<float> no_output;
    const std::vector<float>& shown_output = eval_output ? *eval_output : eval_unavailable ? cache.model_output : no_output;
    {
        std::string line = ItemHeadline(ShownSampleFacts(), selected_layer_sample_, shown_output);
        line += (line.empty() ? "" : " \xC2\xB7 ") + std::to_string(cache.layers.size()) + " layers captured";
        const std::string sample = shown_training_run_.empty() ? SampleText(session_) : shown_training_run_;
        if (!sample.empty()) line += " \xC2\xB7 " + sample;
        if (layer_capture_->Truncated())
            line += " \xC2\xB7 " + std::to_string(layer_capture_->SkippedTensors()) +
                    " tensors past the capture budget were left out";
        // The line wraps; the snapshot list and Item stepper go under it.
        WrappedMuted(line);
        bool controls = false;
        if (layer_capture_labels_.size() > 1) {
            // Snapshot < epoch 10 >: the layer, tab and channel stay.
            controls = true;
            const auto& labels = layer_capture_labels_;
            const auto found = std::find(labels.begin(), labels.end(), layer_capture_label_);
            const size_t index = found == labels.end() ? 0 : static_cast<size_t>(found - labels.begin());
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(DebuggerMuted(), "Snapshot");
            ImGui::SameLine();
            if (ui::SecondaryButton(ICON_FA_CHEVRON_LEFT "##LayerSnapshotPrev", index > 0, "The first snapshot"))
                ShowLayerSnapshot(labels[index - 1]);
            ImGui::SameLine(0.0f, 4.0f);
            const std::string shown = training_snapshots::SnapshotName(layer_capture_label_);
            ImGui::SetNextItemWidth(ImGui::CalcTextSize("epoch 0000").x + ImGui::GetFrameHeight() * 2.0f);
            if (ImGui::BeginCombo("##LayerCaptureLabel", shown.c_str())) {
                for (const auto& label : labels)
                    if (ImGui::Selectable(training_snapshots::SnapshotName(label).c_str(), label == layer_capture_label_))
                        ShowLayerSnapshot(label);
                ImGui::EndCombo();
            }
            ImGui::SameLine(0.0f, 4.0f);
            if (ui::SecondaryButton(ICON_FA_CHEVRON_RIGHT "##LayerSnapshotNext", index + 1 < labels.size(),
                                    "The last snapshot"))
                ShowLayerSnapshot(labels[index + 1]);
            int epoch = 0;
            std::sscanf(layer_capture_label_.c_str(), "epoch_%d", &epoch);
            const std::string note = training_snapshots::StepperText(index, labels.size(), epoch, ShownTrainingRecord());
            SameLineIfFits(ImGui::CalcTextSize(note.c_str()).x, 12.0f);
            ImGui::TextColored(DebuggerMuted(), "%s", note.c_str());

            // Compare with: None or another snapshot of the run.
            if (compare_label_ == layer_capture_label_ ||
                std::find(labels.begin(), labels.end(), compare_label_) == labels.end())
                compare_label_.clear();
            std::vector<std::string> pins = {"None"};
            std::vector<std::string> pin_labels = {""};
            for (const auto& label : labels) {
                if (label == layer_capture_label_) continue;
                pins.push_back(training_snapshots::SnapshotName(label));
                pin_labels.push_back(label);
            }
            const auto picked_pin = std::find(pin_labels.begin(), pin_labels.end(), compare_label_);
            int pin = static_cast<int>(picked_pin - pin_labels.begin());
            float pins_width = ImGui::CalcTextSize("Compare with").x + ImGui::GetStyle().ItemSpacing.x;
            for (const auto& p : pins) pins_width += ui::ButtonWidth(p.c_str(), ui::ButtonSize::Small);
            SameLineIfFits(pins.size() <= 6 ? pins_width : 260.0f, 24.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(DebuggerMuted(), "Compare with");
            ImGui::SameLine();
            if (pins.size() <= 6) {
                std::vector<const char*> texts;
                for (const auto& p : pins) texts.push_back(p.c_str());
                if (ui::SegmentedControl("LayerCompareWith", texts.data(), static_cast<int>(texts.size()), &pin))
                    compare_label_ = pin_labels[static_cast<size_t>(pin)];
            } else {
                ImGui::SetNextItemWidth(ImGui::CalcTextSize("epoch 0000").x + ImGui::GetFrameHeight() * 2.0f);
                if (ImGui::BeginCombo("##LayerCompareWith", pins[static_cast<size_t>(pin)].c_str())) {
                    for (size_t i = 0; i < pins.size(); ++i)
                        if (ImGui::Selectable(pins[i].c_str(), static_cast<int>(i) == pin)) compare_label_ = pin_labels[i];
                    ImGui::EndCombo();
                }
            }
            // The compared snapshot, read off the UI thread.
            const std::string want = compare_label_.empty() ? std::string() : layer_capture_run_id_ + "/" + compare_label_;
            if (want != compare_read_) {
                compare_read_ = want;
                compare_capture_.reset();
                if (!want.empty()) {
                    struct Read {
                        std::shared_ptr<const LayerCapture> capture;
                        std::string error;
                    };
                    auto read = std::make_shared<Read>();
                    AsyncTaskManager::Instance().RunAsync(
                        "Studio Debugger: read a snapshot to compare",
                        [read, run = layer_capture_run_id_, label = compare_label_](LambdaTask&) {
                            read->capture = LoadLayerCapture(run, label, &read->error);
                        },
                        nullptr,
                        [this, read, want](bool, const std::string&) {
                            if (compare_read_ != want) return;  // another snapshot is compared now
                            compare_capture_ = read->capture;
                            if (!read->error.empty())
                                spdlog::warn("Studio Debugger: the compared snapshot was not read: {}", read->error);
                        });
                }
            }
            if (!layer_capture_) return;
        } else {
            compare_label_.clear();
        }
        if (cache.samples > 1) {
            if (controls) ImGui::SameLine(0.0f, 16.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(DebuggerMuted(), "Item");
            ImGui::SameLine();
            if (ui::SecondaryButton(ICON_FA_MINUS "##LayerItemDown", selected_layer_sample_ > 0))
                --selected_layer_sample_;
            ImGui::SameLine();
            ImGui::Text("%zu of %zu", selected_layer_sample_ + 1, cache.samples);
            ImGui::SameLine();
            if (ui::SecondaryButton(ICON_FA_PLUS "##LayerItemUp", selected_layer_sample_ + 1 < cache.samples))
                ++selected_layer_sample_;
        }
    }
    ImGui::Spacing();

    const float list_width = std::min(kLayerListWidth, ImGui::GetContentRegionAvail().x * 0.4f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, DebuggerPanelBg());
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::BeginChild("StudioDebuggerLayerList", ImVec2(list_width, 0.0f), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::TextColored(DebuggerMuted(), "Layer");
    RightAlignOnLine(ImGui::CalcTextSize("Output per sample").x);
    ImGui::TextColored(DebuggerMuted(), "Output per sample");
    const float line = ImGui::GetTextLineHeight();
    for (const auto& section : cache.sections) {
        if (!section.name.empty()) {
            ImGui::Spacing();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(DebuggerFaint(), "%s", section.name.c_str());
            ImGui::PopTextWrapPos();
        }
        for (size_t i : section.layers) {
            const auto& row = cache.rows[i];
            ImGui::PushID(static_cast<int>(i));
            ui::PushListRowColours();
            if (ImGui::Selectable("##layer", selected_layer_ == i, 0,
                                  ImVec2(0.0f, row.kind.empty() ? line + 4.0f : line * 2.0f + 4.0f))) {
                if (selected_layer_ != i) {
                    selected_layer_ = i;
                    selected_layer_tab_ = 0;
                    selected_channel_rank_ = 0;
                    heatmap_row_begin_ = 0;
                    heatmap_kernel_position_ = -1;
                }
            }
            ui::PopListRowColours();
            // The row's words over the Selectable, drawn (not laid out) so the
            // cursor stays where the Selectable left it.
            const ImVec2 min = ImGui::GetItemRectMin();
            const ImVec2 max = ImGui::GetItemRectMax();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float output_width = ImGui::CalcTextSize(row.output.c_str()).x;
            dl->PushClipRect(min, max, true);
            dl->AddText(ImVec2(min.x + 4.0f, min.y + 2.0f), DebuggerU32(DebuggerText()), row.name.c_str());
            dl->AddText(ImVec2(max.x - output_width - 4.0f, min.y + 2.0f), DebuggerU32(DebuggerText()), row.output.c_str());
            if (!row.kind.empty())
                dl->AddText(ImVec2(min.x + 4.0f, min.y + 2.0f + line), DebuggerU32(DebuggerFaint()), row.kind.c_str());
            dl->PopClipRect();
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, DebuggerPanelBg());
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 12.0f));
    ImGui::BeginChild("StudioDebuggerLayerView", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AlwaysUseWindowPadding);

    const CapturedLayer& layer = layer_capture_->Layers()[selected_layer_];
    const lv::LayerInfo& info = cache.layers[selected_layer_];
    const CapturedTensor* output = FindTensor(layer, CapturedTensorRole::Output);

    const core::StudioDebuggerConfig settings = CaptureSettings();
    const int colour = settings.feature_map_colour == core::FeatureMapColour::Grayscale ? 1 : 0;
    const Scale map_scale = colour == 1 ? Scale::Gray : Scale::Heat;

    // The tabs need the output's zeros for the facts: plan with the sample's
    // numerics, read once per layer and item.
    const std::string layer_key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                                  std::to_string(selected_layer_) + "/" + std::to_string(selected_layer_sample_);
    if (cache.key.rfind(layer_key + "|", 0) != 0) {
        cache.numerics = {};
        cache.views = {};
        try {
            DebugTensorNumericSummary output_numerics;
            if (output) {
                output_numerics = ScanDebugTensorNumerics(lv::SampleOf(output->tensor, selected_layer_sample_),
                                                          gui::NodeType::Dense);
            }
            const CapturedTensor* model_input = layer_capture_->Layers().front().Find(CapturedTensorRole::Input);
            const auto picture = model_input ? lv::AsSpatial(model_input->tensor.Shape()) : std::nullopt;
            cache.views = lv::PlanLayerViews(info, output ? &output_numerics : nullptr,
                                             selected_layer_ == cache.first_spatial,
                                             selected_layer_ == 0 && layer.module_index == 0,
                                             picture.has_value(),
                                             !UnitWeightsImage(selected_layer_).empty());
        } catch (const std::exception& e) {
            cache.error = e.what();
        }
        cache.key = layer_key + "|";
    }
    const lv::LayerViews& views = cache.views;

    ImGui::PushStyleColor(ImGuiCol_Text, DebuggerBright());
    ImGui::TextUnformatted(cache.rows[selected_layer_].name.c_str());
    ImGui::PopStyleColor();
    if (!info.kind.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(DebuggerMuted(), "%s", info.kind.c_str());
    }
    if (!views.facts.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerText(), "%s", views.facts.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();

    if (views.tabs.empty()) {
        WrappedMuted("Nothing was captured for this layer (its tensors were past the capture budget).");
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        return;
    }
    if (show_top_images_of_clicked_) {
        bool found = false;
        for (size_t i = 0; i < views.tabs.size(); ++i)
            if (views.tabs[i].view == lv::View::TopImages) {
                selected_layer_tab_ = static_cast<int>(i);
                found = true;
            }
        if (!found) show_top_images_of_clicked_ = false;  // Top images takes it otherwise
    }
    selected_layer_tab_ = std::clamp(selected_layer_tab_, 0, static_cast<int>(views.tabs.size()) - 1);
    {
        std::vector<const char*> labels;
        for (const auto& tab : views.tabs) labels.push_back(tab.label.c_str());
        const int before = selected_layer_tab_;
        SubViewSelector("StudioDebuggerLayerTabs", labels.data(), static_cast<int>(labels.size()),
                        &selected_layer_tab_);
        if (before != selected_layer_tab_) heatmap_row_begin_ = 0;
        for (size_t i = 0; i < views.later.size(); ++i) {
            const auto& later = views.later[i];
            // The tabs draw nothing for a single view: the first later
            // button then starts the line.
            if (labels.size() > 1 || i > 0)
                SameLineIfFits(ui::ButtonWidth(later.label.c_str(), ui::ButtonSize::Small));
            ui::SecondaryButton(later.label.c_str(), false, later.why.c_str());
        }
    }
    ImGui::Spacing();

    const lv::ViewTab& tab = views.tabs[static_cast<size_t>(selected_layer_tab_)];
    const bool weight_view = tab.view == lv::View::Weights || tab.view == lv::View::Gradients;
    const bool parameter_view = weight_view || tab.view == lv::View::Filters;
    const size_t channel_cap = layer_all_channels_ ? kAllChannelsCap : kStrongestShown;
    const std::string view_key = layer_key + "|" + std::to_string(selected_layer_tab_) + "/" +
                                 std::to_string(colour) + "/" + std::to_string(static_cast<int>(settings.channel_sort)) +
                                 "/" + std::to_string(channel_cap) + "/" +
                                 std::to_string(heatmap_row_begin_) + "/" + std::to_string(heatmap_kernel_position_);

    // Reads `tab` of layer `l` into `c`: only what the view draws, from the
    // device. The compared snapshot (6.3d) reads the same channels in the
    // same order, the same rows and the same kernel slice as the shown one.
    struct ViewRead {
        size_t item = 0;
        const std::vector<size_t>* channels = nullptr;  // feature maps: these, in this order
        size_t row_begin = 0;
        int kernel_position = -1;
    };
    const auto read_view = [&](LayerViewCache& c, const CapturedLayer& l, const ViewRead& in) {
        c.DropViewTextures();
        c.error.clear();
        c.bars.clear();
        c.numbers.clear();
        c.spatial.reset();
        c.units.clear();
        c.matrix.clear();
        c.filters = {};
        const CapturedTensor* out = FindTensor(l, CapturedTensorRole::Output);
        try {
            const CapturedTensor* source = nullptr;
            Tensor shown;
            if (parameter_view) {
                source = FindTensor(l, tab.view == lv::View::Gradients ? CapturedTensorRole::Gradient
                                                                           : CapturedTensorRole::Parameter,
                                    tab.tensor);
                if (source) shown = source->tensor;
            } else if (out) {
                source = out;
                shown = lv::SampleOf(out->tensor, in.item);
            }
            if (!source) throw std::runtime_error("the tensor for this view was not captured");
            c.numerics = ScanDebugTensorNumerics(shown, gui::NodeType::Dense);
            c.log_bars = !parameter_view;
            c.numbers = lv::NumbersLines(c.numerics, !parameter_view);
            if (c.numerics.finite_count > 0) {
                const auto histogram = lv::ReadHistogram(shown, static_cast<float>(c.numerics.minimum),
                                                         static_cast<float>(c.numerics.maximum), kHistogramBins);
                c.bars = lv::HistogramBars(histogram.counts, c.log_bars);
            }

            if (tab.view == lv::View::FeatureMaps) {
                c.spatial = lv::AsSpatial(out->tensor.Shape());
                c.channels = lv::ReadChannelSummary(shown);
                c.order = in.channels ? *in.channels
                                      : lv::OrderChannels(c.channels.means, c.channels.maxima, settings.channel_sort);
                c.dead = lv::CountDeadChannels(c.channels.maxima);
                // The scale's top: the 99.5th percentile, so one bright pixel
                // does not darken every map.
                const float top = static_cast<float>(c.numerics.maximum);
                const auto fine = lv::ReadHistogram(shown, 0.0f, top > 0.0f ? top : 1.0f, 512);
                c.high = std::max(1e-6f, lv::HistogramQuantile(fine.counts, 0.0f, top > 0.0f ? top : 1.0f, 0.995));
                std::vector<size_t> picked(c.order.begin(),
                                           c.order.begin() + std::min(in.channels ? c.order.size() : channel_cap,
                                                                      c.order.size()));
                const auto pictures = lv::ReadChannelImages(shown, picked);
                const size_t pixels = c.spatial->height * c.spatial->width;
                for (size_t k = 0; k < picked.size(); ++k) {
                    const std::vector<float> one(pictures.begin() + static_cast<std::ptrdiff_t>(k * pixels),
                                                 pictures.begin() + static_cast<std::ptrdiff_t>((k + 1) * pixels));
                    c.tiles.push_back(Upload(Paint(lv::ToSequential(one, c.high), map_scale),
                                                 static_cast<int>(c.spatial->width),
                                                 static_cast<int>(c.spatial->height)));
                }
            } else if (tab.view == lv::View::Units) {
                c.units = lv::ReadValues(shown);
            } else if (tab.view == lv::View::Filters) {
                auto pictures = lv::BuildFilterPictures(lv::ReadValues(shown), shown.Shape());
                if (!pictures) throw std::runtime_error("these weights are not first-layer filters");
                c.filters = std::move(*pictures);
                const auto& f = c.filters;
                const size_t pixels = f.height * f.width;
                for (const auto& filter : f.filters) {
                    std::vector<unsigned char> rgba;
                    if (f.channels == 3) {
                        rgba.resize(pixels * 4);
                        for (size_t p = 0; p < pixels; ++p) {
                            for (size_t ch = 0; ch < 3; ++ch)
                                rgba[p * 4 + ch] = static_cast<unsigned char>(std::lround(filter.pixels[p * 3 + ch] * 255.0f));
                            rgba[p * 4 + 3] = 255;
                        }
                    } else {
                        rgba = Paint(filter.pixels, Scale::Gray);
                    }
                    c.tiles.push_back(Upload(rgba, static_cast<int>(f.width), static_cast<int>(f.height)));
                }
            } else if (weight_view) {
                c.matrix_info = {source->role, source->name, source->tensor.Shape()};
                const int kernel = in.kernel_position >= static_cast<int>(lv::KernelPositions(c.matrix_info))
                                       ? -1
                                       : in.kernel_position;
                c.kernel_position = kernel >= 0 ? std::optional<size_t>(static_cast<size_t>(kernel)) : std::nullopt;
                c.layout = lv::LayoutAsMatrix(c.matrix_info, c.kernel_position);
                const int max_texture = std::max(1, TextureManager::Instance().MaxTextureSize());
                c.shown_columns = std::min(c.layout.columns, static_cast<size_t>(max_texture));
                const size_t rows_fit = std::max<size_t>(1, kHeatmapCells / std::max<size_t>(1, c.layout.columns));
                c.row_count = std::min({c.layout.rows, kHeatmapRowsCap, rows_fit});
                c.row_begin = std::min(in.row_begin, c.layout.rows - std::min(c.layout.rows, c.row_count));
                c.matrix = lv::ReadMatrixRows(c.matrix_info, source->tensor, c.row_begin, c.row_count,
                                                  c.kernel_position);
                c.row_count = c.layout.columns ? c.matrix.size() / c.layout.columns : 0;
                c.limit = static_cast<float>(c.numerics.maximum_absolute);
                std::vector<float> visible;
                visible.reserve(c.row_count * c.shown_columns);
                for (size_t r = 0; r < c.row_count; ++r)
                    visible.insert(visible.end(), c.matrix.begin() + static_cast<std::ptrdiff_t>(r * c.layout.columns),
                                   c.matrix.begin() + static_cast<std::ptrdiff_t>(r * c.layout.columns + c.shown_columns));
                c.matrix_texture = Upload(Paint(lv::ToDiverging(visible, c.limit), Scale::Diverging),
                                              static_cast<int>(c.shown_columns), static_cast<int>(c.row_count));
            }
        } catch (const std::exception& e) {
            c.error = e.what();
        }
    };

    if (cache.key != view_key) {
        cache.key = view_key;
        read_view(cache, layer, {selected_layer_sample_, nullptr, heatmap_row_begin_, heatmap_kernel_position_});
        if (weight_view && cache.error.empty()) {
            heatmap_row_begin_ = cache.row_begin;
            heatmap_kernel_position_ = cache.kernel_position ? static_cast<int>(*cache.kernel_position) : -1;
        }
    }

    // Compare with another snapshot of the run (6.3d): the same view drawn
    // twice, side by side, the compared one without the controls. Where it
    // looks stays one snapshot at a time (it runs that snapshot's model).
    const bool comparing = compare_capture_ && !compare_label_.empty() && tab.view != lv::View::WhereItLooks &&
                           tab.view != lv::View::TopImages && tab.view != lv::View::Contributions &&
                           tab.view != lv::View::Embedding && tab.view != lv::View::UnitWeights;
    const CapturedLayer* compared = nullptr;
    if (comparing) {
        for (const auto& l : compare_capture_->Layers())
            if (l.node_name == layer.node_name) compared = &l;
    }
    if (compared && cache.error.empty()) {
        if (!compare_view_cache_) compare_view_cache_ = std::make_shared<LayerViewCache>();
        LayerViewCache& other = *compare_view_cache_;
        const std::string other_key = std::to_string(reinterpret_cast<uintptr_t>(compare_capture_.get())) + "|" +
                                      cache.key + "|" + std::to_string(cache.tiles.size());
        if (other.key != other_key) {
            other.key = other_key;
            const std::vector<size_t> channels(
                cache.order.begin(),
                cache.order.begin() + static_cast<std::ptrdiff_t>(std::min(cache.tiles.size(), cache.order.size())));
            read_view(other, *compared,
                      {selected_layer_sample_, &channels, cache.row_begin,
                       cache.kernel_position ? static_cast<int>(*cache.kernel_position) : -1});
        }
    }

    // The pictures each view draws, shared by both sides.
    const auto draw_maps = [&](LayerViewCache& c, bool clickable) {
        DrawLegend(map_scale, "0", "99.5th percentile (max " + Number(c.numerics.maximum) + ")");
        ImGui::Spacing();
        const float avail = ImGui::GetContentRegionAvail().x;
        const float gap = 10.0f;
        // A power of two per row that fits, so 16 or 32 maps fill whole rows.
        const int fit = std::max(1, static_cast<int>((avail + gap) / (96.0f + gap)));
        int columns = 1;
        while (columns * 2 <= fit) columns *= 2;
        const float tile = std::min(120.0f, (avail - gap * (columns - 1)) / columns);
        const float tile_h = tile * static_cast<float>(c.spatial->height) /
                             static_cast<float>(std::max<size_t>(1, c.spatial->width));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (size_t k = 0; k < c.tiles.size(); ++k) {
            if (k % static_cast<size_t>(columns) != 0) ImGui::SameLine(0.0f, gap);
            ImGui::PushID(static_cast<int>(k));
            const ImVec2 at = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##tile", ImVec2(tile, tile_h)) && clickable) {
                selected_channel_rank_ = k;
                NoteClickedChannel(selected_layer_, c.order[k]);
            }
            dl->AddImageRounded(Texture(c.tiles[k]), at, ImVec2(at.x + tile, at.y + tile_h), ImVec2(0, 0),
                                ImVec2(1, 1), IM_COL32_WHITE, 4.0f);
            if (k == selected_channel_rank_ || ImGui::IsItemHovered()) {
                dl->AddRect(ImVec2(at.x - 2, at.y - 2), ImVec2(at.x + tile + 2, at.y + tile_h + 2),
                            DebuggerU32(DebuggerAccentText(), k == selected_channel_rank_ ? 1.0f : 0.5f), 5.0f, 0,
                            2.0f);
            }
            if (ImGui::IsItemHovered()) {
                const size_t ch = c.order[k];
                ImGui::SetTooltip("Channel %zu \xC2\xB7 mean %s \xC2\xB7 max %s", ch,
                                  Number(c.channels.means[ch]).c_str(), Number(c.channels.maxima[ch]).c_str());
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
    };
    const auto draw_filters = [&](LayerViewCache& c) {
        const float avail = ImGui::GetContentRegionAvail().x;
        const float tile = 56.0f;
        const float gap = 10.0f;
        // Eight to a row (the mockup), fewer when the view is narrow.
        const int columns = std::clamp(static_cast<int>((avail + gap) / (tile + gap)), 1, 8);
        const float label_h = ImGui::GetTextLineHeight();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (size_t k = 0; k < c.tiles.size(); ++k) {
            if (k % static_cast<size_t>(columns) != 0) ImGui::SameLine(0.0f, gap);
            ImGui::PushID(static_cast<int>(k));
            const ImVec2 at = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("##filter", ImVec2(tile, tile + label_h + 3.0f));
            dl->AddImageRounded(Texture(c.tiles[k]), at, ImVec2(at.x + tile, at.y + tile), ImVec2(0, 0),
                                ImVec2(1, 1), IM_COL32_WHITE, 3.0f);
            const std::string norm = Number(c.filters.filters[k].norm);
            const float w = ImGui::CalcTextSize(norm.c_str()).x;
            dl->AddText(ImVec2(at.x + (tile - w) * 0.5f, at.y + tile + 3.0f), DebuggerU32(DebuggerMuted(), 1.0f),
                        norm.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Filter %zu \xC2\xB7 weight norm %s", k, norm.c_str());
            ImGui::PopID();
        }
        ImGui::Spacing();
    };
    const auto draw_units = [&](LayerViewCache& c) {
        const ImVec2 size(std::min(ImGui::GetContentRegionAvail().x, 720.0f), 150.0f);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##units", size) && c.units.size() > 0) {
            const float fx = (ImGui::GetIO().MousePos.x - at.x) / size.x;
            NoteClickedChannel(selected_layer_, std::min(c.units.size() - 1, static_cast<size_t>(std::max(0.0f, fx) *
                                                                                              static_cast<float>(c.units.size()))));
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float limit = static_cast<float>(std::max(1e-12, c.numerics.maximum_absolute));
        const float mid = at.y + size.y * 0.5f;
        dl->AddLine(ImVec2(at.x, mid), ImVec2(at.x + size.x, mid), DebuggerU32(DebuggerFaint(), 0.6f));
        const size_t n = c.units.size();
        const ui::Tokens& t = ui::CurrentTokens();
        for (size_t i = 0; i < n; ++i) {
            const float x0 = at.x + size.x * static_cast<float>(i) / static_cast<float>(n);
            const float x1 = std::max(x0 + 1.0f, at.x + size.x * static_cast<float>(i + 1) / static_cast<float>(n) - (n < 128 ? 1.0f : 0.0f));
            const float v = std::isfinite(c.units[i]) ? c.units[i] / limit : 0.0f;
            const float h = v * size.y * 0.5f;
            dl->AddRectFilled(ImVec2(x0, std::min(mid, mid - h)), ImVec2(x1, std::max(mid, mid - h)),
                              ui::ToU32(ui::DivergingAt(t, v >= 0.0f ? 1.0f : -1.0f)));
        }
        if (ImGui::IsItemHovered() && n > 0) {
            const float fx = (ImGui::GetIO().MousePos.x - at.x) / size.x;
            const size_t i = std::min(n - 1, static_cast<size_t>(std::max(0.0f, fx) * static_cast<float>(n)));
            ImGui::SetTooltip("Unit %zu = %s", i, Number(c.units[i]).c_str());
        }
        ImGui::Spacing();
    };
    const auto draw_values = [&](LayerViewCache& c) {
        DrawBars(c.bars, ImVec2(std::min(ImGui::GetContentRegionAvail().x, 480.0f), 140.0f), DebuggerAccent());
        if (c.numerics.finite_count > 0) {
            ImGui::TextColored(DebuggerFaint(), "%s", Number(c.numerics.minimum).c_str());
            ImGui::SameLine(std::min(ImGui::GetContentRegionAvail().x, 480.0f) -
                            ImGui::CalcTextSize(Number(c.numerics.maximum).c_str()).x);
            ImGui::TextColored(DebuggerFaint(), "%s", Number(c.numerics.maximum).c_str());
        }
        ImGui::Spacing();
    };
    const auto draw_heatmap = [&](LayerViewCache& c) {
        const float width = std::min(ImGui::GetContentRegionAvail().x, 900.0f);
        const float cell_h = std::clamp(width / static_cast<float>(std::max<size_t>(1, c.shown_columns)), 3.0f, 24.0f);
        const float height = std::clamp(cell_h * static_cast<float>(c.row_count), 24.0f, 260.0f);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##heatmap", ImVec2(width, height));
        if (c.matrix_texture) {
            ImGui::GetWindowDrawList()->AddImageRounded(Texture(c.matrix_texture), at,
                                                        ImVec2(at.x + width, at.y + height), ImVec2(0, 0),
                                                        ImVec2(1, 1), IM_COL32_WHITE, 4.0f);
        }
        if (ImGui::IsItemHovered() && c.row_count > 0 && c.shown_columns > 0) {
            const ImVec2 m = ImGui::GetIO().MousePos;
            const size_t col = std::min(c.shown_columns - 1,
                                        static_cast<size_t>(std::max(0.0f, (m.x - at.x) / width) * c.shown_columns));
            const size_t row = std::min(c.row_count - 1,
                                        static_cast<size_t>(std::max(0.0f, (m.y - at.y) / height) * c.row_count));
            const float value = c.matrix[row * c.layout.columns + col];
            const size_t flat = lv::MatrixFlatIndex(c.matrix_info, c.row_begin + row, col, c.kernel_position);
            ImGui::SetTooltip("%s", lv::ElementText(c.matrix_info, flat, value).c_str());
        }
        DrawLegend(Scale::Diverging, "-" + Number(c.limit), "+" + Number(c.limit));
    };
    const auto draw_numbers = [](const LayerViewCache& c) {
        for (const auto& l : c.numbers) ImGui::TextColored(DebuggerText(), "%s", l.c_str());
    };

    // A view in three parts: what it shows and its controls (once, above
    // both sides when comparing), its pictures, and its numbers (per side).
    const auto header = [&]() {
        if (tab.view == lv::View::FeatureMaps && cache.spatial) {
            const size_t count = cache.spatial->channels;
            WrappedMuted(lv::ChannelsShownText(std::min(count, layer_all_channels_ ? kAllChannelsCap : kStrongestShown),
                                               count, layer_all_channels_, settings.channel_sort) +
                         " Click one to inspect it.");
            static const char* const kScales[] = {"Heat", "Grayscale"};
            int pick = colour;
            ImGui::TextColored(DebuggerMuted(), "Colour");
            ImGui::SameLine();
            if (ui::SegmentedControl("StudioDebuggerMapColour", kScales, 2, &pick) && pick != colour) {
                if (!capture_session_) capture_session_ = core::EngineConfig::Instance().GetStudioDebugger();
                capture_session_->feature_map_colour = static_cast<core::FeatureMapColour>(pick);
            }
            static const char* const kSorts[] = {"Strongest first", "Highest peak first", "Channel order"};
            const float sort_width = ImGui::CalcTextSize(kSorts[1]).x + ImGui::GetFrameHeight() * 2.0f;
            SameLineIfFits(ImGui::CalcTextSize("Order").x + ImGui::GetStyle().ItemSpacing.x + sort_width, 16.0f);
            ImGui::TextColored(DebuggerMuted(), "Order");
            ImGui::SameLine();
            int sort = static_cast<int>(settings.channel_sort);
            ImGui::SetNextItemWidth(sort_width);
            if (ImGui::Combo("##StudioDebuggerChannelSort", &sort, kSorts, 3) &&
                sort != static_cast<int>(settings.channel_sort)) {
                if (!capture_session_) capture_session_ = core::EngineConfig::Instance().GetStudioDebugger();
                capture_session_->channel_sort = static_cast<core::ChannelSort>(sort);
                selected_channel_rank_ = 0;
            }
            if (count > kStrongestShown) {
                const std::string toggle = layer_all_channels_
                    ? (settings.channel_sort == core::ChannelSort::Strongest ? "The 16 strongest" : "The first 16")
                    : "All " + std::to_string(count) + " channels";
                SameLineIfFits(ui::ButtonWidth(toggle.c_str(), ui::ButtonSize::Regular), 16.0f);
                if (ui::SecondaryButton(toggle.c_str())) {
                    layer_all_channels_ = !layer_all_channels_;
                    selected_channel_rank_ = 0;
                }
            }
            if (comparing) WrappedMuted("Both sides show the same channels, in the shown snapshot's order.");
        } else if (tab.view == lv::View::Filters) {
            WrappedMuted(lv::FiltersText(cache.filters) +
                         " The number under each filter is its weight norm (the size of its weights).");
        } else if (tab.view == lv::View::Units) {
            WrappedMuted("Each unit's value for this sample (" + std::to_string(cache.units.size()) + " units).");
        } else if (tab.view == lv::View::Values) {
            WrappedMuted("Every value of this layer's output for the sample (log scale counts).");
        } else if (weight_view) {
            WrappedMuted((tab.view == lv::View::Gradients ? "Gradients of " : "") + cache.matrix_info.name + " " +
                         lv::ShapeText(cache.matrix_info.shape) + " shown as " + cache.layout.description +
                         ". Hover gives the value and its index." +
                         (comparing ? " Both sides show the same rows and slice, each on its own scale." : ""));
            // Conv weights: all kernel positions side by side, or one
            // position's output-by-input slice.
            if (const size_t positions = lv::KernelPositions(cache.matrix_info)) {
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(DebuggerMuted(), "Slice");
                ImGui::SameLine();
                const std::string preview = heatmap_kernel_position_ < 0
                    ? std::string("All kernel positions")
                    : lv::KernelPositionText(cache.matrix_info, static_cast<size_t>(heatmap_kernel_position_));
                ImGui::SetNextItemWidth(ImGui::CalcTextSize(preview.c_str()).x + ImGui::GetFrameHeight() * 2.0f);
                if (ImGui::BeginCombo("##StudioDebuggerKernelSlice", preview.c_str())) {
                    if (ImGui::Selectable("All kernel positions", heatmap_kernel_position_ < 0)) {
                        heatmap_kernel_position_ = -1;
                        heatmap_row_begin_ = 0;
                    }
                    for (size_t k = 0; k < positions; ++k) {
                        const std::string text = lv::KernelPositionText(cache.matrix_info, k);
                        if (ImGui::Selectable(text.c_str(), heatmap_kernel_position_ == static_cast<int>(k))) {
                            heatmap_kernel_position_ = static_cast<int>(k);
                            heatmap_row_begin_ = 0;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine(0.0f, 16.0f);
            }
            if (cache.layout.rows > cache.row_count) {
                if (ui::SecondaryButton(ICON_FA_MINUS "##RowsUp", cache.row_begin > 0))
                    heatmap_row_begin_ = cache.row_begin - std::min(cache.row_begin, cache.row_count);
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                ImGui::Text("Rows %zu-%zu of %zu", cache.row_begin + 1, cache.row_begin + cache.row_count,
                            cache.layout.rows);
                ImGui::SameLine();
                if (ui::SecondaryButton(ICON_FA_PLUS "##RowsDown", cache.row_begin + cache.row_count < cache.layout.rows))
                    heatmap_row_begin_ = cache.row_begin + cache.row_count;
            } else if (lv::KernelPositions(cache.matrix_info)) {
                ImGui::NewLine();
            }
            if (cache.shown_columns < cache.layout.columns) {
                WrappedMuted("The first " + std::to_string(cache.shown_columns) + " columns (this device's largest "
                             "picture).");
            }
        }
        ImGui::Spacing();
    };
    const auto pictures = [&](LayerViewCache& c, bool shown) {
        if (tab.view == lv::View::FeatureMaps && c.spatial) {
            draw_maps(c, shown);
            ImGui::TextColored(DebuggerMuted(), "%zu dead channels (zero everywhere)", c.dead);
        } else if (tab.view == lv::View::Filters && !c.filters.filters.empty()) {
            draw_filters(c);
        } else if (tab.view == lv::View::Units) {
            draw_units(c);
        } else if (tab.view == lv::View::Values) {
            draw_values(c);
        } else if (weight_view) {
            draw_heatmap(c);
            ImGui::Spacing();
            DrawBars(c.bars, ImVec2(std::min(ImGui::GetContentRegionAvail().x, 360.0f), 90.0f), DebuggerAccentText());
        }
        draw_numbers(c);
    };

    if (!cache.error.empty()) {
        WrappedMuted("This view could not be read: " + cache.error);
    } else if (tab.view == lv::View::WhereItLooks) {
        RenderWhereItLooks(cache.sample_texture, cache.sample_width, cache.sample_height);
    } else if (tab.view == lv::View::TopImages) {
        RenderTopImages();
    } else if (tab.view == lv::View::Contributions) {
        RenderContributions(tab.tensor);
    } else if (tab.view == lv::View::Embedding) {
        RenderEmbedding();
    } else if (tab.view == lv::View::UnitWeights) {
        RenderUnitWeights(tab.tensor);
    } else {
        header();
        if (comparing && ImGui::BeginTable("##LayerCompare", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            ImGui::TextColored(DebuggerBright(), "%s",
                               training_snapshots::SnapshotTitle(layer_capture_label_, ShownTrainingRecord()).c_str());
            pictures(cache, true);
            ImGui::TableNextColumn();
            ImGui::PushID("compared");
            ImGui::TextColored(DebuggerBright(), "%s",
                               training_snapshots::SnapshotTitle(compare_label_, ShownTrainingRecord()).c_str());
            LayerViewCache* other = compare_view_cache_.get();
            if (!compared)
                WrappedMuted("This snapshot did not keep " + layer.node_name + ".");
            else if (!other || other->key.empty())
                WrappedMuted("Reading the snapshot...");
            else if (!other->error.empty())
                WrappedMuted("This view could not be read: " + other->error);
            else
                pictures(*other, false);
            ImGui::PopID();
            ImGui::EndTable();
        } else {
            pictures(cache, true);
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void StudioDebuggerPanel::ReadLayerSample() {
    if (!layer_capture_ || !layer_view_cache_ || layer_capture_->Layers().empty()) return;
    LayerViewCache& cache = *layer_view_cache_;
    const auto& layers = layer_capture_->Layers();

    // The sample as the model saw it (after the data steps), and the model's
    // output for it, read once per item.
    const std::string sample_key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                                   std::to_string(selected_layer_sample_);
    if (cache.sample_key != sample_key) {
        cache.DropSampleTexture();
        cache.model_output.clear();
        cache.sample_key = sample_key;
        try {
            if (const auto* input = layers.front().Find(CapturedTensorRole::Input)) {
                if (lv::AsSpatial(input->tensor.Shape())) {
                    const Tensor sample = lv::SampleOf(input->tensor, selected_layer_sample_);
                    const ImageSheet sheet =
                        lv::SampleSheet(lv::ReadValues(sample), sample.Shape(), layer_capture_->Options().input_channels);
                    if (!sheet.tiles.empty()) {
                        const ImageSheetTile& picture = sheet.tiles.front();
                        cache.sample_texture = Upload(picture.rgba, static_cast<int>(picture.width),
                                                      static_cast<int>(picture.height));
                        cache.sample_width = static_cast<int>(picture.width);
                        cache.sample_height = static_cast<int>(picture.height);
                        if (sheet.tiles.size() > 1) {
                            cache.sample_sheet.push_back({Texture(cache.sample_texture), picture.name});
                            for (size_t i = 1; i < sheet.tiles.size(); ++i)
                                cache.sample_sheet.push_back(
                                    {Texture(Upload(sheet.tiles[i].rgba, static_cast<int>(sheet.tiles[i].width),
                                                    static_cast<int>(sheet.tiles[i].height))),
                                     sheet.tiles[i].name});
                        }
                    }
                }
            }
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                if (const auto* out = it->Find(CapturedTensorRole::Output)) {
                    cache.model_output = lv::ReadValues(lv::SampleOf(out->tensor, selected_layer_sample_));
                    break;
                }
            }
        } catch (const std::exception& e) {
            spdlog::debug("Studio Debugger: layer inspector sample not read: {}", e.what());
        }
    }
}

bool StudioDebuggerPanel::RenderLayersInspector() {
    if (!layer_capture_ || !layer_view_cache_ || layer_view_cache_->layers.empty()) return false;
    LayerViewCache& cache = *layer_view_cache_;
    ReadLayerSample();

    ImGui::BeginChild("StudioDebuggerLayerInspector", ImVec2(0.0f, 0.0f), false);
    RenderTopImagesPick();
    RenderEmbeddingPick();
    RenderUnitWeightsPick();
    if (BeginCard("StudioDebuggerLayerSample")) {
        // A card is its own window: wrap inside it.
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerMuted(), "Sample");
        if (cache.sample_texture && cache.sample_sheet.empty()) {
            const float side = std::min(160.0f, ImGui::GetContentRegionAvail().x);
            const float h = side * static_cast<float>(cache.sample_height) / static_cast<float>(std::max(1, cache.sample_width));
            ImGui::Image(Texture(cache.sample_texture), ImVec2(side, h));
        } else if (cache.sample_texture) {
            // A sheet: two tiles a row, smaller from 8 channels on; they wrap, nothing is clipped.
            const float avail = ImGui::GetContentRegionAvail().x;
            const float side = cache.sample_sheet.size() > 4 ? 88.0f : std::min(160.0f, (avail - 10.0f) * 0.5f);
            ui::SheetTiles(cache.sample_sheet, std::max(48.0f, side),
                           static_cast<float>(cache.sample_height) / static_cast<float>(std::max(1, cache.sample_width)));
        }
        const auto& facts = ShownSampleFacts();
        const std::string item = ItemHeadline(facts, selected_layer_sample_, {});
        if (!item.empty()) ImGui::TextUnformatted(item.c_str());
        const std::string sample = shown_training_run_.empty()
                                       ? SampleText(session_)
                                       : "From " + shown_training_run_ + ", " + training_snapshots::SnapshotName(layer_capture_label_);
        if (!sample.empty()) ImGui::TextColored(DebuggerMuted(), "%s", sample.c_str());
        if (cache.samples > 1) ImGui::TextColored(DebuggerMuted(), "Item %zu of %zu in the captured batch",
                                                  selected_layer_sample_ + 1, cache.samples);
        // The evaluation pass (dropout off, BatchNorm on its running
        // statistics), as Where it looks scores it; the run's own forward
        // only when the model cannot be rebuilt, and then it says so.
        const std::vector<float>* eval_output = EvalModelOutput();
        const std::string eval_unavailable = EvalModelUnavailable();
        const std::vector<float>& output = eval_output ? *eval_output : cache.model_output;
        const bool shown = eval_output || !eval_unavailable.empty();
        const auto prediction = shown ? lv::Predict(output, lv::OutputRuleFromKey(facts.output_rule), facts.class_names)
                                      : std::nullopt;
        if (!shown) {
            ImGui::TextColored(DebuggerMuted(), "Reading the prediction (evaluation pass)...");
        } else if (prediction) {
            ImGui::TextColored(DebuggerBright(), "Prediction %s", prediction->text.c_str());
            if (!prediction->detail.empty()) ImGui::TextColored(DebuggerMuted(), "%s", prediction->detail.c_str());
        } else if (!output.empty() && output.size() <= 16) {
            std::string out = "Model output";
            for (size_t i = 0; i < output.size(); ++i) out += (i ? ", " : " ") + Number(output[i]);
            ImGui::TextColored(DebuggerText(), "%s", out.c_str());
        }
        if (shown) {
            ImGui::TextColored(DebuggerFaint(), "%s",
                               eval_output ? "From the evaluation pass (dropout off, BatchNorm on its running statistics)."
                                           : ("From the run's training step: " + eval_unavailable + ".").c_str());
        }
        ImGui::TextColored(DebuggerFaint(), "After the data steps, as the first layer received it.");
        ImGui::PopTextWrapPos();
    }
    EndCard();
    ImGui::Spacing();

    const auto& views = cache.views;
    const lv::View shown_view = views.tabs.empty()
                                    ? lv::View::Values
                                    : views.tabs[static_cast<size_t>(std::clamp(selected_layer_tab_, 0,
                                                                                static_cast<int>(views.tabs.size()) - 1))]
                                          .view;
    const bool feature_maps = !views.tabs.empty() && shown_view == lv::View::FeatureMaps;
    const std::string& hint = shown_view == lv::View::Filters        ? lv::FiltersHint()
                              : shown_view == lv::View::WhereItLooks
                                  ? (CaptureSettings().where_it_looks_method == core::WhereItLooksMethod::Saliency
                                         ? where_it_looks::SaliencyInspectorHint()
                                         : where_it_looks::InspectorHint())
                              : shown_view == lv::View::TopImages ? lv::TopImagesHint()
                              : shown_view == lv::View::Embedding ? lv::EmbeddingHint()
                              : shown_view == lv::View::UnitWeights ? unit_weights::Hint()
                              : shown_view == lv::View::Contributions && !contributions_inspector_.empty()
                                  ? contributions_inspector_
                                                                     : views.hint;
    if (feature_maps && cache.spatial && selected_channel_rank_ < cache.tiles.size()) {
        if (BeginCard("StudioDebuggerLayerChannel")) {
            ImGui::PushTextWrapPos(0.0f);
            const size_t c = cache.order[selected_channel_rank_];
            ImGui::TextColored(DebuggerMuted(), "Channel %zu of %s", c, cache.rows[selected_layer_].name.c_str());
            const float side = ImGui::GetContentRegionAvail().x;
            const float h = side * static_cast<float>(cache.spatial->height) /
                            static_cast<float>(std::max<size_t>(1, cache.spatial->width));
            ImGui::Image(Texture(cache.tiles[selected_channel_rank_]), ImVec2(side, h));
            ImGui::Text("mean %s \xC2\xB7 max %s \xC2\xB7 rank %zu of %zu", Number(cache.channels.means[c]).c_str(),
                        Number(cache.channels.maxima[c]).c_str(), selected_channel_rank_ + 1, cache.spatial->channels);
            ImGui::TextColored(DebuggerFaint(), "%zu \xC3\x97 %zu per channel.", cache.spatial->height, cache.spatial->width);
            ImGui::PopTextWrapPos();
        }
        EndCard();
    } else if (!hint.empty()) {
        if (BeginCard("StudioDebuggerLayerHint")) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(DebuggerText(), "%s", hint.c_str());
            ImGui::PopTextWrapPos();
        }
        EndCard();
    }
    ImGui::EndChild();
    return true;
}

}  // namespace cyxwiz
