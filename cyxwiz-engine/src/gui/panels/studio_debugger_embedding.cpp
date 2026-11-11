// Studio Debugger > Model > Layers > Embedding (TOFIX178 6.4b, approved mockup
// boards Main and Dense): a flat layer's output over items spread over the
// data, through the run's rebuilt model, projected to 2-D with Explore >
// Projection's methods and coloured by the true or the predicted class. This
// sample is ringed; hover a point for its item, click it for the Inspector.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_items_run.h"
#include "studio_debugger_widgets.h"

#include "../../core/async_task_manager.h"
#include "../../core/count_text.h"
#include "../../core/debug_model_session.h"
#include "../../core/layer_embedding.h"
#include "../../core/layer_tensor_reader.h"
#include "../../core/layer_view_presentation.h"
#include "../../core/texture_manager.h"
#include "../../core/top_images.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrayfire.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <map>

namespace cyxwiz {

namespace lv = layer_view;
namespace le = layer_embedding;
namespace ti = top_images;

namespace {

constexpr size_t kItems = 320;
constexpr size_t kChunkItems = 64;
constexpr size_t kChunkElements = size_t{8} << 20;

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

struct StudioDebuggerPanel::EmbeddingState {
    // From this session's settings every frame.
    int method = 0;  // PCA, t-SNE, UMAP
    int colour = 0;  // true class, predicted class

    std::string result_key;  // capture / layer / method
    le::Projection projection;
    le::Scaled points;  // the items, then this sample unless it is one of them
    size_t item_count = 0;
    size_t own_index = 0;  // this sample's point
    DataItems items;
    std::vector<int> labels;
    std::vector<int> predicted;
    std::map<size_t, uint32_t> thumbs;
    std::optional<size_t> picked;
    int shown_frame = -10;

    void DropThumbs() {
        for (auto& [item, id] : thumbs)
            if (id) TextureManager::Instance().DeleteTexture(id);
        thumbs.clear();
    }
    ~EmbeddingState() { DropThumbs(); }
};

void StudioDebuggerPanel::RenderEmbedding() {
    if (!embedding_) embedding_ = std::make_shared<EmbeddingState>();
    if (!embedding_run_) embedding_run_ = std::make_shared<ItemsRun>();
    EmbeddingState& s = *embedding_;
    ItemsRun& run = *embedding_run_;
    s.shown_frame = ImGui::GetFrameCount();
    {
        const core::StudioDebuggerConfig settings = CaptureSettings();
        s.method = static_cast<int>(settings.embedding_method);
        s.colour = static_cast<int>(settings.embedding_colour);
    }
    UpdateLayerModelSession();
    const auto& layers = layer_capture_->Layers();
    const CapturedLayer& layer = layers[selected_layer_];
    const CapturedTensor* output = layer.Find(CapturedTensorRole::Output);
    const CapturedTensor* input = layers.front().Find(CapturedTensorRole::Input);
    const std::string unavailable = EvalModelUnavailable();
    if (!unavailable.empty()) {
        Wrapped(DebuggerMuted(), "Embedding runs the run's model again, and it cannot be rebuilt: " + unavailable + ".");
        return;
    }
    if (!output || output->tensor.Shape().size() != 2) {
        Wrapped(DebuggerMuted(), "The run did not keep this layer's output (Layer capture: keep activations).");
        return;
    }
    const size_t features = output->tensor.Shape()[1];
    const auto picture_shape = input ? lv::AsSpatial(input->tensor.Shape()) : std::nullopt;
    const bool pictures = picture_shape.has_value();  // any channel count: a sheet's first tile
    const std::string items_word = pictures ? "images" : "samples";
    const auto& facts = ShownSampleFacts();
    const lv::OutputRule rule = lv::OutputRuleFromKey(facts.output_rule);
    const auto method = static_cast<le::Method>(s.method);
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                            std::to_string(layer.module_index) + "/" + std::to_string(s.method);
    const auto session = LayerModelSession();
    const ui::Tokens& t = ui::CurrentTokens();

    const auto start = [&]() {
        s.picked.reset();
        // This sample goes through the same evaluation pass as the items (the
        // captured output is the training step's: BatchNorm on the batch's
        // statistics, so it would sit apart from them).
        const CapturedLayer& first = layer_capture_->Layers().front();
        const CapturedTensor* first_input = first.Find(CapturedTensorRole::Input);
        if (!first_input || first.module_index != 0) {
            run.error = "the run did not keep the model's input";
            return;
        }
        const Tensor own_input = first_input->tensor;
        const size_t own_item = selected_layer_sample_;
        const DataItemsJob job = BeginItems(run, data_items_callback_, "the " + items_word);
        if (!job) return;
        struct Computed {
            DataItems items;
            std::vector<int> labels;
            std::vector<int> predicted;
            le::Projection projection;
            size_t item_count = 0;
            size_t own_index = 0;
            bool cancelled = false;
            std::string error;
        };
        auto computed = std::make_shared<Computed>();
        auto state = embedding_;
        auto shared_run = embedding_run_;
        const size_t module = layer.module_index;
        const std::vector<std::string> class_names = facts.class_names;
        AsyncTaskManager::Instance().RunAsync(
            "Studio Debugger: embedding",
            [shared_run, computed, job, session, module, method, rule, own_input, own_item, features,
             class_names](LambdaTask&) {
                ItemsRun& r = *shared_run;
                try {
                    computed->items = job(kItems, [&r] { return r.cancel->load(); });
                    if (!computed->items.why.empty()) throw std::runtime_error(computed->items.why);
                    const auto& shape = computed->items.features.Shape();
                    const size_t n = shape.size() == 4 ? shape.back() : (shape.empty() ? 0 : shape.front());
                    const size_t per_item = n ? computed->items.features.NumElements() / n : 1;
                    r.total->store(n);
                    r.stage->store(2);
                    const size_t chunk =
                        std::clamp<size_t>(kChunkElements / std::max<size_t>(1, per_item), 1, kChunkItems);
                    std::vector<float> rows;
                    rows.reserve((n + 1) * features);
                    for (size_t begin = 0; begin < n; begin += chunk) {
                        if (r.cancel->load()) {
                            computed->cancelled = true;
                            return;
                        }
                        const size_t end = std::min(n, begin + chunk);
                        const Tensor part = ti::ItemRange(computed->items.features, begin, end);
                        const auto values = lv::ReadValues(session->LayerOutput(part, module));
                        if (values.size() != (end - begin) * features)
                            throw std::runtime_error("the layer's output is not one row per item");
                        rows.insert(rows.end(), values.begin(), values.end());
                        // The model's answer for each item, for "predicted class".
                        const auto out = lv::ReadValues(session->Forward(part));
                        const size_t per = out.size() / (end - begin);
                        for (size_t i = 0; i < end - begin; ++i) {
                            const std::vector<float> one(out.begin() + static_cast<std::ptrdiff_t>(i * per),
                                                         out.begin() + static_cast<std::ptrdiff_t>((i + 1) * per));
                            const auto p = lv::Predict(one, rule, class_names);
                            computed->predicted.push_back(p ? p->class_index : -1);
                        }
                        r.done->store(end);
                    }
                    const auto own = lv::ReadValues(
                        session->LayerOutput(ti::ItemRange(own_input, own_item, own_item + 1), module));
                    if (own.size() != features) throw std::runtime_error("this sample's output is not one row");
                    const auto among = le::SameRow(rows, n, features, own);
                    computed->item_count = n;
                    computed->own_index = among ? *among : n;
                    if (!among) rows.insert(rows.end(), own.begin(), own.end());  // this sample, ringed
                    r.stage->store(3);
                    computed->projection = le::Project(rows, rows.size() / features, features, method);
                    if (!computed->projection.error.empty()) throw std::runtime_error(computed->projection.error);
                    if (computed->items.labels.NumElements() > 0)
                        computed->labels =
                            lv::ItemLabels(lv::ReadValues(computed->items.labels), computed->items.labels.Shape());
                } catch (const std::exception& e) {
                    computed->error = e.what();
                }
            },
            nullptr,
            [state, shared_run, computed, key](bool, const std::string&) {
                shared_run->computing = false;
                if (!computed->error.empty()) {
                    shared_run->error = computed->error;
                    return;
                }
                if (computed->cancelled) return;
                state->DropThumbs();
                state->result_key = key;
                state->projection = std::move(computed->projection);
                state->points = le::ToUnit(state->projection);
                state->item_count = computed->item_count;
                state->own_index = computed->own_index;
                state->items = std::move(computed->items);
                state->labels = std::move(computed->labels);
                state->predicted = std::move(computed->predicted);
            });
    };
    if (session && RetryItems(run)) start();

    // The settings bar: Method, Colour by, how many, Compute / Stop.
    const bool busy = run.Busy();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::BeginChild("EmbeddingSettings", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    {
        ImGui::BeginDisabled(busy);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "Method");
        ImGui::SameLine();
        static const char* const kMethods[] = {"PCA", "t-SNE", "UMAP"};
        int method_choice = s.method;
        if (ui::SegmentedControl("EmbeddingMethod", kMethods, 3, &method_choice))
            ChangeSessionSettings([method_choice](core::StudioDebuggerConfig& c) {
                c.embedding_method = static_cast<core::EmbeddingMethod>(method_choice);
            });
        ImGui::EndDisabled();
        SameLineIfFits(ImGui::CalcTextSize("Colour by True class Predicted class").x + 60.0f, 18.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "Colour by");
        ImGui::SameLine();
        static const char* const kColours[] = {"True class", "Predicted class"};
        int colour_choice = s.colour;
        if (ui::SegmentedControl("EmbeddingColour", kColours, 2, &colour_choice))
            ChangeSessionSettings([colour_choice](core::StudioDebuggerConfig& c) {
                c.embedding_colour = static_cast<core::EmbeddingColour>(colour_choice);
            });
        SameLineIfFits(ImGui::CalcTextSize("Save as my default").x, 18.0f);
        RenderSaveViewDefaults("Embedding", "This method and colouring are your defaults",
                               [](const core::StudioDebuggerConfig& session, core::StudioDebuggerConfig& defaults) {
                                   defaults.embedding_method = session.embedding_method;
                                   defaults.embedding_colour = session.embedding_colour;
                               });
        const std::string count = std::to_string(kItems) + " " + items_word + " spread over the data";
        SameLineIfFits(ImGui::CalcTextSize(count.c_str()).x, 18.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "%s", count.c_str());
        const char* action = busy ? "Stop" : (s.result_key == key ? "Compute again" : "Compute");
        const float width = ui::ButtonWidth(action, ui::ButtonSize::Regular);
        SameLineIfFits(width, 18.0f);
        RightAlignOnLine(width);
        if (busy) {
            if (ui::DangerButton("Stop")) StopItems(run);
        } else if (ui::PrimaryButton(action, session != nullptr, "Rebuilding the run's model...")) {
            start();
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::Spacing();

    if (busy) {
        DrawItemsRunCard(run, "Computing the embedding of " + layer.node_name,
                         {{"Prepare " + std::to_string(kItems) + " " + items_word +
                               " (spread over the data, the data steps as validation sees them)",
                           "preparing"},
                          {"Run this snapshot's model on the compute device", ""},
                          {std::string("Project to 2-D with ") + le::MethodName(method), "projecting"}},
                         2);
        return;
    }
    if (!run.error.empty()) Wrapped(DebuggerMuted(), "Not computed: " + run.error + ".");
    if (s.result_key != key || s.points.x.empty()) {
        ImGui::PushTextWrapPos(std::min(ImGui::GetContentRegionMax().x, ImGui::GetCursorPosX() + 760.0f));
        ImGui::TextColored(DebuggerBright(), "Where do %s land in %s's %s features?", items_word.c_str(),
                           layer.node_name.c_str(), CountText(features).c_str());
        ImGui::TextColored(DebuggerMuted(), "%s",
                           ("Compute takes " + std::to_string(kItems) + " " + items_word +
                            " spread over the data, runs them through this snapshot's model, projects this layer's "
                            "output to 2-D (PCA in a moment; t-SNE and UMAP take a few seconds) and colours each "
                            "point by its class. This sample is ringed.")
                               .c_str());
        ImGui::PopTextWrapPos();
        return;
    }

    // The result: the lead, the plot and its legend.
    const size_t n = std::min(s.item_count, s.points.x.size());
    const bool by_predicted = s.colour == 1;
    const std::vector<int>& colours = by_predicted ? s.predicted : s.labels;
    const std::vector<std::string> names = facts.class_names.empty() ? s.items.class_names : facts.class_names;
    size_t wrong = 0;
    for (size_t i = 0; i < n && i < s.labels.size() && i < s.predicted.size(); ++i)
        wrong += s.labels[i] >= 0 && s.predicted[i] >= 0 && s.labels[i] != s.predicted[i];
    Wrapped(DebuggerMuted(), "This layer's " + CountText(features) + " features of " + std::to_string(n) + " " +
                                 items_word + ", projected to 2-D (" + s.projection.about + "). Colour: " +
                                 (by_predicted ? "the predicted class; the " + std::to_string(wrong) +
                                                     " the model gets wrong are ringed."
                                               : std::string("the true class.")) +
                                 " Hover a point for its item; click it for the "
                                 "Inspector.");
    ImGui::Spacing();
    // The legend sits beside the plot when the column has room for both, below it otherwise.
    const float legend_w = 240.0f;
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool legend_beside = avail >= legend_w + 20.0f + 420.0f;
    const float plot_w = std::clamp(legend_beside ? avail - legend_w - 20.0f : avail, 200.0f, 640.0f);
    // Below the plot the legend needs about two lines of keys and the note.
    const float legend_h = legend_beside ? 0.0f : ImGui::GetTextLineHeightWithSpacing() * 5.0f;
    const float plot_h = std::clamp(ImGui::GetContentRegionAvail().y - legend_h, 180.0f, plot_w * 0.69f);
    const float pad = 12.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##EmbeddingPlot", ImVec2(plot_w, plot_h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + plot_w, at.y + plot_h), ui::ToU32(t.bg_input), 6.0f);
    const auto colour_of = [&](int c) {
        if (c < 0) return ui::ToU32(t.text_faint);
        return ui::ToU32(t.series[static_cast<size_t>(c) % ui::Tokens::kSeriesCount]);
    };
    const auto place = [&](size_t i) {
        return ImVec2(at.x + pad + s.points.x[i] * (plot_w - 2 * pad), at.y + pad + (1.0f - s.points.y[i]) * (plot_h - 2 * pad));
    };
    for (size_t i = 0; i < n; ++i) {
        const ImVec2 p = place(i);
        dl->AddCircleFilled(p, 3.5f, colour_of(i < colours.size() ? colours[i] : -1));
        if (by_predicted && i < s.labels.size() && i < s.predicted.size() && s.labels[i] >= 0 &&
            s.labels[i] != s.predicted[i])
            dl->AddCircle(p, 6.0f, ui::ToU32(t.warning), 0, 1.5f);
    }
    if (s.picked && *s.picked < n) dl->AddCircle(place(*s.picked), 8.0f, ui::ToU32(t.accent_text), 0, 2.0f);
    if (s.own_index < s.points.x.size()) dl->AddCircle(place(s.own_index), 7.5f, ui::ToU32(t.text_bright), 0, 2.0f);
    std::optional<size_t> hovered;
    if (ImGui::IsItemHovered()) {
        const ImVec2 m = ImGui::GetIO().MousePos;
        const float ux = (m.x - at.x - pad) / (plot_w - 2 * pad);
        const float uy = 1.0f - (m.y - at.y - pad) / (plot_h - 2 * pad);
        hovered = le::Nearest(s.points, ux, uy, 10.0f / (plot_w - 2 * pad), (plot_h - 2 * pad) / (plot_w - 2 * pad));
        if (hovered && *hovered < n) {
            const size_t i = *hovered;
            const std::string name = i < s.items.names.size() && !s.items.names[i].empty()
                                         ? s.items.names[i]
                                         : "item " + (i < s.items.indices.size() ? std::to_string(s.items.indices[i]) : "");
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(name.c_str());
            if (i == s.own_index) ImGui::TextColored(DebuggerMuted(), "this sample");
            if (i < s.labels.size() && s.labels[i] >= 0)
                ImGui::TextColored(DebuggerMuted(), "label %s", lv::ClassName(s.labels[i], names).c_str());
            if (i < s.predicted.size() && s.predicted[i] >= 0)
                ImGui::TextColored(DebuggerMuted(), "predicted %s", lv::ClassName(s.predicted[i], names).c_str());
            ImGui::EndTooltip();
        } else if (hovered) {
            ImGui::SetTooltip("This sample");
        }
    }
    if (clicked && hovered && *hovered < n) s.picked = *hovered;

    // The legend: the classes, then the rings.
    if (legend_beside)
        ImGui::SameLine(0.0f, 20.0f);
    else
        ImGui::Spacing();
    ImGui::BeginGroup();
    const auto counts = le::ClassCounts(colours.size() > n ? std::vector<int>(colours.begin(), colours.begin() + static_cast<std::ptrdiff_t>(n))
                                                          : colours,
                                        names);
    // Beside the plot one key per line; below it the keys flow across the line.
    bool first_key = true;
    const auto key_line = [&](const std::string& text, auto&& mark) {
        if (!legend_beside && !first_key) SameLineIfFits(16.0f + ImGui::CalcTextSize(text.c_str()).x + 8.0f, 18.0f);
        first_key = false;
        const ImVec2 dot = ImGui::GetCursorScreenPos();
        mark(ImGui::GetWindowDrawList(), ImVec2(dot.x + 7.0f, dot.y + ImGui::GetTextLineHeight() * 0.5f));
        ImGui::Dummy(ImVec2(16.0f, ImGui::GetTextLineHeight()));
        ImGui::SameLine();
        ImGui::TextUnformatted(text.c_str());
    };
    for (const auto& c : counts)
        key_line((by_predicted ? "predicted " : "") + c.name + " (" + std::to_string(c.count) + ")",
                 [&](ImDrawList* d, ImVec2 p) { d->AddCircleFilled(p, 4.5f, colour_of(c.label)); });
    if (by_predicted)
        key_line("wrong prediction (" + std::to_string(wrong) + ")",
                 [&](ImDrawList* d, ImVec2 p) { d->AddCircle(p, 6.0f, ui::ToU32(t.warning), 0, 1.5f); });
    key_line("this sample", [&](ImDrawList* d, ImVec2 p) { d->AddCircle(p, 6.0f, ui::ToU32(t.text_bright), 0, 2.0f); });
    if (s.picked && *s.picked < n)
        key_line("picked for the Inspector",
                 [&](ImDrawList* d, ImVec2 p) { d->AddCircle(p, 6.0f, ui::ToU32(t.accent_text), 0, 2.0f); });
    ImGui::Spacing();
    ImGui::PushTextWrapPos(legend_beside ? ImGui::GetCursorPosX() + legend_w : 0.0f);
    ImGui::TextColored(DebuggerMuted(), "%s",
                       "Separate clouds: this layer already tells the classes apart; mixed ones: the layers after it "
                       "still have work to do.");
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
}

bool StudioDebuggerPanel::RenderEmbeddingPick() {
    if (!embedding_ || !embedding_->picked || ImGui::GetFrameCount() - embedding_->shown_frame > 1) return false;
    EmbeddingState& s = *embedding_;
    const size_t i = *s.picked;
    if (i >= std::min(s.item_count, s.points.x.size())) return false;
    const auto& facts = ShownSampleFacts();
    const auto names = facts.class_names.empty() ? s.items.class_names : facts.class_names;
    if (BeginCard("EmbeddingPick")) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerMuted(), "Embedding point");
        auto found = s.thumbs.find(i);
        if (found == s.thumbs.end()) {
            uint32_t id = 0;
            try {
                const Tensor one = ti::ItemRange(s.items.features, i, i + 1);
                auto shape = one.Shape();
                if (shape.size() == 4) shape.pop_back();
                if (const auto picture =
                        lv::SamplePicture(lv::ReadValues(one), shape, layer_capture_->Options().input_channels))
                    id = TextureManager::Instance().CreateTextureFromUint8Data(
                        picture->rgba.data(), static_cast<int>(picture->width), static_cast<int>(picture->height), 4);
            } catch (const std::exception& e) {
                spdlog::debug("Embedding: picture {} not made: {}", i, e.what());
            }
            found = s.thumbs.emplace(i, id).first;
        }
        if (found->second) {
            const float side = std::min(272.0f, ImGui::GetContentRegionAvail().x);
            ImGui::Image(static_cast<ImTextureID>(found->second), ImVec2(side, side));
        }
        const std::string name = i < s.items.names.size() && !s.items.names[i].empty()
                                     ? s.items.names[i]
                                     : "item " + (i < s.items.indices.size() ? std::to_string(s.items.indices[i]) : "");
        ImGui::TextColored(DebuggerBright(), "%s", name.c_str());
        if (i < s.labels.size() && s.labels[i] >= 0)
            ImGui::Text("label %s", lv::ClassName(s.labels[i], names).c_str());
        if (i < s.predicted.size() && s.predicted[i] >= 0)
            ImGui::TextColored(DebuggerMuted(), "predicted %s", lv::ClassName(s.predicted[i], names).c_str());
        if (i < s.items.indices.size()) {
            ImGui::Spacing();
            if (ui::SecondaryButton("Make it the picked sample")) {
                selected_sample_index_ = static_cast<int>(s.items.indices[i]);
                picked_from_top_images_ = name;
            }
            if (!picked_from_top_images_.empty() && selected_sample_index_ == static_cast<int>(s.items.indices[i]))
                ImGui::TextColored(DebuggerAccentText(), "Picked: run Local Debug to look inside the model for it.");
        }
        ImGui::PopTextWrapPos();
    }
    EndCard();
    ImGui::Spacing();
    return true;
}

}  // namespace cyxwiz
