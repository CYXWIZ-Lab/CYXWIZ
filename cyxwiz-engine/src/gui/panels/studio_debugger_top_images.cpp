// Studio Debugger > Model > Layers > Top images (TOFIX178 6.3e, approved
// mockup boards Main and TopImages): the items from the data that excite a
// layer's channels most. Items spread over the data run through the run's
// rebuilt model (DebugModelSession) in chunks off the UI thread; rows are the
// Dense head's strongest votes, the clicked channel, or this sample's
// strongest channels.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/async_task_manager.h"
#include "../../core/debug_model_session.h"
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
#include <atomic>
#include <map>

namespace cyxwiz {

namespace lv = layer_view;
namespace ti = top_images;

namespace {

constexpr size_t kTopItems = 320;      // items spread over the data
constexpr size_t kShownPerRow = 9;     // the 9 strongest per channel
constexpr size_t kStrongestRows = 16;  // "the 16 strongest for this sample"
constexpr size_t kChunkItems = 64;
constexpr size_t kChunkElements = size_t{8} << 20;
constexpr float kThumb = 64.0f;
constexpr float kLabelWidth = 150.0f;  // the row's words, left of its pictures

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

std::string Value(float v) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3g", v);
    return buffer;
}

}  // namespace

struct StudioDebuggerPanel::TopImagesState {
    bool mean = false;  // Ranked by: Maximum | Mean (spatial layers)
    int choice = 0;     // the Channels option picked
    bool computing = false;
    std::shared_ptr<std::atomic<size_t>> done = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::atomic<size_t>> total = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::atomic<bool>> cancel = std::make_shared<std::atomic<bool>>(false);
    std::string result_key;  // capture / layer / rank the result is for
    ti::Strengths strengths;
    DataItems items;
    std::vector<int> labels;
    std::map<size_t, uint32_t> thumbs;  // item -> picture
    std::string error;

    void DropThumbs() {
        for (auto& [item, id] : thumbs)
            if (id) TextureManager::Instance().DeleteTexture(id);
        thumbs.clear();
    }
    ~TopImagesState() { DropThumbs(); }
};

void StudioDebuggerPanel::NoteClickedChannel(size_t layer, size_t channel) {
    clicked_channel_layer_ = layer;
    clicked_channel_ = channel;
}

void StudioDebuggerPanel::RenderTopImages() {
    if (!top_images_) top_images_ = std::make_shared<TopImagesState>();
    TopImagesState& s = *top_images_;
    UpdateLayerModelSession();
    const auto& layers = layer_capture_->Layers();
    const CapturedLayer& layer = layers[selected_layer_];
    const CapturedTensor* output = layer.Find(CapturedTensorRole::Output);
    const CapturedTensor* input = layers.front().Find(CapturedTensorRole::Input);
    const std::string unavailable = EvalModelUnavailable();
    if (!unavailable.empty()) {
        Wrapped(DebuggerMuted(), "Top images run the run's model again, and it cannot be rebuilt: " + unavailable + ".");
        return;
    }
    if (!output) {
        Wrapped(DebuggerMuted(), "The run did not keep this layer's output (Layer capture: keep activations).");
        return;
    }
    const auto& out_shape = output->tensor.Shape();
    const bool spatial = out_shape.size() == 4;
    const size_t channels = spatial ? out_shape[2] : (out_shape.size() == 2 ? out_shape[1] : 0);
    const auto picture_shape = input ? lv::AsSpatial(input->tensor.Shape()) : std::nullopt;
    const bool pictures = picture_shape && (picture_shape->channels == 1 || picture_shape->channels == 3);
    const std::string items_word = pictures ? "images" : "samples";

    // The Dense output head that reads these channels: the last layer with
    // weights [outputs, channels].
    std::vector<ti::Vote> votes;
    for (auto it = layers.rbegin(); it != layers.rend() && votes.empty(); ++it) {
        if (it->module_index <= layer.module_index) break;
        for (const auto& t : it->tensors) {
            const auto& w = t.tensor.Shape();
            if (t.role != CapturedTensorRole::Parameter || w.size() != 2 || w[1] != channels) continue;
            if (&*it != &layers.back()) break;  // a head is the model's last layer
            votes = ti::StrongestVotes(lv::ReadValues(t.tensor), w[0], w[1], 2);
            break;
        }
    }
    const auto& facts = ShownSampleFacts();
    const auto class_name = [&](int index) {
        return lv::ClassName(index, facts.class_names.empty() ? s.items.class_names : facts.class_names);
    };

    // Channels: the head's votes (when there is a head), the clicked
    // channel, this sample's strongest (a spatial layer).
    std::vector<const char*> options;
    if (!votes.empty()) options.push_back("The Dense head's strongest votes");
    options.push_back("The clicked channel");
    if (spatial) options.push_back("The 16 strongest for this sample");
    s.choice = std::clamp(s.choice, 0, static_cast<int>(options.size()) - 1);
    const std::string picked_option = options[static_cast<size_t>(s.choice)];
    const size_t clicked = clicked_channel_layer_ == selected_layer_ ? std::min(clicked_channel_, channels - 1) : 0;

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), "Images");
    ImGui::SameLine(110.0f);
    Wrapped(DebuggerText(), std::to_string(kTopItems) + ", spread over the data, as validation prepares them" +
                                (s.items.description.empty() ? "" : " (" + s.items.description + ")"));
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), "Channels");
    ImGui::SameLine(110.0f);
    ui::SegmentedControl("TopImagesChannels", options.data(), static_cast<int>(options.size()), &s.choice);
    if (spatial) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "Ranked by");
        ImGui::SameLine(110.0f);
        static const char* const kRanks[] = {"Maximum", "Mean"};
        int rank = s.mean ? 1 : 0;
        if (ui::SegmentedControl("TopImagesRank", kRanks, 2, &rank)) s.mean = rank == 1;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Maximum: the channel's largest value over its map (it fires somewhere).\n"
                              "Mean: its mean over the map (it fires all over).");
    }

    const ti::Rank rank = s.mean ? ti::Rank::Mean : ti::Rank::Maximum;
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                            std::to_string(layer.module_index) + "/" + std::to_string(s.mean);
    ImGui::Spacing();
    if (s.computing) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerAccentText(), "Computing");
        ImGui::SameLine();
        ImGui::Text("%zu of %zu %s", s.done->load(), s.total->load(), items_word.c_str());
        ImGui::SameLine(0.0f, 16.0f);
        if (ui::DangerButton("Stop")) s.cancel->store(true);
    } else {
        const bool have = s.result_key == key;
        const auto session = LayerModelSession();
        if (ui::PrimaryButton(have ? "Compute again" : "Compute", session != nullptr, "Rebuilding the run's model...")) {
            const DataItemsStart start = data_items_callback_ ? data_items_callback_() : DataItemsStart{};
            const DataItemsJob job = start.job;
            if (!job) {
                s.error = DataItemsNotStarted(start, "the " + items_word);
            } else {
                s.computing = true;
                s.error.clear();
                s.done->store(0);
                s.total->store(kTopItems);
                s.cancel->store(false);
                struct Computed {
                    ti::Strengths strengths;
                    DataItems items;
                    std::vector<int> labels;
                    std::string error;
                };
                auto computed = std::make_shared<Computed>();
                auto state = top_images_;
                const size_t module = layer.module_index;
                AsyncTaskManager::Instance().RunAsync(
                    "Studio Debugger: top images",
                    [state, computed, job, session, module, rank](LambdaTask&) {
                        try {
                            computed->items = job(kTopItems, [state] { return state->cancel->load(); });
                            if (!computed->items.why.empty()) throw std::runtime_error(computed->items.why);
                            const auto& shape = computed->items.features.Shape();
                            const size_t n = shape.size() == 4 ? shape.back() : (shape.empty() ? 0 : shape.front());
                            const size_t per_item = n ? computed->items.features.NumElements() / n : 1;
                            state->total->store(n);
                            const size_t chunk = std::clamp<size_t>(kChunkElements / std::max<size_t>(1, per_item), 1,
                                                                    kChunkItems);
                            computed->strengths = ti::Run(
                                *session, computed->items.features, module, rank, chunk,
                                [state](size_t done, size_t) { state->done->store(done); },
                                [state] { return state->cancel->load(); });
                            if (computed->items.labels.NumElements() > 0)
                                computed->labels = lv::ItemLabels(lv::ReadValues(computed->items.labels),
                                                                  computed->items.labels.Shape());
                        } catch (const std::exception& e) {
                            computed->error = e.what();
                        }
                    },
                    nullptr,
                    [state, computed, key](bool, const std::string&) {
                        state->computing = false;
                        if (!computed->error.empty()) {
                            state->error = computed->error;
                            return;
                        }
                        if (computed->strengths.cancelled) return;
                        state->DropThumbs();
                        state->result_key = key;
                        state->strengths = std::move(computed->strengths);
                        state->items = std::move(computed->items);
                        state->labels = std::move(computed->labels);
                    });
            }
        }
        ImGui::SameLine(0.0f, 12.0f);
        Wrapped(DebuggerMuted(), std::to_string(kTopItems) + " " + items_word + " through the model in batches on the "
                                 "device; Stop ends it. Not saved: computed when asked.");
    }
    if (!s.error.empty()) Wrapped(DebuggerMuted(), "Not computed: " + s.error + ".");
    if (s.result_key != key || s.strengths.channels != channels) return;

    // The rows: which channels, and what each says.
    struct Row {
        size_t channel = 0;
        std::string sub;
        std::string vote;
        int towards = -1;
    };
    std::vector<Row> rows;
    if (picked_option == std::string("The Dense head's strongest votes")) {
        for (const auto& v : votes) {
            char weight[32];
            std::snprintf(weight, sizeof(weight), "%+.3f", v.weight);
            rows.push_back({v.channel, std::string("Dense weight ") + weight, "votes " + class_name(v.towards), v.towards});
        }
    } else if (picked_option == std::string("The clicked channel")) {
        rows.push_back({clicked,
                        spatial ? std::string(s.mean ? "its mean" : "its maximum") + " over " +
                                      std::to_string(out_shape[0]) + " \xC3\x97 " + std::to_string(out_shape[1])
                                : std::string("its value"),
                        clicked_channel_layer_ == selected_layer_ ? "clicked" : "click a channel to pick it", -1});
    } else {
        // This sample's strongest channels, as its captured output ranks them.
        try {
            const Tensor one = lv::SampleOf(output->tensor, selected_layer_sample_);
            auto with_batch = one.Shape();
            with_batch.push_back(1);
            const auto own = ti::ChannelStrengths(Tensor::FromRowMajorFlatArray(one.GetRowMajorFlatArray(), with_batch),
                                                  rank);
            const auto order = ti::TopItems([&] {
                ti::Strengths by_channel;  // channels as items: rank them
                by_channel.channels = 1;
                by_channel.items = own.channels;
                by_channel.values = own.values;
                return by_channel;
            }(), 0, kStrongestRows);
            for (size_t r = 0; r < order.size(); ++r)
                rows.push_back({order[r], "rank " + std::to_string(r + 1) + " for this sample", "", -1});
        } catch (const std::exception& e) {
            Wrapped(DebuggerMuted(), std::string("This sample's channels could not be read: ") + e.what());
        }
    }

    ImGui::Spacing();
    Wrapped(DebuggerMuted(), "The " + std::to_string(kShownPerRow) + " " + items_word + " (of " +
                                 std::to_string(s.strengths.items) + ") that excite each channel most" +
                                 (spatial ? std::string(", ranked by the channel's ") + (s.mean ? "mean" : "maximum") +
                                                " over its map. Patches (the part the channel sees) come later."
                                          : "."));
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (const auto& row : rows) {
        ImGui::PushID(static_cast<int>(row.channel));
        // The row's words wrap in their column; the pictures wrap under the
        // first when the view is narrow (nothing is clipped).
        const float row_x = ImGui::GetCursorPosX();
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(row_x + kLabelWidth - 10.0f);
        ImGui::Text("Channel %zu", row.channel);
        ImGui::TextColored(DebuggerMuted(), "%s", row.sub.c_str());
        if (!row.vote.empty())
            ImGui::TextColored(row.towards >= 0 ? ui::DivergingAt(t, row.towards == 0 ? -0.8f : 0.8f) : DebuggerMuted(),
                               "%s", row.vote.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        ImGui::SameLine(row_x + kLabelWidth);
        ImGui::BeginGroup();
        const float pictures_x = ImGui::GetCursorPosX();
        const auto top = ti::TopItems(s.strengths, row.channel, kShownPerRow);
        for (size_t k = 0; k < top.size(); ++k) {
            const size_t item = top[k];
            if (k) {
                ImGui::SameLine(0.0f, 6.0f);
                if (ImGui::GetContentRegionAvail().x < kThumb) {
                    ImGui::NewLine();
                    ImGui::SetCursorPosX(pictures_x);
                }
            }
            ImGui::PushID(static_cast<int>(k));
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const bool pressed = ImGui::InvisibleButton("##item", ImVec2(kThumb, kThumb));
            if (pictures) {
                auto found = s.thumbs.find(item);
                if (found == s.thumbs.end()) {
                    uint32_t id = 0;
                    try {
                        const Tensor one = ti::ItemRange(s.items.features, item, item + 1);
                        auto shape = one.Shape();
                        shape.pop_back();
                        if (const auto picture = lv::SamplePicture(lv::ReadValues(one), shape))
                            id = TextureManager::Instance().CreateTextureFromUint8Data(
                                picture->rgba.data(), static_cast<int>(picture->width), static_cast<int>(picture->height), 4);
                    } catch (const std::exception& e) {
                        spdlog::debug("Top images: picture {} not made: {}", item, e.what());
                    }
                    found = s.thumbs.emplace(item, id).first;
                }
                if (found->second)
                    dl->AddImageRounded(static_cast<ImTextureID>(found->second), at, ImVec2(at.x + kThumb, at.y + kThumb),
                                        ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 3.0f);
            } else {
                dl->AddRectFilled(at, ImVec2(at.x + kThumb, at.y + kThumb), ui::ToU32(t.bg_raised), 3.0f);
                const std::string text = item < s.items.indices.size() ? "#" + std::to_string(s.items.indices[item]) : "";
                dl->AddText(ImVec2(at.x + 6.0f, at.y + 6.0f), ui::ToU32(t.text), text.c_str());
            }
            const std::string name = item < s.items.names.size() && !s.items.names[item].empty()
                                         ? s.items.names[item]
                                         : (item < s.items.indices.size() ? items_word.substr(0, items_word.size() - 1) +
                                                                                " " + std::to_string(s.items.indices[item])
                                                                          : std::string());
            const std::string label = item < s.labels.size() && s.labels[item] >= 0 ? class_name(s.labels[item]) : "";
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s%s%s \xC2\xB7 channel value %s\nClick to make it the debugger's picked sample.",
                                  name.c_str(), label.empty() ? "" : " \xC2\xB7 label ", label.c_str(),
                                  Value(s.strengths.At(row.channel, item)).c_str());
            }
            if (pressed && item < s.items.indices.size()) {
                selected_sample_index_ = static_cast<int>(s.items.indices[item]);
                picked_from_top_images_ = name;
            }
            ImGui::PopID();
        }
        ImGui::EndGroup();
        ImGui::PopID();
        ImGui::Spacing();
    }
    if (!picked_from_top_images_.empty())
        Wrapped(DebuggerAccentText(), "Picked " + picked_from_top_images_ + " (sample " +
                                          std::to_string(selected_sample_index_) + "): run Local Debug to see it.");
    else
        Wrapped(DebuggerFaint(), "Hover a picture for its name, label and the channel's value. Click one to make it "
                                 "the debugger's picked sample.");
}

}  // namespace cyxwiz
