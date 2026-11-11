// Studio Debugger > Model > Layers > Top images (TOFIX178 6.3e, approved
// mockup boards Main, TopImages and TopImagesStates): the items from the data
// that excite a layer's channels most. Items spread over the data run through
// the run's rebuilt model (DebugModelSession) in chunks off the UI thread.
// One settings bar; while it runs, a card with its steps (load the data,
// prepare the items, run the model); then one card per channel with
// captioned pictures, the picked one large in the Inspector.

#include "studio_debugger_panel.h"
#include "studio_debugger_items_run.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/async_task_manager.h"
#include "../../core/debug_model_session.h"
#include "../../core/layer_tensor_reader.h"
#include "../../core/layer_view_presentation.h"
#include "../../core/texture_manager.h"
#include "../../core/top_images.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrayfire.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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
constexpr float kTileOne = 104.0f;   // a channel's pictures when one channel is shown
constexpr float kTileMany = 72.0f;   // ... when several are
constexpr float kInspectorPicture = 272.0f;

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

struct StudioDebuggerPanel::TopImagesState {
    // Ranked by (Maximum | Mean, spatial layers) and the Channels option
    // picked: from this session's settings every frame.
    bool mean = false;
    int choice = 0;


    // The result.
    std::string result_key;  // capture / layer / rank the result is for
    ti::Strengths strengths;
    DataItems items;
    std::vector<int> labels;
    std::map<size_t, uint32_t> thumbs;  // item -> picture

    // The picture picked for the Inspector.
    std::optional<size_t> picked_item;
    size_t picked_channel = 0;
    size_t picked_rank = 0;
    int shown_frame = -10;  // the frame the view last drew (the Inspector follows it)

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
    s.shown_frame = ImGui::GetFrameCount();
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
    const ui::Tokens& t = ui::CurrentTokens();

    // The Dense output head that reads these channels: the model's last layer
    // with weights [outputs, channels].
    std::vector<ti::Vote> votes;
    for (auto it = layers.rbegin(); it != layers.rend() && votes.empty(); ++it) {
        if (it->module_index <= layer.module_index) break;
        for (const auto& tensor : it->tensors) {
            const auto& w = tensor.tensor.Shape();
            if (tensor.role != CapturedTensorRole::Parameter || w.size() != 2 || w[1] != channels) continue;
            if (&*it != &layers.back()) break;
            votes = ti::StrongestVotes(lv::ReadValues(tensor.tensor), w[0], w[1], 2);
            break;
        }
    }
    const auto& facts = ShownSampleFacts();
    const auto class_name = [&](int index) {
        return lv::ClassName(index, facts.class_names.empty() ? s.items.class_names : facts.class_names);
    };

    // Channels: the head's votes (with a head), the clicked channel, this
    // sample's strongest (a conv layer).
    // The user's choice where the layer offers it, else the layer's first.
    using core::TopImagesChannels;
    std::vector<const char*> options;
    std::vector<TopImagesChannels> option_kinds;
    const auto offer = [&](const char* text, TopImagesChannels kind) {
        options.push_back(text);
        option_kinds.push_back(kind);
    };
    if (!votes.empty()) offer("The Dense head's strongest votes", TopImagesChannels::HeadVotes);
    offer("The clicked channel", TopImagesChannels::Clicked);
    if (spatial) offer("The 16 strongest for this sample", TopImagesChannels::Strongest);
    if (show_top_images_of_clicked_) {  // asked from a unit's Inspector card
        show_top_images_of_clicked_ = false;
        ChangeSessionSettings(
            [](core::StudioDebuggerConfig& c) { c.top_images_channels = TopImagesChannels::Clicked; });
    }
    const core::StudioDebuggerConfig settings = CaptureSettings();
    const auto wanted = std::find(option_kinds.begin(), option_kinds.end(), settings.top_images_channels);
    s.choice = wanted == option_kinds.end() ? 0 : static_cast<int>(wanted - option_kinds.begin());
    s.mean = settings.top_images_rank == core::TopImagesRank::Mean;
    const std::string picked_option = options[static_cast<size_t>(s.choice)];
    const bool clicked_here = clicked_channel_layer_ == selected_layer_;
    const size_t clicked = clicked_here ? std::min(clicked_channel_, channels - 1) : 0;
    const ti::Rank rank = s.mean ? ti::Rank::Mean : ti::Rank::Maximum;
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                            std::to_string(layer.module_index) + "/" + std::to_string(s.mean);
    if (!top_images_run_) top_images_run_ = std::make_shared<ItemsRun>();
    ItemsRun& run = *top_images_run_;
    const bool busy = run.Busy();
    const auto session = LayerModelSession();

    // Starts a computation now, or after the graph's data has loaded.
    const auto start = [&]() {
        s.picked_item.reset();
        const DataItemsJob job = BeginItems(run, data_items_callback_, "the " + items_word);
        if (!job) return;
        struct Computed {
            ti::Strengths strengths;
            DataItems items;
            std::vector<int> labels;
            std::string error;
        };
        auto computed = std::make_shared<Computed>();
        auto state = top_images_;
        auto shared_run = top_images_run_;
        const size_t module = layer.module_index;
        AsyncTaskManager::Instance().RunAsync(
            "Studio Debugger: top images",
            [shared_run, computed, job, session, module, rank](LambdaTask&) {
                ItemsRun& r = *shared_run;
                try {
                    computed->items = job(kTopItems, [&r] { return r.cancel->load(); });
                    if (!computed->items.why.empty()) throw std::runtime_error(computed->items.why);
                    const auto& shape = computed->items.features.Shape();
                    const size_t n = shape.size() == 4 ? shape.back() : (shape.empty() ? 0 : shape.front());
                    const size_t per_item = n ? computed->items.features.NumElements() / n : 1;
                    r.total->store(n);
                    r.stage->store(2);
                    const size_t chunk = std::clamp<size_t>(kChunkElements / std::max<size_t>(1, per_item), 1,
                                                            kChunkItems);
                    computed->strengths = ti::Run(
                        *session, computed->items.features, module, rank, chunk,
                        [&r](size_t done, size_t) { r.done->store(done); }, [&r] { return r.cancel->load(); });
                    if (computed->items.labels.NumElements() > 0)
                        computed->labels = lv::ItemLabels(lv::ReadValues(computed->items.labels),
                                                          computed->items.labels.Shape());
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
                if (computed->strengths.cancelled) return;
                state->DropThumbs();
                state->result_key = key;
                state->strengths = std::move(computed->strengths);
                state->items = std::move(computed->items);
                state->labels = std::move(computed->labels);
            });
    };
    if (session && RetryItems(run)) start();

    // The settings bar: Channels, Ranked by, how many, Compute / Stop.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::BeginChild("TopImagesSettings", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    {
        ImGui::BeginDisabled(busy);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "Channels");
        ImGui::SameLine();
        int choice = s.choice;
        if (ui::SegmentedControl("TopImagesChannels", options.data(), static_cast<int>(options.size()), &choice)) {
            const TopImagesChannels kind = option_kinds[static_cast<size_t>(choice)];
            ChangeSessionSettings([kind](core::StudioDebuggerConfig& c) { c.top_images_channels = kind; });
        }
        if (spatial) {
            SameLineIfFits(ImGui::CalcTextSize("Ranked by Maximum Mean").x + 60.0f, 18.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(DebuggerMuted(), "Ranked by");
            ImGui::SameLine();
            static const char* const kRanks[] = {"Maximum", "Mean"};
            int ranked = s.mean ? 1 : 0;
            if (ui::SegmentedControl("TopImagesRank", kRanks, 2, &ranked))
                ChangeSessionSettings([ranked](core::StudioDebuggerConfig& c) {
                    c.top_images_rank = ranked == 1 ? core::TopImagesRank::Mean : core::TopImagesRank::Maximum;
                });
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Maximum: the channel's largest value over its map (it fires somewhere).\n"
                                  "Mean: its mean over the map (it fires all over).");
        }
        ImGui::EndDisabled();
        SameLineIfFits(ImGui::CalcTextSize("Save as my default").x, 18.0f);
        RenderSaveViewDefaults("TopImages", "These Top images choices are your defaults",
                               [](const core::StudioDebuggerConfig& session, core::StudioDebuggerConfig& defaults) {
                                   defaults.top_images_channels = session.top_images_channels;
                                   defaults.top_images_rank = session.top_images_rank;
                               });
        const std::string count = std::to_string(kTopItems) + " " + items_word + " spread over the data";
        SameLineIfFits(ImGui::CalcTextSize(count.c_str()).x, 18.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "%s", count.c_str());
        const bool have = s.result_key == key;
        const char* action = busy ? "Stop" : (have ? "Compute again" : "Compute");
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

    const std::string channel_words = picked_option == std::string("The clicked channel")
                                          ? "channel " + std::to_string(clicked)
                                          : (picked_option == std::string("The Dense head's strongest votes")
                                                 ? "the head's strongest channels"
                                                 : "this sample's strongest channels");

    // While it runs: its steps, ticked as they end.
    if (busy) {
        DrawItemsRunCard(run, "Computing top images for " + channel_words,
                         {{"Prepare " + std::to_string(kTopItems) + " " + items_word +
                               " (spread over the data, the data steps as validation sees them)",
                           "preparing"},
                          {"Run this snapshot's model on the compute device", ""}},
                         2);
        return;
    }
    if (!run.error.empty()) Wrapped(DebuggerMuted(), "Not computed: " + run.error + ".");

    // Ready: what Compute does.
    if (s.result_key != key || s.strengths.channels != channels) {
        ImGui::PushTextWrapPos(std::min(ImGui::GetContentRegionMax().x, ImGui::GetCursorPosX() + 760.0f));
        ImGui::TextColored(DebuggerBright(), "Which %s make %s fire?", items_word.c_str(), channel_words.c_str());
        ImGui::TextColored(DebuggerMuted(), "%s",
                           ("Compute takes " + std::to_string(kTopItems) + " " + items_word +
                            " spread over the data (both ends of a folder sorted by class), runs them through this "
                            "snapshot's model on the compute device and shows the 9 that excite each channel most. "
                            "The first time the data may need to load (it says so); after that a few seconds.")
                               .c_str());
        if (picked_option == std::string("The clicked channel") && !clicked_here)
            ImGui::TextColored(DebuggerMuted(), "Click a channel in Feature maps or Units to pick it (channel 0 now).");
        ImGui::PopTextWrapPos();
        return;
    }

    // The result: one card per channel.
    struct Row {
        size_t channel = 0;
        std::string sub;
        std::string vote;
        int towards = -1;
    };
    std::vector<Row> rows;
    if (picked_option == std::string("The Dense head's strongest votes")) {
        for (const auto& v : votes)
            rows.push_back({v.channel, "Dense weight " + ti::ValueText(v.weight), "votes " + class_name(v.towards),
                            v.towards});
    } else if (picked_option == std::string("The clicked channel")) {
        rows.push_back({clicked,
                        (clicked_here ? std::string("clicked in Feature maps \xC2\xB7 ") : std::string()) +
                            (spatial ? std::string(s.mean ? "its mean" : "its maximum") + " over " +
                                           std::to_string(out_shape[0]) + " \xC3\x97 " + std::to_string(out_shape[1])
                                     : std::string("its value")),
                        "", -1});
    } else {
        try {
            const Tensor one = lv::SampleOf(output->tensor, selected_layer_sample_);
            auto with_batch = one.Shape();
            with_batch.push_back(1);
            const auto own =
                ti::ChannelStrengths(Tensor::FromRowMajorFlatArray(one.GetRowMajorFlatArray(), with_batch), rank);
            ti::Strengths by_channel;  // channels as items: rank them
            by_channel.channels = 1;
            by_channel.items = own.channels;
            by_channel.values = own.values;
            const auto order = ti::TopItems(by_channel, 0, kStrongestRows);
            for (size_t r = 0; r < order.size(); ++r)
                rows.push_back({order[r], "rank " + std::to_string(r + 1) + " for this sample", "", -1});
        } catch (const std::exception& e) {
            Wrapped(DebuggerMuted(), std::string("This sample's channels could not be read: ") + e.what());
        }
    }
    Wrapped(DebuggerMuted(), "The " + std::to_string(kShownPerRow) + " " + items_word + " (of " +
                                 std::to_string(s.strengths.items) + ") that excite " +
                                 (rows.size() == 1 ? "channel " + std::to_string(rows.front().channel)
                                                   : std::string("each channel")) +
                                 " most, strongest first" + (s.items.description.empty() ? "" : " (" + s.items.description + ")") +
                                 ". Click one to inspect it." +
                                 (spatial ? " Patches (the part a channel sees) come later." : ""));
    ImGui::Spacing();

    const float tile = rows.size() == 1 ? kTileOne : kTileMany;
    const float gap = 10.0f;
    const auto name_of = [&](size_t item) {
        if (item < s.items.names.size() && !s.items.names[item].empty()) return s.items.names[item];
        return item < s.items.indices.size() ? items_word.substr(0, items_word.size() - 1) + " " +
                                                   std::to_string(s.items.indices[item])
                                             : std::string();
    };
    const auto label_of = [&](size_t item) {
        return item < s.labels.size() && s.labels[item] >= 0 ? class_name(s.labels[item]) : std::string();
    };
    for (const auto& row : rows) {
        const auto top = ti::TopItems(s.strengths, row.channel, kShownPerRow);
        std::vector<std::string> top_labels;
        std::vector<float> top_values;
        for (size_t item : top) {
            top_labels.push_back(label_of(item));
            top_values.push_back(s.strengths.At(row.channel, item));
        }
        ImGui::PushID(static_cast<int>(row.channel));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 12.0f));
        ImGui::BeginChild("card", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::TextColored(DebuggerBright(), "Channel %zu", row.channel);
        const std::string sub = row.sub + (row.vote.empty() ? "" : " \xC2\xB7 ");
        SameLineIfFits(ImGui::CalcTextSize(sub.c_str()).x, 12.0f);
        ImGui::TextColored(DebuggerMuted(), "%s", sub.c_str());
        if (!row.vote.empty()) {
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(row.towards >= 0 ? ui::DivergingAt(t, row.towards == 0 ? -0.8f : 0.8f) : DebuggerMuted(),
                               "%s", row.vote.c_str());
        }
        const std::string summary = ti::RowSummary(top_labels, top_values);
        SameLineIfFits(ImGui::CalcTextSize(summary.c_str()).x, 16.0f);
        ImGui::TextColored(DebuggerText(), "%s", summary.c_str());
        ImGui::Spacing();

        // The pictures in even rows, each captioned; nothing past the edge.
        const float avail = ImGui::GetContentRegionAvail().x;
        const int columns = std::max(1, static_cast<int>((avail + gap) / (tile + gap)));
        const float caption_h = ImGui::GetTextLineHeight() * 2.0f + 6.0f;  // "#1 · dog", its value
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (size_t k = 0; k < top.size(); ++k) {
            const size_t item = top[k];
            if (k % static_cast<size_t>(columns) != 0) ImGui::SameLine(0.0f, gap);
            ImGui::PushID(static_cast<int>(k));
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const bool selected = s.picked_item == item && s.picked_channel == row.channel;
            if (ImGui::InvisibleButton("##item", ImVec2(tile, tile + caption_h))) {
                s.picked_item = item;
                s.picked_channel = row.channel;
                s.picked_rank = k + 1;
                picked_from_top_images_.clear();
            }
            if (selected || ImGui::IsItemHovered())
                dl->AddRectFilled(ImVec2(at.x - 4.0f, at.y - 4.0f), ImVec2(at.x + tile + 4.0f, at.y + tile + caption_h + 2.0f),
                                  ui::ToU32(selected ? t.selection : t.hover), 6.0f);
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
                    dl->AddImageRounded(static_cast<ImTextureID>(found->second), at, ImVec2(at.x + tile, at.y + tile),
                                        ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 4.0f);
            } else {
                dl->AddRectFilled(at, ImVec2(at.x + tile, at.y + tile), ui::ToU32(t.bg_input), 4.0f);
                const std::string text = name_of(item);
                dl->PushClipRect(at, ImVec2(at.x + tile, at.y + tile), true);
                dl->AddText(ImVec2(at.x + 6.0f, at.y + 6.0f), ui::ToU32(t.text), text.c_str());
                dl->PopClipRect();
            }
            // Two short lines under the picture, centred: rank and label, value.
            const std::string caption = ti::Caption(k + 1, label_of(item));
            const std::string value = ti::ValueText(s.strengths.At(row.channel, item));
            const float line = ImGui::GetTextLineHeight();
            dl->PushClipRect(ImVec2(at.x - 4.0f, at.y), ImVec2(at.x + tile + 4.0f, at.y + tile + caption_h + 2.0f), true);
            const float cw = ImGui::CalcTextSize(caption.c_str()).x;
            dl->AddText(ImVec2(at.x + std::max(0.0f, (tile - cw) * 0.5f), at.y + tile + 3.0f), ui::ToU32(t.text),
                        caption.c_str());
            const float vw = ImGui::CalcTextSize(value.c_str()).x;
            dl->AddText(ImVec2(at.x + std::max(0.0f, (tile - vw) * 0.5f), at.y + tile + 3.0f + line),
                        ui::ToU32(t.text_dim), value.c_str());
            dl->PopClipRect();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s%s \xC2\xB7 channel value %s", name_of(item).c_str(),
                                  label_of(item).empty() ? "" : (" \xC2\xB7 label " + label_of(item)).c_str(), value.c_str());
            ImGui::PopID();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        ImGui::PopID();
        ImGui::Spacing();
    }
}

bool StudioDebuggerPanel::RenderTopImagesPick() {
    if (!top_images_ || !top_images_->picked_item || ImGui::GetFrameCount() - top_images_->shown_frame > 1) return false;
    TopImagesState& s = *top_images_;
    const size_t item = *s.picked_item;
    if (item >= s.strengths.items || s.picked_channel >= s.strengths.channels) return false;
    const auto& facts = ShownSampleFacts();
    const auto class_names = facts.class_names.empty() ? s.items.class_names : facts.class_names;
    if (BeginCard("TopImagesPick")) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerMuted(), "Top image");
        const auto found = s.thumbs.find(item);
        if (found != s.thumbs.end() && found->second) {
            const float side = std::min(kInspectorPicture, ImGui::GetContentRegionAvail().x);
            ImGui::Image(static_cast<ImTextureID>(found->second), ImVec2(side, side));
        }
        const std::string name = item < s.items.names.size() && !s.items.names[item].empty()
                                     ? s.items.names[item]
                                     : (item < s.items.indices.size() ? "item " + std::to_string(s.items.indices[item])
                                                                      : std::string());
        ImGui::TextColored(DebuggerBright(), "%s", name.c_str());
        const std::string label = item < s.labels.size() && s.labels[item] >= 0 ? lv::ClassName(s.labels[item], class_names)
                                                                                 : std::string();
        ImGui::Text("%s%srank %zu of %zu", label.empty() ? "" : ("label " + label).c_str(),
                    label.empty() ? "" : " \xC2\xB7 ", s.picked_rank, s.strengths.items);
        ImGui::TextColored(DebuggerMuted(), "Channel %zu's %s: %s", s.picked_channel,
                           s.mean ? "mean over its map" : "maximum over its map",
                           ti::ValueText(s.strengths.At(s.picked_channel, item)).c_str());
        ImGui::Spacing();
        if (item < s.items.indices.size()) {
            if (ui::SecondaryButton("Make it the picked sample")) {
                selected_sample_index_ = static_cast<int>(s.items.indices[item]);
                picked_from_top_images_ = name;
            }
            if (!picked_from_top_images_.empty())
                ImGui::TextColored(DebuggerAccentText(), "Picked (sample %d): run Local Debug to look inside the model "
                                                         "for it.",
                                   selected_sample_index_);
            else
                ImGui::TextColored(DebuggerFaint(), "Then run Local Debug to look inside the model for it.");
        }
        ImGui::PopTextWrapPos();
    }
    EndCard();
    ImGui::Spacing();
    return true;
}

}  // namespace cyxwiz
