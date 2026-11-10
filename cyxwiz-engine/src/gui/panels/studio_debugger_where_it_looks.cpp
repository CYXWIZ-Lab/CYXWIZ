// Studio Debugger > Model > Layers > Where it looks (TOFIX178 6.3b, approved
// mockup board WhereItLooks): the run's model rebuilt from its capture
// (DebugModelSession, TOFIX202) in eval mode, its prediction for the picked
// item (owner decision 2026-11-09: the shown prediction and occlusion use
// the same evaluation pass), and two methods: occlusion of the input (image
// patches, tokens or features, each covered in turn, the picked output's
// score change drawn over the input) and saliency (6.3c, approved board
// Saliency: the gradient of the picked score with respect to the input,
// its size or gradient x input). All model work runs off the UI thread.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/async_task_manager.h"
#include "../../core/debug_model_session.h"
#include "../../core/layer_tensor_reader.h"
#include "../../core/layer_view_presentation.h"
#include "../../core/live_graph_compile.h"
#include "../../core/occlusion.h"
#include "../../core/saliency.h"
#include "../../core/where_it_looks_presentation.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace cyxwiz {

namespace lv = layer_view;
namespace wl = where_it_looks;

namespace {

constexpr size_t kOcclusionChunk = 64;          // covered inputs a forward
constexpr size_t kChunkElements = size_t{8} << 20;  // ... and at most this many values
constexpr size_t kFeatureMeanRows = 512;            // a table's fill: the mean over this many rows

std::vector<float> HostRowMajor(const Tensor& t) {
    const af::array flat = t.GetRowMajorFlatArray();
    std::vector<float> values(static_cast<size_t>(flat.elements()));
    if (!values.empty()) flat.as(f32).host(values.data());
    return values;
}

// One item of the model's input, with its batch dimension of 1.
Tensor ItemInput(const Tensor& input, size_t item, occlusion::InputKind kind) {
    Tensor one = lv::SampleOf(input, item);
    std::vector<size_t> shape = one.Shape();
    if (kind == occlusion::InputKind::Image)
        shape.push_back(1);
    else
        shape.insert(shape.begin(), 1);
    return Tensor::FromRowMajorFlatArray(one.GetRowMajorFlatArray(), shape);
}

void WrappedMuted(const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(DebuggerMuted(), "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void WrappedFaint(const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(DebuggerFaint(), "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void WrappedText(const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(DebuggerText(), "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// The diverging scale (blue, the surface, red) between its two labels.
void DrawDivergingLegend(const std::string& low, const std::string& high) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(DebuggerMuted(), "%s", low.c_str());
    ImGui::SameLine();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float width = 160.0f, height = 10.0f;
    const float y = at.y + (ImGui::GetTextLineHeight() - height) * 0.5f;
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
    constexpr int kSteps = 48;
    for (int i = 0; i < kSteps; ++i) {
        const float v = static_cast<float>(i) / (kSteps - 1) * 2.0f - 1.0f;
        const float x0 = at.x + width * i / kSteps, x1 = at.x + width * (i + 1) / kSteps;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(x0, y), ImVec2(x1, y + height), ui::ToU32(ui::DivergingAt(t, v)));
    }
    ImGui::SameLine();
    ImGui::TextColored(DebuggerMuted(), "%s", high.c_str());
}

// The Heat scale (the feature maps': black through red to yellow) between its labels.
void DrawHeatLegend(const std::string& low, const std::string& high) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(DebuggerMuted(), "%s", low.c_str());
    ImGui::SameLine();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float width = 160.0f, height = 10.0f;
    const float y = at.y + (ImGui::GetTextLineHeight() - height) * 0.5f;
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
    constexpr int kSteps = 48;
    for (int i = 0; i < kSteps; ++i) {
        const float v = static_cast<float>(i) / (kSteps - 1);
        const float x0 = at.x + width * i / kSteps, x1 = at.x + width * (i + 1) / kSteps;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(x0, y), ImVec2(x1, y + height), ui::ToU32(ui::HeatAt(t, v)));
    }
    ImGui::SameLine();
    ImGui::TextColored(DebuggerMuted(), "%s", high.c_str());
}

}  // namespace

struct StudioDebuggerPanel::WhereItLooksState {
    // The rebuilt model, for one capture.
    const LayerCapture* capture = nullptr;
    std::shared_ptr<DebugModelSession> session;
    bool building = false;
    std::string unavailable;  // why the model cannot be rebuilt
    // The canvas had no clean compile when the capture opened: try again once
    // it has one (the first compile after opening a project, or a fix).
    bool waiting_for_clean = false;

    // The evaluation pass for the picked item.
    size_t eval_item = std::numeric_limits<size_t>::max();
    std::vector<float> eval_output;
    bool evaluating = false;

    // The occlusion of one item and target.
    int target = -1;  // output index; -1 = the predicted class
    bool computing = false;
    std::shared_ptr<std::atomic<size_t>> done = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::atomic<size_t>> total = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::atomic<bool>> cancel = std::make_shared<std::atomic<bool>>(false);
    std::string result_key;  // capture / item / target the result is for
    occlusion::Plan plan;
    std::vector<occlusion::Piece> pieces;
    occlusion::Result result;
    std::vector<float> item_values;  // tokens: ids; features: values
    std::string fill_source;
    std::string headline;
    std::string error;

    // The saliency of one item and target (6.3c).
    std::string saliency_key;  // capture / item / target the result is for
    saliency::Result saliency;

    bool Busy() const { return building || evaluating || computing; }
};

void StudioDebuggerPanel::UpdateLayerModelSession() {
    if (!where_it_looks_) where_it_looks_ = std::make_shared<WhereItLooksState>();
    WhereItLooksState& s = *where_it_looks_;
    if (!layer_capture_ || layer_capture_->Layers().empty()) return;
    // Why there is no clean compile to rebuild from; it changes while the view waits.
    const auto no_clean_reason = [this]() -> std::string {
        if (!live_compile_ || live_compile_->CanvasEmpty())
            return "no graph is open on the canvas; open the run's graph and the model is rebuilt from it";
        if (!live_compile_->UpToDate())
            return "the canvas's graph is still compiling; the model is rebuilt from it when it finishes";
        return "the canvas's graph does not compile cleanly now; the run's model is rebuilt from it";
    };
    if (s.waiting_for_clean && !s.Busy()) {
        if (live_compile_ && live_compile_->CleanConfig())
            s.capture = nullptr;  // rebuild below
        else
            s.unavailable = no_clean_reason();
    }

    if (s.capture != layer_capture_.get() && !s.Busy()) {
        s.capture = layer_capture_.get();
        s.session.reset();
        s.unavailable.clear();
        s.waiting_for_clean = false;
        s.eval_item = std::numeric_limits<size_t>::max();
        s.eval_output.clear();
        s.result_key.clear();
        s.target = -1;
        // The canvas's clean compile is the run's graph when its layers are
        // the captured ones (FromCapture checks node by node).
        const TrainingConfiguration* clean = live_compile_ ? live_compile_->CleanConfig() : nullptr;
        if (!clean) {
            s.waiting_for_clean = true;
            s.unavailable = no_clean_reason();
            return;
        }
        s.building = true;
        struct Built {
            std::shared_ptr<DebugModelSession> session;
            std::string error;
        };
        auto built = std::make_shared<Built>();
        auto capture = layer_capture_;
        auto state = where_it_looks_;
        AsyncTaskManager::Instance().RunAsync(
            "Studio Debugger: rebuild the run's model",
            [built, capture, config = *clean](LambdaTask&) {
                try {
                    built->session = DebugModelSession::FromCapture(config, *capture, &built->error);
                } catch (const std::exception& e) {
                    built->error = e.what();
                }
            },
            nullptr,
            [state, built, key = capture.get()](bool, const std::string&) {
                state->building = false;
                if (state->capture != key) return;
                state->session = built->session;
                if (!built->session) state->unavailable = built->error.empty() ? "it could not be rebuilt" : built->error;
            });
        return;
    }

    // The evaluation pass for the picked item.
    if (!s.session || s.Busy() || s.eval_item == selected_layer_sample_) return;
    const CapturedLayer& first = layer_capture_->Layers().front();
    const CapturedTensor* input = first.Find(CapturedTensorRole::Input);
    if (!input || first.module_index != 0) {
        s.unavailable = "the run did not keep the model's input";
        return;
    }
    s.evaluating = true;
    s.eval_item = selected_layer_sample_;
    auto output = std::make_shared<std::vector<float>>();
    auto state = where_it_looks_;
    const Tensor input_tensor = input->tensor;
    const size_t item = selected_layer_sample_;
    const bool spatial = input_tensor.Shape().size() == 4;
    AsyncTaskManager::Instance().RunAsync(
        "Studio Debugger: evaluation pass",
        [state, output, input_tensor, item, spatial](LambdaTask&) {
            try {
                const Tensor one = ItemInput(input_tensor, item,
                                             spatial ? occlusion::InputKind::Image : occlusion::InputKind::Features);
                *output = HostRowMajor(state->session->Forward(one));
            } catch (const std::exception& e) {
                spdlog::warn("Studio Debugger: evaluation pass failed: {}", e.what());
            }
        },
        nullptr,
        [state, output, item](bool, const std::string&) {
            state->evaluating = false;
            if (state->eval_item == item) state->eval_output = std::move(*output);
        });
}

std::shared_ptr<DebugModelSession> StudioDebuggerPanel::LayerModelSession() const {
    if (!where_it_looks_ || where_it_looks_->capture != layer_capture_.get()) return nullptr;
    return where_it_looks_->session;
}

const std::vector<float>* StudioDebuggerPanel::EvalModelOutput() const {
    if (!where_it_looks_ || where_it_looks_->capture != layer_capture_.get() ||
        where_it_looks_->eval_item != selected_layer_sample_ || where_it_looks_->eval_output.empty())
        return nullptr;
    return &where_it_looks_->eval_output;
}

std::string StudioDebuggerPanel::EvalModelUnavailable() const {
    if (!where_it_looks_ || where_it_looks_->capture != layer_capture_.get()) return {};
    return where_it_looks_->unavailable;
}

void StudioDebuggerPanel::RenderWhereItLooks(uint32_t sample_texture, int sample_width, int sample_height) {
    UpdateLayerModelSession();
    WhereItLooksState& s = *where_it_looks_;
    const CapturedLayer& first = layer_capture_->Layers().front();
    const CapturedTensor* input = first.Find(CapturedTensorRole::Input);
    const StudioDebuggerSampleFacts& facts = ShownSampleFacts();
    const lv::OutputRule rule = lv::OutputRuleFromKey(facts.output_rule);

    if (!s.unavailable.empty()) {
        WrappedMuted("Where it looks runs the run's model again, and it cannot be rebuilt: " + s.unavailable + ".");
        return;
    }
    if (s.building || (!s.session && !s.unavailable.empty()) || !s.session) {
        WrappedMuted("Rebuilding the run's model from its capture...");
        return;
    }
    if (!input) {
        WrappedMuted("The run did not keep the model's input (Layer capture: keep activations).");
        return;
    }

    // What the input is.
    std::vector<size_t> item_shape = lv::SampleOf(input->tensor, 0).Shape();
    // Token ids enter an Embedding: the first layer keeps an embedding table
    // and the graph is a text graph.
    const bool embeds = session_.graph_domain == "Text";
    if (item_shape.size() == 3)
        item_shape.push_back(1);
    else
        item_shape.insert(item_shape.begin(), 1);
    const auto kind = occlusion::KindOf(item_shape, embeds);
    if (!kind) {
        WrappedMuted("Where it looks covers an image, tokens or table features; this model's input (" +
                     lv::ShapeText(item_shape) + ") is none of them yet.");
        return;
    }

    // Explain: the outputs, predicted and true first.
    const std::vector<float>* eval = EvalModelOutput();
    if (!eval) {
        WrappedMuted("Reading the model's prediction (evaluation pass)...");
        return;
    }
    const auto names = wl::OutputNames(rule, eval->size(), facts.class_names);
    const auto prediction = lv::Predict(*eval, rule, facts.class_names);
    const int predicted = prediction ? prediction->class_index : -1;
    const int truth = selected_layer_sample_ < facts.labels.size() ? facts.labels[selected_layer_sample_] : -1;
    const auto targets = wl::Targets(names, predicted, truth);
    if (targets.empty()) {
        WrappedMuted("The model's output has nothing to explain.");
        return;
    }
    if (s.target < 0 || static_cast<size_t>(s.target) >= names.size()) s.target = targets.front().index;

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), "Explain");
    ImGui::SameLine(110.0f);
    {
        // Predicted and true as a switch; any other output from a list.
        const size_t lead = std::min<size_t>(targets.size(), truth >= 0 && truth != predicted ? 2 : 1);
        std::vector<const char*> labels;
        int picked = -1;
        for (size_t i = 0; i < lead; ++i) {
            labels.push_back(targets[i].label.c_str());
            if (targets[i].index == s.target) picked = static_cast<int>(i);
        }
        int choice = picked;
        if (ui::SegmentedControl("WhereItLooksTarget", labels.data(), static_cast<int>(labels.size()), &choice) &&
            choice >= 0 && !s.computing)
            s.target = targets[static_cast<size_t>(choice)].index;
        if (targets.size() > lead) {
            ImGui::SameLine(0.0f, 12.0f);
            const std::string preview = picked < 0 ? names[static_cast<size_t>(s.target)] : "Another class";
            ImGui::SetNextItemWidth(ImGui::CalcTextSize(preview.c_str()).x + ImGui::GetFrameHeight() * 2.0f);
            if (ImGui::BeginCombo("##WhereItLooksOther", preview.c_str())) {
                for (size_t i = lead; i < targets.size(); ++i)
                    if (ImGui::Selectable(targets[i].label.c_str(), targets[i].index == s.target) && !s.computing)
                        s.target = targets[i].index;
                ImGui::EndCombo();
            }
        }
    }
    ImGui::Indent(110.0f);
    WrappedFaint("Later: a box or a mask region, and the next token, once the model has those heads.");
    ImGui::Unindent(110.0f);

    // Method and, for saliency, what it shows: this session's choice, from
    // the user's defaults (Preferences > Studio Debugger).
    const core::StudioDebuggerConfig settings = CaptureSettings();
    const bool use_saliency = settings.where_it_looks_method == core::WhereItLooksMethod::Saliency;
    const auto set_session = [&](const std::function<void(core::StudioDebuggerConfig&)>& change) {
        if (!capture_session_) capture_session_ = core::EngineConfig::Instance().GetStudioDebugger();
        change(*capture_session_);
    };
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), "Method");
    ImGui::SameLine(110.0f);
    {
        static const char* const kMethods[] = {"Occlusion", "Saliency"};
        int method = use_saliency ? 1 : 0;
        if (ui::SegmentedControl("WhereItLooksMethod", kMethods, 2, &method) && !s.computing)
            set_session([&](core::StudioDebuggerConfig& c) {
                c.where_it_looks_method = method == 1 ? core::WhereItLooksMethod::Saliency : core::WhereItLooksMethod::Occlusion;
            });
        ImGui::SameLine(0.0f, 12.0f);
        WrappedMuted(use_saliency ? (*kind == occlusion::InputKind::Tokens
                                         ? "Token ids have no gradient: it is taken at the Embedding's output, one "
                                           "vector per token."
                                         : "One forward and one backward pass, on the device.")
                                  : "Forward passes only: works for any model on the device.");
    }
    if (use_saliency) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "Show");
        ImGui::SameLine(110.0f);
        static const char* const kShows[] = {"Gradient size", "Gradient \xC3\x97 input"};
        int show = settings.saliency_show == core::SaliencyShow::TimesInput ? 1 : 0;
        if (ui::SegmentedControl("WhereItLooksShow", kShows, 2, &show))
            set_session([&](core::StudioDebuggerConfig& c) {
                c.saliency_show = show == 1 ? core::SaliencyShow::TimesInput : core::SaliencyShow::Size;
            });
        ImGui::SameLine(0.0f, 12.0f);
        const core::StudioDebuggerConfig saved = core::EngineConfig::Instance().GetStudioDebugger();
        const bool differs = saved.where_it_looks_method != settings.where_it_looks_method ||
                             saved.saliency_show != settings.saliency_show;
        if (ui::LinkButton("Save as my default", differs)) {
            core::StudioDebuggerConfig defaults = saved;
            defaults.where_it_looks_method = settings.where_it_looks_method;
            defaults.saliency_show = settings.saliency_show;
            auto& engine_config = core::EngineConfig::Instance();
            engine_config.SetStudioDebugger(defaults);
            engine_config.Save();
        }
        if (!differs && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Saliency and this Show are your defaults");
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), "Of");
        ImGui::SameLine(110.0f);
        WrappedText(wl::SaliencyOfText(rule, eval->size(), s.target, names[static_cast<size_t>(s.target)]));
    }

    // The plan for this item.
    occlusion::Plan plan;
    plan.kind = *kind;
    plan.sample_shape = item_shape;
    std::string fill_source;
    std::vector<float> item_values;
    if (*kind == occlusion::InputKind::Image) {
        fill_source = "the image's mean per channel";
    } else {
        item_values = HostRowMajor(lv::SampleOf(input->tensor, selected_layer_sample_));
        if (*kind == occlusion::InputKind::Tokens) {
            const TrainingConfiguration* clean = live_compile_ ? live_compile_->CleanConfig() : nullptr;
            const float pad = clean ? static_cast<float>(clean->text_preprocessing.pad_value) : 0.0f;
            plan.fill = {pad};
            fill_source = "the pad id (" + std::to_string(static_cast<long long>(pad)) + ")";
        } else {
            fill_source = "its mean over up to 512 of the graph's prepared rows, spread over the data";
        }
    }
    const float pad_id = plan.fill.empty() ? 0.0f : plan.fill[0];
    if (use_saliency) {
        RenderSaliency(*kind, item_values, pad_id, names[static_cast<size_t>(s.target)], sample_texture, sample_width,
                       sample_height);
        return;
    }
    const auto pieces = occlusion::PiecesOf(plan, item_values, pad_id);
    size_t elements = 1;
    for (size_t d : item_shape) elements *= d;
    const size_t chunk = std::clamp<size_t>(kChunkElements / std::max<size_t>(1, elements), 1, kOcclusionChunk);

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), *kind == occlusion::InputKind::Features ? "Replace with" : "Cover with");
    ImGui::SameLine(110.0f);
    WrappedText(wl::CoverText(plan, fill_source));

    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                            std::to_string(selected_layer_sample_) + "/" + std::to_string(s.target);
    ImGui::Spacing();
    if (s.computing) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerAccentText(), "Computing");
        ImGui::SameLine();
        ImGui::Text("%zu of %zu forward passes", s.done->load(), s.total->load());
        ImGui::SameLine(0.0f, 16.0f);
        if (ui::DangerButton("Stop")) s.cancel->store(true);
    } else {
        const bool have = s.result_key == key;
        if (ui::PrimaryButton(have ? "Compute again" : "Compute", !pieces.empty(), "Nothing in this input to cover")) {
            s.computing = true;
            s.error.clear();
            s.done->store(0);
            s.total->store(pieces.size());
            s.cancel->store(false);
            auto state = where_it_looks_;
            auto result = std::make_shared<occlusion::Result>();
            auto worker_plan = std::make_shared<occlusion::Plan>(plan);
            auto error = std::make_shared<std::string>();
            DataItemsJob items_job;
            if (*kind == occlusion::InputKind::Features) {
                const DataItemsStart start = data_items_callback_ ? data_items_callback_() : DataItemsStart{};
                items_job = start.job;
                if (!items_job) {
                    s.computing = false;
                    s.error = DataItemsNotStarted(start, "the feature means");
                }
            }
            if (s.computing) {
                const Tensor input_tensor = input->tensor;
                const size_t item = selected_layer_sample_;
                const int target = s.target;
                AsyncTaskManager::Instance().RunAsync(
                    "Studio Debugger: where it looks",
                    [state, result, worker_plan, error, input_tensor, item, target, rule, pieces, chunk,
                     items_job](LambdaTask&) {
                        try {
                            const Tensor one = ItemInput(input_tensor, item, worker_plan->kind);
                            if (worker_plan->kind == occlusion::InputKind::Image) {
                                // The fill: the image's mean per channel ([H, W, C] row-major).
                                const auto v = HostRowMajor(one);
                                const size_t c = worker_plan->sample_shape[2];
                                std::vector<double> sum(c, 0.0);
                                for (size_t i = 0; i < v.size(); ++i) sum[i % c] += v[i];
                                worker_plan->fill.assign(c, 0.0f);
                                for (size_t k = 0; k < c; ++k)
                                    worker_plan->fill[k] = static_cast<float>(sum[k] / static_cast<double>(v.size() / c));
                            } else if (worker_plan->kind == occlusion::InputKind::Features) {
                                // Each feature's mean over up to 512 rows spread over the data.
                                const DataItems rows = items_job(kFeatureMeanRows, [&] { return state->cancel->load(); });
                                const auto& shape = rows.features.Shape();
                                if (!rows.why.empty() || shape.size() != 2 || shape[1] != worker_plan->sample_shape[1])
                                    throw std::runtime_error(rows.why.empty() ? "the feature means could not be read"
                                                                              : rows.why);
                                // Rows [N, F] row-major: ArrayFire (F, N); the mean over N.
                                const af::array by_row =
                                    af::moddims(rows.features.GetRowMajorFlatArray().as(f32),
                                                static_cast<dim_t>(shape[1]), static_cast<dim_t>(shape[0]));
                                worker_plan->fill = HostRowMajor(Tensor::FromRowMajorFlatArray(
                                    af::flat(af::mean(by_row, 1)), {shape[1]}));
                            }
                            *result = occlusion::Run(
                                *state->session, one, *worker_plan, pieces, rule, target, chunk,
                                [state](size_t done, size_t) { state->done->store(done); },
                                [state] { return state->cancel->load(); });
                        } catch (const std::exception& e) {
                            *error = e.what();
                        }
                    },
                    nullptr,
                    [state, result, worker_plan, error, key, pieces, item_values](bool, const std::string&) {
                        state->computing = false;
                        if (!error->empty()) {
                            state->error = *error;
                            return;
                        }
                        if (result->cancelled) return;
                        state->result_key = key;
                        state->plan = *worker_plan;
                        state->pieces = pieces;
                        state->result = *result;
                        state->item_values = item_values;
                    });
            }
        }
        WrappedMuted(wl::PassesText(plan, pieces.size(), chunk));
    }
    if (!s.error.empty()) WrappedMuted("Not computed: " + s.error + ".");
    if (s.result_key != key) return;

    // The result.
    const std::string target_name = names[static_cast<size_t>(s.target)];
    std::vector<std::string> piece_names;
    if (s.plan.kind == occlusion::InputKind::Tokens) {
        for (float id : s.item_values) piece_names.push_back("#" + std::to_string(static_cast<long long>(id)));
    } else if (s.plan.kind == occlusion::InputKind::Features) {
        for (size_t i = 0; i < s.item_values.size(); ++i) piece_names.push_back("feature " + std::to_string(i));
    }
    ImGui::Spacing();
    WrappedText(wl::Headline(s.plan, s.pieces, s.result, target_name, piece_names));
    const auto scaled = wl::Scaled(s.result.change);
    const ui::Tokens& t = ui::CurrentTokens();
    const auto colour = [&](float v, float alpha) {
        ImVec4 c = ui::DivergingAt(t, v >= 0.0f ? 1.0f : -1.0f);
        c.w = alpha * std::min(1.0f, std::fabs(v));
        return ui::ToU32(c);
    };
    ImDrawList* dl = ImGui::GetWindowDrawList();

    if (s.plan.kind == occlusion::InputKind::Image) {
        const auto [grid_rows, grid_columns] = occlusion::PatchGrid(s.plan);
        const float side = std::min(280.0f, ImGui::GetContentRegionAvail().x * 0.45f);
        const float h = sample_width > 0 ? side * static_cast<float>(sample_height) / static_cast<float>(sample_width) : side;
        if (sample_texture) {
            ImGui::Image(static_cast<ImTextureID>(sample_texture), ImVec2(side, h));
            ImGui::SameLine(0.0f, 24.0f);
        }
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##WhereItLooksHeat", ImVec2(side, h));
        // The picture dimmed under the heat (as the mockup), so the colours read.
        if (sample_texture) dl->AddImage(static_cast<ImTextureID>(sample_texture), at, ImVec2(at.x + side, at.y + h));
        dl->AddRectFilled(at, ImVec2(at.x + side, at.y + h), DebuggerU32(DebuggerPanelBg(), 0.55f));
        const float cw = side / static_cast<float>(std::max<size_t>(1, grid_columns));
        const float ch = h / static_cast<float>(std::max<size_t>(1, grid_rows));
        for (size_t i = 0; i < scaled.size(); ++i) {
            const float x = at.x + cw * static_cast<float>(i % grid_columns);
            const float y = at.y + ch * static_cast<float>(i / grid_columns);
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + cw, y + ch), colour(scaled[i], 0.85f));
        }
        if (ImGui::IsItemHovered() && !scaled.empty()) {
            const ImVec2 m = ImGui::GetIO().MousePos;
            const size_t col = std::min(grid_columns - 1, static_cast<size_t>(std::max(0.0f, (m.x - at.x) / cw)));
            const size_t row = std::min(grid_rows - 1, static_cast<size_t>(std::max(0.0f, (m.y - at.y) / ch)));
            const size_t i = std::min(s.pieces.size() - 1, row * grid_columns + col);
            ImGui::SetTooltip("Covering %s: %s", wl::PieceText(s.plan, s.pieces[i], {}).c_str(),
                              wl::ChangeText(s.result.change[i]).c_str());
        }
    } else if (s.plan.kind == occlusion::InputKind::Tokens) {
        // Every token as a chip in order; covered ones coloured by their change.
        std::vector<int> piece_of(s.item_values.size(), -1);
        for (size_t i = 0; i < s.pieces.size(); ++i) piece_of[s.pieces[i].position] = static_cast<int>(i);
        const float pad_x = 8.0f, pad_y = 4.0f;
        for (size_t p = 0; p < s.item_values.size(); ++p) {
            if (piece_of[p] < 0) continue;  // padding
            const std::string text = piece_names[p];
            const ImVec2 size(ImGui::CalcTextSize(text.c_str()).x + pad_x * 2, ImGui::GetTextLineHeight() + pad_y * 2);
            SameLineIfFits(size.x, 6.0f);
            ImGui::PushID(static_cast<int>(p));
            const ImVec2 at = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("##token", size);
            const size_t i = static_cast<size_t>(piece_of[p]);
            dl->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), DebuggerU32(DebuggerFaint(), 0.25f), 4.0f);
            dl->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), colour(scaled[i], 0.9f), 4.0f);
            dl->AddText(ImVec2(at.x + pad_x, at.y + pad_y), DebuggerU32(DebuggerBright()), text.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Token %zu (id %s): %s when replaced", p, text.c_str() + 1,
                                  wl::ChangeText(s.result.change[i]).c_str());
            ImGui::PopID();
        }
        ImGui::NewLine();
        WrappedMuted("Tokens are shown by their id in the vocabulary; padding is left out.");
    } else {
        if (ImGui::BeginTable("##WhereItLooksFeatures", 4, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Feature");
            ImGui::TableSetupColumn("Value");
            ImGui::TableSetupColumn("Score change when replaced", ImGuiTableColumnFlags_WidthFixed, 420.0f);
            ImGui::TableSetupColumn("Change");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < s.pieces.size(); ++i) {
                const size_t p = s.pieces[i].position;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(piece_names[p].c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.4g", s.item_values[p]);
                ImGui::TableNextColumn();
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float width = 420.0f, mid = at.x + width * 0.5f, hh = ImGui::GetTextLineHeight();
                ImGui::Dummy(ImVec2(width, hh));
                dl->AddRectFilled(at, ImVec2(at.x + width, at.y + hh), DebuggerU32(DebuggerFaint(), 0.2f), 3.0f);
                const float len = width * 0.5f * std::fabs(scaled[i]);
                const float x0 = scaled[i] < 0.0f ? mid - len : mid, x1 = scaled[i] < 0.0f ? mid : mid + len;
                dl->AddRectFilled(ImVec2(x0, at.y), ImVec2(x1, at.y + hh), colour(scaled[i] < 0 ? -1.0f : 1.0f, 1.0f), 3.0f);
                dl->AddLine(ImVec2(mid, at.y - 2), ImVec2(mid, at.y + hh + 2), DebuggerU32(DebuggerMuted(), 0.8f));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(wl::ChangeText(s.result.change[i]).c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::Spacing();
    DrawDivergingLegend("lowers " + target_name, "raises");
    WrappedMuted("Blue: covering it lowers the score, so the model relies on it. Red: covering it raises the score. "
                 "Scores come from the evaluation pass (dropout off, BatchNorm on its running statistics).");
}

void StudioDebuggerPanel::RenderSaliency(occlusion::InputKind kind, const std::vector<float>& item_values,
                                         float pad_id, const std::string& target_name, uint32_t sample_texture,
                                         int sample_width, int sample_height) {
    WhereItLooksState& s = *where_it_looks_;
    const CapturedLayer& first = layer_capture_->Layers().front();
    const CapturedTensor* input = first.Find(CapturedTensorRole::Input);
    const lv::OutputRule rule = lv::OutputRuleFromKey(ShownSampleFacts().output_rule);
    const bool times_input = CaptureSettings().saliency_show == core::SaliencyShow::TimesInput;
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(layer_capture_.get())) + "/" +
                            std::to_string(selected_layer_sample_) + "/" + std::to_string(s.target);

    ImGui::Spacing();
    if (s.computing) {
        ImGui::TextColored(DebuggerAccentText(), "Computing");
    } else {
        if (ui::PrimaryButton(s.saliency_key == key ? "Compute again" : "Compute")) {
            s.computing = true;
            s.error.clear();
            auto state = where_it_looks_;
            auto result = std::make_shared<saliency::Result>();
            auto error = std::make_shared<std::string>();
            const Tensor input_tensor = input->tensor;
            const size_t item = selected_layer_sample_;
            const int target = s.target;
            const size_t embedding_layer = first.module_index;
            AsyncTaskManager::Instance().RunAsync(
                "Studio Debugger: saliency",
                [state, result, error, input_tensor, item, target, rule, kind, embedding_layer](LambdaTask&) {
                    try {
                        *result = saliency::Run(*state->session, ItemInput(input_tensor, item, kind), kind, rule,
                                                target, embedding_layer);
                    } catch (const std::exception& e) {
                        *error = e.what();
                    }
                },
                nullptr,
                [state, result, error, key](bool, const std::string&) {
                    state->computing = false;
                    if (!error->empty()) {
                        state->error = *error;
                        return;
                    }
                    state->saliency_key = key;
                    state->saliency = std::move(*result);
                });
        }
        ImGui::SameLine(0.0f, 12.0f);
        WrappedMuted(wl::SaliencyPassesText());
    }
    if (!s.error.empty()) WrappedMuted("Not computed: " + s.error + ".");
    if (s.saliency_key != key) return;

    // The result: one value per pixel, token or feature.
    const saliency::Result& r = s.saliency;
    const auto& values = times_input ? r.times_input : r.size;
    std::vector<std::string> piece_names;
    std::vector<size_t> considered;
    if (kind == occlusion::InputKind::Tokens) {
        for (size_t p = 0; p < item_values.size(); ++p) {
            piece_names.push_back("#" + std::to_string(static_cast<long long>(item_values[p])));
            if (item_values[p] != pad_id) considered.push_back(p);
        }
    } else if (kind == occlusion::InputKind::Features) {
        for (size_t i = 0; i < item_values.size(); ++i) piece_names.push_back("feature " + std::to_string(i));
    }
    ImGui::Spacing();
    WrappedText(wl::SaliencyHeadline(kind, r, times_input, considered, target_name, piece_names));
    const auto scaled = times_input ? wl::Scaled(values) : wl::ScaledSize(values);
    const ui::Tokens& t = ui::CurrentTokens();
    const auto colour = [&](float v, float alpha) {
        ImVec4 c = times_input ? ui::DivergingAt(t, v >= 0.0f ? 1.0f : -1.0f) : ui::HeatAt(t, v);
        c.w = alpha * std::min(1.0f, times_input ? std::fabs(v) : std::sqrt(v));
        return ui::ToU32(c);
    };
    ImDrawList* dl = ImGui::GetWindowDrawList();

    if (kind == occlusion::InputKind::Image && r.rows > 0 && r.columns > 0 && scaled.size() == r.rows * r.columns) {
        const float side = std::min(280.0f, ImGui::GetContentRegionAvail().x * 0.45f);
        const float h = sample_width > 0 ? side * static_cast<float>(sample_height) / static_cast<float>(sample_width) : side;
        if (sample_texture) {
            ImGui::Image(static_cast<ImTextureID>(sample_texture), ImVec2(side, h));
            ImGui::SameLine(0.0f, 24.0f);
        }
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##SaliencyHeat", ImVec2(side, h));
        if (sample_texture) dl->AddImage(static_cast<ImTextureID>(sample_texture), at, ImVec2(at.x + side, at.y + h));
        dl->AddRectFilled(at, ImVec2(at.x + side, at.y + h), DebuggerU32(DebuggerPanelBg(), 0.65f));
        const float cw = side / static_cast<float>(r.columns), ch = h / static_cast<float>(r.rows);
        for (size_t i = 0; i < scaled.size(); ++i) {
            const float x = at.x + cw * static_cast<float>(i % r.columns);
            const float y = at.y + ch * static_cast<float>(i / r.columns);
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + cw, y + ch), colour(scaled[i], 0.95f));
        }
        if (ImGui::IsItemHovered()) {
            const ImVec2 m = ImGui::GetIO().MousePos;
            const size_t col = std::min(r.columns - 1, static_cast<size_t>(std::max(0.0f, (m.x - at.x) / cw)));
            const size_t row = std::min(r.rows - 1, static_cast<size_t>(std::max(0.0f, (m.y - at.y) / ch)));
            const size_t i = row * r.columns + col;
            ImGui::SetTooltip("Row %zu, column %zu: %s", row, col, wl::SaliencyValueText(values[i], times_input).c_str());
        }
        WrappedMuted(times_input ? "Gradient \xC3\x97 input per pixel, summed over the channels."
                                 : "Gradient size per pixel: the largest over its channels.");
    } else if (kind == occlusion::InputKind::Tokens && values.size() == item_values.size()) {
        const float pad_x = 8.0f, pad_y = 4.0f;
        for (size_t p : considered) {
            const std::string& text = piece_names[p];
            const ImVec2 size(ImGui::CalcTextSize(text.c_str()).x + pad_x * 2, ImGui::GetTextLineHeight() + pad_y * 2);
            SameLineIfFits(size.x, 6.0f);
            ImGui::PushID(static_cast<int>(p));
            const ImVec2 at = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("##token", size);
            dl->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), DebuggerU32(DebuggerFaint(), 0.25f), 4.0f);
            dl->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), colour(scaled[p], 0.9f), 4.0f);
            dl->AddText(ImVec2(at.x + pad_x, at.y + pad_y), DebuggerU32(DebuggerBright()), text.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Token %zu (id %s): %s", p, text.c_str() + 1,
                                  wl::SaliencyValueText(values[p], times_input).c_str());
            ImGui::PopID();
        }
        ImGui::NewLine();
        WrappedMuted(times_input ? "Each chip: the gradient \xC2\xB7 the token's embedding. Tokens by their id; padding left out."
                                 : "Each chip: the length of the gradient at the token's embedding. Tokens by their id; "
                                   "padding left out.");
    } else if (kind == occlusion::InputKind::Features && values.size() == item_values.size()) {
        if (ImGui::BeginTable("##SaliencyFeatures", 4, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Feature");
            ImGui::TableSetupColumn("Value");
            ImGui::TableSetupColumn(times_input ? "Gradient \xC3\x97 value" : "Gradient size",
                                    ImGuiTableColumnFlags_WidthFixed, 420.0f);
            ImGui::TableSetupColumn("");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < values.size(); ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(piece_names[i].c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.4g", item_values[i]);
                ImGui::TableNextColumn();
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float width = 420.0f, hh = ImGui::GetTextLineHeight();
                ImGui::Dummy(ImVec2(width, hh));
                dl->AddRectFilled(at, ImVec2(at.x + width, at.y + hh), DebuggerU32(DebuggerFaint(), 0.2f), 3.0f);
                if (times_input) {
                    const float mid = at.x + width * 0.5f, len = width * 0.5f * std::fabs(scaled[i]);
                    const float x0 = scaled[i] < 0.0f ? mid - len : mid, x1 = scaled[i] < 0.0f ? mid : mid + len;
                    dl->AddRectFilled(ImVec2(x0, at.y), ImVec2(x1, at.y + hh), colour(scaled[i] < 0 ? -1.0f : 1.0f, 1.0f), 3.0f);
                    dl->AddLine(ImVec2(mid, at.y - 2), ImVec2(mid, at.y + hh + 2), DebuggerU32(DebuggerMuted(), 0.8f));
                } else {
                    dl->AddRectFilled(at, ImVec2(at.x + width * scaled[i], at.y + hh), colour(scaled[i], 1.0f), 3.0f);
                }
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(wl::SaliencyValueText(values[i], times_input).c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::Spacing();
    if (times_input) {
        DrawDivergingLegend("lowers " + target_name, "raises");
        WrappedMuted("Blue: this element pulls the score down. Red: it pushes the score up. From the evaluation pass "
                     "(dropout off, BatchNorm on its running statistics).");
    } else {
        DrawHeatLegend("no effect", "strongest");
        WrappedMuted("Bright: a small change here moves the score most. The size says nothing about the direction; "
                     "Gradient \xC3\x97 input does.");
    }
}

}  // namespace cyxwiz
