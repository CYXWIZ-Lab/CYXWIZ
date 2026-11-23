// Studio Debugger > Model > Layers > Contributions (TOFIX178 6.4, approved
// mockup boards Main and Dense): how a Dense layer's output was computed for
// the picked sample, output = sum of w x x + b, from the captured input and
// weights of that step. The model's head explains an output (the predicted
// class first); a hidden Dense explains a unit (this sample's strongest
// first). A clicked row shows in the Inspector where its input comes from.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/count_text.h"
#include "../../core/dense_contributions.h"
#include "../../core/layer_tensor_reader.h"
#include "../../core/layer_view_presentation.h"
#include "../../core/where_it_looks_presentation.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz {

namespace lv = layer_view;
namespace ct = contributions;

namespace {

constexpr size_t kTopTerms = 8;
constexpr size_t kUnitChoices = 3;  // a hidden layer: its strongest units as buttons

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

}  // namespace

void StudioDebuggerPanel::RenderContributions(const std::string& weight_name) {
    const auto& layers = layer_capture_->Layers();
    const CapturedLayer& layer = layers[selected_layer_];
    const CapturedTensor* input = layer.Find(CapturedTensorRole::Input);
    const CapturedTensor* output = layer.Find(CapturedTensorRole::Output);
    const CapturedTensor* weight = layer.Find(CapturedTensorRole::Parameter, weight_name);
    const CapturedTensor* bias = layer.Find(CapturedTensorRole::Parameter, "bias");
    if (!input || !weight || weight->tensor.Shape().size() != 2) {
        Wrapped(DebuggerMuted(), "The run did not keep this layer's input and weights (Layer capture: keep activations "
                                 "and parameters).");
        return;
    }
    const size_t outputs = weight->tensor.Shape()[0];
    const size_t inputs = weight->tensor.Shape()[1];
    const bool head = &layer == &layers.back();
    const auto& facts = ShownSampleFacts();
    const lv::OutputRule rule = lv::OutputRuleFromKey(facts.output_rule);
    const std::string key = std::to_string(layer_capture_id_) + "/" +
                            std::to_string(selected_layer_) + "/" + std::to_string(selected_layer_sample_);
    if (contributions_key_ != key) {
        contributions_key_ = key;
        contributions_output_ = -1;
        contributions_pick_.reset();
    }

    try {
        const std::vector<float> x = lv::ReadValues(lv::SampleOf(input->tensor, selected_layer_sample_));
        const std::vector<float> y =
            output ? lv::ReadValues(lv::SampleOf(output->tensor, selected_layer_sample_)) : std::vector<float>{};
        if (x.size() != inputs) throw std::runtime_error("the input does not match the weights");

        // Which output (the head) or unit (a hidden layer).
        std::vector<std::string> names;
        if (head) {
            names = where_it_looks::OutputNames(rule, outputs, facts.class_names);
            if (names.size() != outputs) {  // one logit: its name is its class
                names.assign(outputs, "the output");
            }
        }
        std::vector<size_t> by_strength(outputs);
        std::iota(by_strength.begin(), by_strength.end(), size_t{0});
        if (y.size() == outputs)
            std::stable_sort(by_strength.begin(), by_strength.end(), [&](size_t a, size_t b) { return y[a] > y[b]; });
        if (contributions_output_ < 0 || static_cast<size_t>(contributions_output_) >= outputs)
            contributions_output_ = static_cast<int>(by_strength.front());
        const size_t picked = static_cast<size_t>(contributions_output_);
        const auto output_name = [&](size_t o) {
            return head ? (outputs == 1 ? std::string("the logit") : names[o] + " logit") : "unit " + std::to_string(o);
        };

        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(DebuggerMuted(), head ? "Explain" : "Unit");
        ImGui::SameLine(80.0f);
        std::vector<std::string> labels;
        std::vector<size_t> choices;
        for (size_t i = 0; i < std::min(outputs, head ? outputs : kUnitChoices); ++i) {
            const size_t o = by_strength[i];
            choices.push_back(o);
            labels.push_back(head ? (i == 0 && outputs > 1 ? "Predicted: " + names[o] : (outputs == 1 ? "The output" : names[o]))
                                  : std::to_string(o));
        }
        if (head && choices.size() > 6) {  // many classes: the first few as buttons
            choices.resize(6);
            labels.resize(6);
        }
        std::vector<const char*> texts;
        for (const auto& l : labels) texts.push_back(l.c_str());
        int choice = static_cast<int>(std::find(choices.begin(), choices.end(), picked) - choices.begin());
        if (choice >= static_cast<int>(choices.size())) choice = -1;
        if (ui::SegmentedControl("ContributionsOutput", texts.data(), static_cast<int>(texts.size()), &choice) && choice >= 0)
            contributions_output_ = static_cast<int>(choices[static_cast<size_t>(choice)]);
        if (outputs > choices.size()) {
            ImGui::SameLine(0.0f, 10.0f);
            const std::string preview = choice < 0 ? output_name(picked) : (head ? "Another class" : "Unit");
            ImGui::SetNextItemWidth(ImGui::CalcTextSize(preview.c_str()).x + ImGui::GetFrameHeight() * 2.0f);
            if (ImGui::BeginCombo("##ContributionsMore", preview.c_str())) {
                for (size_t o = 0; o < outputs; ++o) {
                    const std::string text = head ? names[o] : "unit " + std::to_string(o) +
                                                                   (y.size() == outputs ? " \xC2\xB7 " + Fixed(y[o], 3) : "");
                    if (ImGui::Selectable(text.c_str(), o == picked)) contributions_output_ = static_cast<int>(o);
                }
                ImGui::EndCombo();
            }
        }
        ImGui::SameLine(0.0f, 12.0f);
        Wrapped(DebuggerMuted(), head ? "Which output: the predicted class first, then the others."
                                      : "A hidden layer has no classes: its strongest units for this sample first, any "
                                        "unit from the list.");

        // The terms.
        const std::vector<float> b = bias ? lv::ReadValues(bias->tensor) : std::vector<float>{};
        const auto e = ct::Explain(ct::MatrixRow(weight->tensor, picked), x, picked < b.size() ? b[picked] : 0.0, kTopTerms);
        const CapturedLayer* source = selected_layer_ > 0 ? &layers[selected_layer_ - 1] : nullptr;
        // The layer the inputs were flattened from: the nearest earlier layer
        // with a [H, W, C] output of as many values.
        std::vector<size_t> source_shape;
        std::string source_name;
        for (size_t i = selected_layer_; i-- > 0;) {
            const CapturedTensor* out = layers[i].Find(CapturedTensorRole::Output);
            if (!out) continue;
            const auto per = lv::SampleOf(out->tensor, 0).Shape();
            size_t elements = 1;
            for (size_t d : per) elements *= d;
            if (per.size() == 3 && elements == inputs) {
                source_shape = per;
                source_name = layers[i].node_name;
                break;
            }
            if (elements != inputs) break;  // a layer of another size: no flatten chain further back
        }
        const std::string from = source ? source->node_name : std::string("the input");
        Wrapped(DebuggerMuted(), "How " + output_name(picked) + " was computed for this sample: " +
                                     (head && outputs == 1 ? std::string("logit") : output_name(picked)) +
                                     " = the sum of w \xC3\x97 x + b over " +
                                     CountText(inputs) + " inputs (" + from + ")" +
                                     (head ? (rule == lv::OutputRule::Softmax ? ", then softmax over " + std::to_string(outputs) + " classes"
                                              : rule == lv::OutputRule::Sigmoid ? std::string(", then sigmoid")
                                                                                : std::string())
                                           : std::string()) +
                                     ". The largest contributions first; click a row for where its input comes from.");
        ImGui::Spacing();
        const ui::Tokens& t = ui::CurrentTokens();
        const float top_abs = e.top.empty() ? 1.0f : std::max(1e-12f, std::fabs(e.top.front().wx));
        // Fixed columns for the numbers; the bars take what is left (nothing
        // past the edge), the value after each bar.
        if (ImGui::BeginTable("##contributions", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Input", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("input 00000").x + 16.0f);
            ImGui::TableSetupColumn("w", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("+0.0000").x + 16.0f);
            ImGui::TableSetupColumn("x", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000.000").x + 16.0f);
            ImGui::TableSetupColumn("w \xC3\x97 x", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (const auto& term : e.top) {
                ImGui::PushID(static_cast<int>(term.input));
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                const bool selected = contributions_pick_ == term.input;
                if (ImGui::Selectable(("input " + std::to_string(term.input)).c_str(), selected,
                                      ImGuiSelectableFlags_SpanAllColumns))
                    contributions_pick_ = term.input;
                ImGui::TableNextColumn();
                ImGui::TextColored(DebuggerMuted(), "%s", ct::Signed(term.w).c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(DebuggerMuted(), "%s", Fixed(term.x, 3).c_str());
                ImGui::TableNextColumn();
                const std::string value = ct::Signed(term.wx);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float room = std::max(20.0f, ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("+0.0000").x - 12.0f);
                const float width = room * std::fabs(term.wx) / top_abs;
                const float h = ImGui::GetTextLineHeight();
                ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x, at.y + h * 0.2f), ImVec2(at.x + width, at.y + h * 0.8f),
                                                          ui::ToU32(ui::DivergingAt(t, term.wx < 0 ? -0.9f : 0.9f)), 3.0f);
                ImGui::Dummy(ImVec2(room + 8.0f, h));
                ImGui::SameLine(0.0f, 0.0f);
                ImGui::TextUnformatted(value.c_str());
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        // The sum, and what follows it.
        std::string line = "The sum of w \xC3\x97 x = " + Fixed(e.sum, 3) + ", b = " + Fixed(e.bias, 3) + ", " +
                           output_name(picked) + " = " + Fixed(e.value, 3);
        if (head && y.size() == outputs) {
            for (size_t o = 0; o < outputs && outputs > 1; ++o)
                if (o != picked) line += "; " + output_name(o) + " = " + Fixed(y[o], 3);
            const std::vector<float>* eval = EvalModelOutput();
            if (const auto p = lv::Predict(eval ? *eval : y, rule, facts.class_names))
                line += " \xE2\x86\x92 " + p->text + (eval ? " (the evaluation pass)" : "");
        } else if (!head && selected_layer_ + 1 < layers.size()) {
            const CapturedLayer& next = layers[selected_layer_ + 1];
            if (const CapturedTensor* next_out = next.Find(CapturedTensorRole::Output)) {
                const auto after = lv::ReadValues(lv::SampleOf(next_out->tensor, selected_layer_sample_));
                if (after.size() == outputs) line += " \xE2\x86\x92 " + next.node_name + " " + Fixed(after[picked], 3);
            }
        }
        if (e.rest > 0)
            line += ". " + CountText(e.rest) + " more inputs contribute " + Fixed(e.rest_sum, 3) + ".";
        ImGui::Spacing();
        BeginCard("ContributionsSum");
        Wrapped(DebuggerText(), line);
        EndCard();
        Wrapped(DebuggerFaint(), "The captured input and weights of this step: the same numbers the forward used. Blue "
                                 "pulls the output down, red pushes it up.");

        // The Inspector's note for the clicked row.
        contributions_inspector_.clear();
        if (contributions_pick_) {
            const size_t i = *contributions_pick_;
            const std::string place = ct::InputPlace(i, source_shape);
            contributions_inspector_ = "Input " + std::to_string(i) + " of " + CountText(inputs) +
                                       (place.empty() ? "" : ": " + source_name + " at " + place +
                                                                 " (the Flatten keeps H, W, C order)") +
                                       ". Its weight towards " + output_name(picked) + " is " +
                                       ct::Signed(ct::MatrixRow(weight->tensor, picked)[i]) + "; this sample's value " +
                                       Fixed(x[i], 3) + ".";
        }
    } catch (const std::exception& e) {
        Wrapped(DebuggerMuted(), std::string("This view could not be read: ") + e.what());
    }
}

}  // namespace cyxwiz
