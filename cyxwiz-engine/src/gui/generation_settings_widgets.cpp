#include "generation_settings_widgets.h"

#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui_stdlib.h>

#include <algorithm>

namespace gui {
namespace {

using cyxwiz::GenerationPreviewDecoding;
using cyxwiz::GenerationPreviewTemplate;

void Title(const char* text) {
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_bright, "%s", text);
}

// A labelled field in a table cell, filling the cell.
template <typename Draw>
bool Cell(const char* label, Draw draw) {
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "%s", label);
    ImGui::SetNextItemWidth(-1.0f);
    return draw();
}

}  // namespace

void GenerationHint(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_faint, "%s", text);
    ImGui::PopTextWrapPos();
}

void GenerationFlowSameLine(float width, float gap) {
    ImGui::SameLine(0.0f, gap);
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

std::string GenerationShown(const std::string& text) {
    std::string shown;
    for (char c : text) shown += c == '\n' ? std::string("\\n") : std::string(1, c);
    return shown;
}

void DrawGenerationSegments(const std::vector<std::pair<std::string, ImVec4>>& segments) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(80.0f, ImGui::GetContentRegionAvail().x);
    const float line_height = ImGui::GetTextLineHeightWithSpacing();
    float x = 0.0f, y = 0.0f;
    for (const auto& [text, colour] : segments) {
        const ImU32 col = ImGui::GetColorU32(colour);
        size_t i = 0;
        while (i < text.size()) {
            if (text[i] == '\n') {
                x = 0.0f;
                y += line_height;
                ++i;
                continue;
            }
            size_t end = i;
            while (end < text.size() && text[end] != '\n' && text[end] != ' ') ++end;
            if (end < text.size() && text[end] == ' ') ++end;  // Keep the space with its word.
            const char* begin = text.c_str() + i;
            const char* stop = text.c_str() + end;
            float w = ImGui::CalcTextSize(begin, stop).x;
            if (x > 0.0f && x + w > width) {
                x = 0.0f;
                y += line_height;
            }
            // A word wider than the line is broken where it fills the line.
            while (w > width && stop > begin + 1) {
                const char* cut = begin;
                while (cut < stop && ImGui::CalcTextSize(begin, cut + 1).x <= width - x) ++cut;
                while (cut > begin + 1 && (static_cast<unsigned char>(*cut) & 0xC0) == 0x80) --cut;
                if (cut <= begin) cut = begin + 1;
                draw->AddText(ImVec2(origin.x + x, origin.y + y), col, begin, cut);
                x = 0.0f;
                y += line_height;
                begin = cut;
                w = ImGui::CalcTextSize(begin, stop).x;
            }
            draw->AddText(ImVec2(origin.x + x, origin.y + y), col, begin, stop);
            x += w;
            i = end;
        }
    }
    ImGui::Dummy(ImVec2(width, y + line_height));
}

bool RenderGenerationTemplateSettings(cyxwiz::TrainingGenerationPreviewSettings& settings, const char* title,
                                      const char* system_note) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Title(title);
    const char* presets[] = {"Role lines", "ChatML", "Llama 3"};
    int preset = static_cast<int>(settings.chat_template);
    if (cyxwiz::ui::SegmentedControl("##gen_template", presets, 3, &preset)) {
        settings.chat_template = static_cast<GenerationPreviewTemplate>(preset);
        changed = true;
    }
    GenerationFlowSameLine(cyxwiz::ui::ButtonWidth("From tokenizer", cyxwiz::ui::ButtonSize::Small), 8.0f);
    cyxwiz::ui::SecondaryButton("From tokenizer", false, "This tokenizer has no chat template");
    switch (settings.chat_template) {
        case GenerationPreviewTemplate::RoleLines:
            GenerationHint("A label before each turn, one turn per line. Use the labels the data was trained with.");
            break;
        case GenerationPreviewTemplate::ChatML:
            GenerationHint("<|im_start|> and <|im_end|> markers. Use it only when the data was built with them.");
            break;
        case GenerationPreviewTemplate::Llama3:
            GenerationHint("Llama 3 header and end-of-turn markers. Use it only when the data was built with them.");
            break;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "System prompt%s%s", system_note && *system_note ? " \xC2\xB7 " : "",
                       system_note ? system_note : "");
    ImGui::PopTextWrapPos();
    if (ImGui::InputTextMultiline("##system_prompt", &settings.system_prompt,
                                  ImVec2(-1.0f, ImGui::GetTextLineHeight() * 4.0f)))
        changed = true;
    if (settings.chat_template == GenerationPreviewTemplate::RoleLines &&
        ImGui::BeginTable("##labels", 3, ImGuiTableFlags_SizingStretchSame)) {
        const std::pair<const char*, std::string*> labels[] = {{"System", &settings.label_system},
                                                               {"User", &settings.label_user},
                                                               {"Assistant", &settings.label_assistant}};
        ImGui::TableNextRow();
        for (int i = 0; i < 3; ++i) {
            ImGui::TableSetColumnIndex(i);
            ImGui::TextColored(t.text_dim, "%s", labels[i].first);
        }
        ImGui::TableNextRow();
        for (int i = 0; i < 3; ++i) {
            ImGui::TableSetColumnIndex(i);
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::PushID(i);
            if (ImGui::InputText("##label", labels[i].second)) changed = true;
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    return changed;
}

bool RenderGenerationStopSettings(cyxwiz::TrainingGenerationPreviewSettings& settings, const char* title,
                                  std::string& new_stop) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Title(title);
    cyxwiz::ui::Chip("End-of-sequence token", t.accent_text);
    for (const auto& stop : cyxwiz::GenerationPreviewTemplateStopTexts(settings)) {
        const std::string label = GenerationShown(stop) + "  (template)";
        GenerationFlowSameLine(cyxwiz::ui::ChipWidth(label.c_str()));
        cyxwiz::ui::Chip(label.c_str(), t.text_dim);
    }
    int remove = -1;
    for (size_t i = 0; i < settings.stop_texts.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        const std::string label = GenerationShown(settings.stop_texts[i]) + "  x";
        GenerationFlowSameLine(cyxwiz::ui::ChipButtonWidth(label.c_str()));
        if (cyxwiz::ui::ChipButton(label.c_str())) remove = static_cast<int>(i);
        cyxwiz::ui::Tooltip("Remove this stop text");
        ImGui::PopID();
    }
    if (remove >= 0) {
        settings.stop_texts.erase(settings.stop_texts.begin() + remove);
        changed = true;
    }
    const float add_width = cyxwiz::ui::ButtonWidth("Add", cyxwiz::ui::ButtonSize::Small);
    ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x - add_width - 8.0f));
    const bool entered = ImGui::InputTextWithHint("##new_stop", "Add a stop text, e.g. </answer>", &new_stop,
                                                  ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    const bool can_add = !new_stop.empty() &&
                         settings.stop_texts.size() < static_cast<size_t>(cyxwiz::kGenerationPreviewMaxStopTexts);
    if ((cyxwiz::ui::SecondaryButton("Add", can_add, "Type a stop text first (at most 8)") || entered) && can_add) {
        settings.stop_texts.push_back(new_stop);
        new_stop.clear();
        changed = true;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Max new tokens");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::InputInt("##max_new_tokens", &settings.max_new_tokens)) {
        settings.max_new_tokens = std::clamp(settings.max_new_tokens, 1, cyxwiz::kGenerationPreviewMaxNewTokens);
        changed = true;
    }
    const char* cap = "1 to 1024, capped by the context the prompt leaves";
    GenerationFlowSameLine(ImGui::CalcTextSize(cap).x);
    ImGui::TextColored(t.text_faint, "%s", cap);
    return changed;
}

bool RenderGenerationDecodingSettings(cyxwiz::TrainingGenerationPreviewSettings& settings, const char* title,
                                      bool samples_each, bool* random_seed) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Title(title);
    const char* decodings[] = {"Greedy", "Sampling", "Beam search"};
    int decoding = static_cast<int>(settings.decoding);
    float segmented = 0.0f;
    for (const char* label : decodings) segmented += ImGui::CalcTextSize(label).x + 24.0f;
    cyxwiz::ui::SameLineRight(segmented);
    if (cyxwiz::ui::SegmentedControl("##gen_decoding", decodings, 3, &decoding)) {
        settings.decoding = static_cast<GenerationPreviewDecoding>(decoding);
        if (settings.decoding != GenerationPreviewDecoding::Sampling) settings.samples_per_prompt = 1;
        changed = true;
    }
    // Fields on one line when there is room, else two per line.
    const auto columns = [](int fields) { return ImGui::GetContentRegionAvail().x >= 130.0f * fields ? fields : 2; };
    if (settings.decoding == GenerationPreviewDecoding::Sampling) {
        const int fields = samples_each ? 6 : 5;
        if (ImGui::BeginTable("##sampling", columns(fields), ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            changed |= Cell("Temperature", [&] {
                return ImGui::InputFloat("##temperature", &settings.temperature, 0.0f, 0.0f, "%.2f");
            });
            ImGui::TableNextColumn();
            changed |= Cell("Top-k", [&] {
                if (!ImGui::InputInt("##top_k", &settings.top_k, 0)) return false;
                settings.top_k = std::max(0, settings.top_k);
                return true;
            });
            ImGui::TableNextColumn();
            changed |= Cell("Top-p", [&] { return ImGui::InputFloat("##top_p", &settings.top_p, 0.0f, 0.0f, "%.2f"); });
            ImGui::TableNextColumn();
            changed |= Cell("Min-p", [&] {
                if (!ImGui::InputFloat("##min_p", &settings.min_p, 0.0f, 0.0f, "%.2f")) return false;
                settings.min_p = std::clamp(settings.min_p, 0.0f, 0.99f);
                return true;
            });
            ImGui::TableNextColumn();
            changed |= Cell("Seed", [&] {
                int seed = static_cast<int>(settings.seed);
                if (!ImGui::InputInt("##seed", &seed, 0)) return false;
                settings.seed = static_cast<unsigned>(std::max(0, seed));
                return true;
            });
            if (samples_each) {
                ImGui::TableNextColumn();
                changed |= Cell("Samples each", [&] {
                    if (!ImGui::InputInt("##samples", &settings.samples_per_prompt, 0)) return false;
                    settings.samples_per_prompt =
                        std::clamp(settings.samples_per_prompt, 1, cyxwiz::kGenerationPreviewMaxSamplesPerPrompt);
                    return true;
                });
            }
            ImGui::EndTable();
        }
        if (random_seed) {
            ImGui::Checkbox("New seed for each reply", random_seed);
            GenerationHint(*random_seed ? "Each reply draws differently; its seed is shown under it."
                                        : "Every reply uses the seed above, so the same message gives the same reply.");
        }
        GenerationHint("Draws each token at random from the most likely ones. Top-k 0, top-p 1 and min-p 0 are "
                       "off; min-p keeps tokens at least that fraction as likely as the best one.");
    } else if (settings.decoding == GenerationPreviewDecoding::Beam) {
        if (ImGui::BeginTable("##beam", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            changed |= Cell("Beams", [&] {
                if (!ImGui::InputInt("##beams", &settings.num_beams, 1)) return false;
                settings.num_beams = std::clamp(settings.num_beams, 2, 8);
                return true;
            });
            ImGui::TableNextColumn();
            changed |= Cell("Length penalty", [&] {
                if (!ImGui::InputFloat("##length_penalty", &settings.length_penalty, 0.0f, 0.0f, "%.2f")) return false;
                settings.length_penalty = std::clamp(settings.length_penalty, 0.0f, 2.0f);
                return true;
            });
            ImGui::EndTable();
        }
        GenerationHint("Keeps the most likely sequences, not single tokens, and returns the best one. The same "
                       "output every time; slower (each beam is recomputed). A length penalty above 1 favours "
                       "longer outputs.");
    } else {
        GenerationHint("Always picks the most likely next token: the same output every time. Best for comparing; "
                       "loops show plainly.");
    }

    // Penalties apply to every decoding.
    ImGui::TextColored(t.text_dim, "Penalties");
    bool penalty = settings.repetition_penalty > 1.0f;
    if (ImGui::Checkbox("Repetition", &penalty)) {
        settings.repetition_penalty = penalty ? 1.1f : 1.0f;
        changed = true;
    }
    if (penalty) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80.0f);
        if (ImGui::InputFloat("##repetition_penalty", &settings.repetition_penalty, 0.0f, 0.0f, "%.2f")) {
            settings.repetition_penalty = std::clamp(settings.repetition_penalty, 1.0f, 2.0f);
            changed = true;
        }
    }
    if (ImGui::BeginTable("##penalties", columns(3), ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        changed |= Cell("Frequency", [&] {
            if (!ImGui::InputFloat("##frequency_penalty", &settings.frequency_penalty, 0.0f, 0.0f, "%.2f")) return false;
            settings.frequency_penalty = std::clamp(settings.frequency_penalty, 0.0f, 2.0f);
            return true;
        });
        ImGui::TableNextColumn();
        changed |= Cell("Presence", [&] {
            if (!ImGui::InputFloat("##presence_penalty", &settings.presence_penalty, 0.0f, 0.0f, "%.2f")) return false;
            settings.presence_penalty = std::clamp(settings.presence_penalty, 0.0f, 2.0f);
            return true;
        });
        ImGui::TableNextColumn();
        changed |= Cell("No-repeat n-gram", [&] {
            if (!ImGui::InputInt("##no_repeat_ngram", &settings.no_repeat_ngram, 0)) return false;
            settings.no_repeat_ngram = std::clamp(settings.no_repeat_ngram, 0, 10);
            return true;
        });
        ImGui::EndTable();
    }
    GenerationHint("Repetition (1 to 2) makes tokens already used less likely. Frequency and presence (0 to 2) "
                   "lower tokens by how often, or whether, the output used them. No repeated n-gram (0 is off) "
                   "never repeats a run of that many tokens.");
    return changed;
}

}  // namespace gui
