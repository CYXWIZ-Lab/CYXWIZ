#include "data_loader_generation_tab.h"

#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/arrow_dataset.h"
#include "../core/graph_compiler_dataset_hooks.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <set>
#include <sstream>

namespace gui {
namespace {

using cyxwiz::GenerationPreviewCadence;
using cyxwiz::GenerationPreviewCheckKind;
using cyxwiz::GenerationPreviewDecoding;
using cyxwiz::GenerationPreviewMode;
using cyxwiz::GenerationPreviewTemplate;
using cyxwiz::GenerationPreviewWeights;

constexpr const char* kPrefix = "generation_preview_";

// A card: a tinted child that grows with its content (no outline).
struct Card {
    explicit Card(const char* id) {
        const auto& t = cyxwiz::ui::CurrentTokens();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
        ImGui::BeginChild(id, ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    }
    ~Card() {
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
};

void Title(const char* text, const char* note = nullptr) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TextColored(t.text_bright, "%s", text);
    if (note && *note) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", note);
    }
}

void Hint(const char* text) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "%s", text);
    ImGui::PopTextWrapPos();
}

// Puts the next item, `width` wide, on this line when it fits, else on the next.
void FlowSameLine(float width, float gap = -1.0f) {
    ImGui::SameLine(0.0f, gap);
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

// `text` cut with "..." to fit `width` pixels.
std::string FitText(const std::string& text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    std::string cut = text;
    while (!cut.empty() && ImGui::CalcTextSize((cut + "...").c_str()).x > width) {
        cut.pop_back();
        while (!cut.empty() && (static_cast<unsigned char>(cut.back()) & 0xC0) == 0x80) cut.pop_back();
    }
    return cut + "...";
}

float SegmentedWidth(const char* const* labels, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; ++i) total += ImGui::CalcTextSize(labels[i], nullptr, true).x + 24.0f;
    return total;
}

std::string FirstLine(const std::string& text, size_t max_chars) {
    std::string line = text.substr(0, text.find('\n'));
    if (line.size() > max_chars) line = line.substr(0, max_chars) + "...";
    return line;
}

std::string Shown(const std::string& text) {
    std::string shown;
    for (char c : text) shown += c == '\n' ? std::string("\\n") : std::string(1, c);
    return shown;
}

// Draws text segments in their own colours, wrapped at the current width.
void DrawSegments(const std::vector<std::pair<std::string, ImVec4>>& segments) {
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
            const float w = ImGui::CalcTextSize(begin, stop).x;
            if (x > 0.0f && x + w > width) {
                x = 0.0f;
                y += line_height;
            }
            draw->AddText(ImVec2(origin.x + x, origin.y + y), col, begin, stop);
            x += w;
            i = end;
        }
    }
    ImGui::Dummy(ImVec2(width, y + line_height));
}

const char* CheckKindLabel(GenerationPreviewCheckKind kind) {
    switch (kind) {
        case GenerationPreviewCheckKind::None: return "No check";
        case GenerationPreviewCheckKind::Contains: return "Output contains";
        case GenerationPreviewCheckKind::NotContains: return "Output does not contain";
        case GenerationPreviewCheckKind::Regex: return "Output matches the pattern";
        case GenerationPreviewCheckKind::EndsWithStop: return "Stops by itself";
    }
    return "";
}

std::string CheckShort(const cyxwiz::GenerationPreviewCheck& check) {
    switch (check.kind) {
        case GenerationPreviewCheckKind::Contains: return "contains " + check.value;
        case GenerationPreviewCheckKind::NotContains: return "not " + check.value;
        case GenerationPreviewCheckKind::Regex: return "pattern";
        case GenerationPreviewCheckKind::EndsWithStop: return "stops by itself";
        default: return "no check";
    }
}

}  // namespace

void DataLoaderGenerationTab::Load(const std::map<std::string, std::string>& params) {
    settings_ = {};
    old_format_note_.clear();
    selected_ = 0;
    std::map<std::string, std::string> own;
    for (const auto& [key, value] : params)
        if (key.rfind(kPrefix, 0) == 0) own[key] = value;
    const bool enabled = own.count("generation_preview_enabled") && own["generation_preview_enabled"] == "true";
    if (own.count("generation_preview_prompts") || own.count("generation_preview_every_epochs")) {
        // Saved before this tab existed: those prompts are not read (owner rule: no legacy shims).
        if (enabled) {
            old_format_note_ = "This Data Loader was saved before the Generation tab and its prompts are not "
                               "used. Add the prompts again below, then apply.";
        }
        settings_.enabled = enabled;
        return;
    }
    // Read the stored settings even when previews are off, so turning them on keeps them.
    own["generation_preview_enabled"] = "true";
    try {
        settings_ = cyxwiz::ParseTrainingGenerationPreview(own);
    } catch (const std::exception&) {
        settings_ = {};
        // Keep what parses on its own: the prompts and the mode.
        try {
            auto only = own;
            for (const char* key : {"generation_preview_stop_texts", "generation_preview_max_new_tokens",
                                    "generation_preview_temperature", "generation_preview_top_k",
                                    "generation_preview_top_p", "generation_preview_every"})
                only.erase(key);
            settings_ = cyxwiz::ParseTrainingGenerationPreview(only);
        } catch (const std::exception&) {
            settings_ = {};
        }
    }
    settings_.enabled = enabled;
}

void DataLoaderGenerationTab::Store(std::map<std::string, std::string>& params) const {
    params.erase("generation_preview_prompts");
    params.erase("generation_preview_every_epochs");
    for (const auto& [key, value] : cyxwiz::SerializeTrainingGenerationPreview(settings_)) params[key] = value;
}

std::string DataLoaderGenerationTab::Problem() const {
    return cyxwiz::ValidateTrainingGenerationPreview(settings_);
}

bool DataLoaderGenerationTab::Render(const GenerationTabDataset& dataset) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    if (ImGui::Checkbox("Generate samples while training", &settings_.enabled)) changed = true;
    ImGui::BeginDisabled(!settings_.enabled);
    const char* modes[] = {"Continuation · base model", "Chat / instruction · fine-tuned model"};
    FlowSameLine(SegmentedWidth(modes, 2), 20.0f);
    int mode = settings_.mode == GenerationPreviewMode::Chat ? 1 : 0;
    if (cyxwiz::ui::SegmentedControl("##gen_mode", modes, 2, &mode)) {
        settings_.mode = mode == 1 ? GenerationPreviewMode::Chat : GenerationPreviewMode::Continuation;
        changed = true;
    }
    if (dataset.found) {
        const auto suggested = cyxwiz::SuggestedGenerationPreviewMode(dataset.supervision);
        const bool matches = suggested == settings_.mode;
        const char* note = matches ? (suggested == GenerationPreviewMode::Chat
                                          ? "Matches the data: it trains on responses only"
                                          : "Matches the data: plain text windows")
                                   : (suggested == GenerationPreviewMode::Chat
                                          ? "The data trains on responses only: chat is suggested"
                                          : "The data is plain text windows: continuation is suggested");
        FlowSameLine(ImGui::CalcTextSize(note).x, 12.0f);
        ImGui::TextColored(matches ? t.text_dim : t.warning, "%s", note);
    }
    if (!old_format_note_.empty()) ImGui::TextColored(t.warning, "%s", old_format_note_.c_str());
    ImGui::Spacing();

    if (ImGui::GetContentRegionAvail().x < 900.0f) {
        changed |= RenderPrompts();
        changed |= RenderPromptEditor();
        RenderModelSees(dataset);
        if (settings_.mode == GenerationPreviewMode::Chat) changed |= RenderTemplate();
        changed |= RenderStops();
        changed |= RenderDecoding();
        changed |= RenderSchedule();
    } else if (ImGui::BeginTable("##gen_columns", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("left", ImGuiTableColumnFlags_WidthStretch, 1.25f);
        ImGui::TableSetupColumn("right", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        changed |= RenderPrompts();
        changed |= RenderPromptEditor();
        RenderModelSees(dataset);
        ImGui::TableSetColumnIndex(1);
        if (settings_.mode == GenerationPreviewMode::Chat) changed |= RenderTemplate();
        changed |= RenderStops();
        changed |= RenderDecoding();
        changed |= RenderSchedule();
        ImGui::EndTable();
    }
    ImGui::EndDisabled();
    if (settings_.enabled) {
        if (const auto problem = Problem(); !problem.empty()) ImGui::TextColored(t.error, "%s", problem.c_str());
    }
    Hint("Saved with the graph. Each run writes generation_previews/ under its checkpoint folder (baseline, "
         "epoch_N, step_N); the Console shows one block per pass.");
    return changed;
}

bool DataLoaderGenerationTab::RenderPrompts() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Card card("##gen_prompts");
    auto& prompts = settings_.prompts;
    const std::string count = std::to_string(prompts.size()) + " of " +
                              std::to_string(cyxwiz::kGenerationPreviewMaxPrompts);
    Title("Prompts", count.c_str());
    const float add_width = cyxwiz::ui::ButtonWidth("+ Add prompt", cyxwiz::ui::ButtonSize::Small);
    cyxwiz::ui::SameLineRight(add_width);
    const bool can_add = prompts.size() < static_cast<size_t>(cyxwiz::kGenerationPreviewMaxPrompts);
    if (cyxwiz::ui::SecondaryButton("+ Add prompt", can_add, "At most 8 prompts")) {
        prompts.push_back({});
        selected_ = static_cast<int>(prompts.size()) - 1;
        changed = true;
    }
    if (prompts.empty()) Hint("No prompts yet. Add one, then type what the model should continue or answer.");
    int remove = -1, move_up = -1, move_down = -1;
    for (int i = 0; i < static_cast<int>(prompts.size()); ++i) {
        ImGui::PushID(i);
        const float row_x = ImGui::GetCursorPosX();
        const float row_width = ImGui::GetContentRegionAvail().x;
        const std::string short_check = FitText(CheckShort(prompts[i].check), row_width * 0.4f);
        const float check_width = ImGui::CalcTextSize(short_check.c_str()).x;
        const std::string label = FitText(std::to_string(i + 1) + "   " +
                                              (prompts[i].text.empty() ? std::string("(empty)")
                                                                       : FirstLine(prompts[i].text, 200)),
                                          row_width - check_width - 24.0f);
        if (ImGui::Selectable((label + "##row").c_str(), selected_ == i)) selected_ = i;
        ImGui::SameLine(row_x + row_width - check_width);
        ImGui::TextColored(t.text_dim, "%s", short_check.c_str());
        if (ImGui::BeginPopupContextItem("##prompt_menu")) {
            if (ImGui::MenuItem("Move up", nullptr, false, i > 0)) move_up = i;
            if (ImGui::MenuItem("Move down", nullptr, false, i + 1 < static_cast<int>(prompts.size()))) move_down = i;
            if (ImGui::MenuItem("Remove")) remove = i;
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
    if (move_up > 0) std::swap(prompts[move_up], prompts[move_up - 1]), selected_ = move_up - 1, changed = true;
    if (move_down >= 0) std::swap(prompts[move_down], prompts[move_down + 1]), selected_ = move_down + 1, changed = true;
    if (remove >= 0) {
        prompts.erase(prompts.begin() + remove);
        selected_ = std::clamp(selected_, 0, std::max(0, static_cast<int>(prompts.size()) - 1));
        changed = true;
    }
    if (!prompts.empty()) Hint("Right-click a prompt to move or remove it.");
    return changed;
}

bool DataLoaderGenerationTab::RenderPromptEditor() {
    auto& prompts = settings_.prompts;
    if (prompts.empty()) return false;
    selected_ = std::clamp(selected_, 0, static_cast<int>(prompts.size()) - 1);
    auto& prompt = prompts[static_cast<size_t>(selected_)];
    bool changed = false;
    Card card("##gen_prompt_editor");
    const bool chat = settings_.mode == GenerationPreviewMode::Chat;
    const std::string title = std::string(chat ? "User message" : "Prompt text") + " · prompt " +
                              std::to_string(selected_ + 1);
    Title(title.c_str());
    ImGui::PushID(selected_);
    if (ImGui::InputTextMultiline("##prompt_text", &prompt.text, ImVec2(-1.0f, ImGui::GetTextLineHeight() * 7.0f)))
        changed = true;
    const char* kinds[] = {"No check", "Output contains", "Output does not contain", "Output matches the pattern",
                           "Stops by itself"};
    int kind = static_cast<int>(prompt.check.kind);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "Check");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::Combo("##check_kind", &kind, kinds, IM_ARRAYSIZE(kinds))) {
        prompt.check.kind = static_cast<GenerationPreviewCheckKind>(kind);
        changed = true;
    }
    const bool needs_value = prompt.check.kind == GenerationPreviewCheckKind::Contains ||
                             prompt.check.kind == GenerationPreviewCheckKind::NotContains ||
                             prompt.check.kind == GenerationPreviewCheckKind::Regex;
    if (needs_value) {
        FlowSameLine(180.0f);
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::InputTextWithHint("##check_value", prompt.check.kind == GenerationPreviewCheckKind::Regex
                                                          ? "pattern, e.g. Psalm \\d+:\\d+" : "text",
                                     &prompt.check.value))
            changed = true;
    }
    ImGui::PopID();
    Hint("Checks are optional. Each output shows pass or fail next to it in the Generations view.");
    return changed;
}

void DataLoaderGenerationTab::RenderModelSees(const GenerationTabDataset& dataset) {
    if (settings_.prompts.empty()) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    const auto& prompt = settings_.prompts[static_cast<size_t>(selected_)];
    const auto rendered = cyxwiz::RenderGenerationPreviewPrompt(settings_, prompt.text);
    Card card("##gen_model_sees");
    const std::string note = "prompt " + std::to_string(selected_ + 1) +
                             (settings_.mode == GenerationPreviewMode::Chat ? ", template applied" : "");
    Title("What the model sees", note.c_str());
    std::string count;
    ImVec4 count_colour = t.text_dim;
    if (dataset.tokenizer && dataset.context > 0) {
        const size_t tokens = dataset.tokenizer->Encode(rendered.text).size();
        count = std::to_string(tokens) + " / " + std::to_string(dataset.context) + " tokens";
        const int room = dataset.context - static_cast<int>(tokens);
        if (room <= 0) count_colour = t.error;
        else if (room < settings_.max_new_tokens) count_colour = t.warning;
    } else {
        count = "token count shows when the dataset is loaded";
    }
    const float count_width = ImGui::CalcTextSize(count.c_str()).x;
    cyxwiz::ui::SameLineRight(count_width);
    ImGui::TextColored(count_colour, "%s", count.c_str());

    std::vector<std::pair<std::string, ImVec4>> segments;
    for (const auto& segment : rendered.segments)
        segments.push_back({segment.text, segment.from_template ? t.accent_text : t.text});
    segments.push_back({"|", t.success});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##gen_rendered", ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 9.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding);
    DrawSegments(segments);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::TextColored(t.accent_text, "from the template");
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::TextColored(t.text, "your prompt");
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::TextColored(t.success, "| generation starts here");
}

bool DataLoaderGenerationTab::RenderTemplate() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Card card("##gen_template");
    Title("Chat template");
    const char* presets[] = {"Role lines", "ChatML", "Llama 3"};
    int preset = static_cast<int>(settings_.chat_template);
    if (cyxwiz::ui::SegmentedControl("##gen_template", presets, 3, &preset)) {
        settings_.chat_template = static_cast<GenerationPreviewTemplate>(preset);
        changed = true;
    }
    FlowSameLine(cyxwiz::ui::ButtonWidth("From tokenizer", cyxwiz::ui::ButtonSize::Small), 8.0f);
    cyxwiz::ui::SecondaryButton("From tokenizer", false, "This tokenizer has no chat template");
    switch (settings_.chat_template) {
        case GenerationPreviewTemplate::RoleLines:
            Hint("A label before each turn, one turn per line. Use the labels the data was trained with.");
            break;
        case GenerationPreviewTemplate::ChatML:
            Hint("<|im_start|> and <|im_end|> markers. Use it only when the data was built with them.");
            break;
        case GenerationPreviewTemplate::Llama3:
            Hint("Llama 3 header and end-of-turn markers. Use it only when the data was built with them.");
            break;
    }
    ImGui::TextColored(t.text_dim, "System prompt · shared by every prompt");
    if (ImGui::InputTextMultiline("##system_prompt", &settings_.system_prompt,
                                  ImVec2(-1.0f, ImGui::GetTextLineHeight() * 4.0f)))
        changed = true;
    if (settings_.chat_template == GenerationPreviewTemplate::RoleLines &&
        ImGui::BeginTable("##labels", 3, ImGuiTableFlags_SizingStretchSame)) {
        const std::pair<const char*, std::string*> labels[] = {{"System", &settings_.label_system},
                                                               {"User", &settings_.label_user},
                                                               {"Assistant", &settings_.label_assistant}};
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

bool DataLoaderGenerationTab::RenderStops() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Card card("##gen_stops");
    Title("Stop when the model writes");
    cyxwiz::ui::Chip("End-of-sequence token", t.accent_text);
    for (const auto& stop : cyxwiz::GenerationPreviewTemplateStopTexts(settings_)) {
        const std::string label = Shown(stop) + "  (template)";
        FlowSameLine(cyxwiz::ui::ChipWidth(label.c_str()));
        cyxwiz::ui::Chip(label.c_str(), t.text_dim);
    }
    int remove = -1;
    for (size_t i = 0; i < settings_.stop_texts.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        const std::string label = Shown(settings_.stop_texts[i]) + "  x";
        FlowSameLine(cyxwiz::ui::ChipButtonWidth(label.c_str()));
        if (cyxwiz::ui::ChipButton(label.c_str())) remove = static_cast<int>(i);
        cyxwiz::ui::Tooltip("Remove this stop text");
        ImGui::PopID();
    }
    if (remove >= 0) {
        settings_.stop_texts.erase(settings_.stop_texts.begin() + remove);
        changed = true;
    }
    const float add_width = cyxwiz::ui::ButtonWidth("Add", cyxwiz::ui::ButtonSize::Small);
    ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x - add_width - 8.0f));
    const bool entered = ImGui::InputTextWithHint("##new_stop", "Add a stop text, e.g. </answer>", &new_stop_,
                                                  ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    const bool can_add = !new_stop_.empty() &&
                         settings_.stop_texts.size() < static_cast<size_t>(cyxwiz::kGenerationPreviewMaxStopTexts);
    if ((cyxwiz::ui::SecondaryButton("Add", can_add, "Type a stop text first (at most 8)") || entered) && can_add) {
        settings_.stop_texts.push_back(new_stop_);
        new_stop_.clear();
        changed = true;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Max new tokens");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::InputInt("##max_new_tokens", &settings_.max_new_tokens)) {
        settings_.max_new_tokens = std::clamp(settings_.max_new_tokens, 1, cyxwiz::kGenerationPreviewMaxNewTokens);
        changed = true;
    }
    const char* cap = "1 to 1024, capped by the context the prompt leaves";
    FlowSameLine(ImGui::CalcTextSize(cap).x);
    ImGui::TextColored(t.text_faint, "%s", cap);
    return changed;
}

bool DataLoaderGenerationTab::RenderDecoding() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Card card("##gen_decoding");
    Title("Decoding");
    const char* decodings[] = {"Greedy", "Sampling"};
    int decoding = settings_.decoding == GenerationPreviewDecoding::Sampling ? 1 : 0;
    cyxwiz::ui::SameLineRight(ImGui::CalcTextSize("Greedy").x + ImGui::CalcTextSize("Sampling").x + 48.0f);
    if (cyxwiz::ui::SegmentedControl("##gen_decoding", decodings, 2, &decoding)) {
        settings_.decoding = decoding == 1 ? GenerationPreviewDecoding::Sampling : GenerationPreviewDecoding::Greedy;
        if (settings_.decoding == GenerationPreviewDecoding::Greedy) settings_.samples_per_prompt = 1;
        changed = true;
    }
    if (settings_.decoding == GenerationPreviewDecoding::Sampling) {
        if (ImGui::BeginTable("##sampling", 5, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(t.text_dim, "Temperature");
            ImGui::SetNextItemWidth(-1.0f);
            changed |= ImGui::InputFloat("##temperature", &settings_.temperature, 0.0f, 0.0f, "%.2f");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(t.text_dim, "Top-k");
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::InputInt("##top_k", &settings_.top_k, 0)) {
                settings_.top_k = std::max(0, settings_.top_k);
                changed = true;
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::TextColored(t.text_dim, "Top-p");
            ImGui::SetNextItemWidth(-1.0f);
            changed |= ImGui::InputFloat("##top_p", &settings_.top_p, 0.0f, 0.0f, "%.2f");
            ImGui::TableSetColumnIndex(3);
            ImGui::TextColored(t.text_dim, "Seed");
            ImGui::SetNextItemWidth(-1.0f);
            int seed = static_cast<int>(settings_.seed);
            if (ImGui::InputInt("##seed", &seed, 0)) {
                settings_.seed = static_cast<unsigned>(std::max(0, seed));
                changed = true;
            }
            ImGui::TableSetColumnIndex(4);
            ImGui::TextColored(t.text_dim, "Samples each");
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::InputInt("##samples", &settings_.samples_per_prompt, 0)) {
                settings_.samples_per_prompt =
                    std::clamp(settings_.samples_per_prompt, 1, cyxwiz::kGenerationPreviewMaxSamplesPerPrompt);
                changed = true;
            }
            ImGui::EndTable();
        }
        Hint("A fixed seed makes passes comparable: each prompt gets the same random draws every time. "
             "Top-k 0 is off.");
    } else {
        Hint("Always picks the most likely next token. Best for comparing epochs; loops show plainly.");
    }
    bool penalty = settings_.repetition_penalty > 1.0f;
    if (ImGui::Checkbox("Repetition penalty", &penalty)) {
        settings_.repetition_penalty = penalty ? 1.1f : 1.0f;
        changed = true;
    }
    if (penalty) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        if (ImGui::InputFloat("##repetition_penalty", &settings_.repetition_penalty, 0.0f, 0.0f, "%.2f")) {
            settings_.repetition_penalty = std::clamp(settings_.repetition_penalty, 1.0f, 2.0f);
            changed = true;
        }
        const char* note = "1 to 2; makes tokens already used less likely";
        FlowSameLine(ImGui::CalcTextSize(note).x);
        ImGui::TextColored(t.text_faint, "%s", note);
    }
    return changed;
}

bool DataLoaderGenerationTab::RenderSchedule() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    Card card("##gen_schedule");
    Title("When");
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Every");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::InputInt("##every", &settings_.every, 0)) {
        settings_.every = std::max(1, settings_.every);
        changed = true;
    }
    ImGui::SameLine();
    const char* units[] = {"epochs", "optimizer steps"};
    int unit = settings_.cadence == GenerationPreviewCadence::Steps ? 1 : 0;
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("##every_unit", &unit, units, 2)) {
        settings_.cadence = unit == 1 ? GenerationPreviewCadence::Steps : GenerationPreviewCadence::Epochs;
        changed = true;
    }
    changed |= ImGui::Checkbox("Before training starts (baseline)", &settings_.baseline);
    changed |= ImGui::Checkbox("After the last epoch", &settings_.after_last);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Weights");
    ImGui::SameLine();
    const char* weights[] = {"Current", "Best so far"};
    int weight = settings_.weights == GenerationPreviewWeights::Best ? 1 : 0;
    if (cyxwiz::ui::SegmentedControl("##gen_weights", weights, 2, &weight)) {
        settings_.weights = weight == 1 ? GenerationPreviewWeights::Best : GenerationPreviewWeights::Current;
        changed = true;
    }
    if (settings_.weights == GenerationPreviewWeights::Best)
        Hint("Uses the best validation checkpoint so far; until one exists, the current weights.");
    return changed;
}

GenerationTabDataset ResolveGenerationTabDataset(NodeEditor* editor, int loader_node_id) {
    GenerationTabDataset found;
    if (!editor) return found;
    const auto& nodes = editor->GetNodes();
    const auto& links = editor->GetLinks();
    std::set<int> seen{loader_node_id};
    std::vector<int> frontier{loader_node_id};
    const MLNode* input = nullptr;
    for (size_t i = 0; i < frontier.size() && !input; ++i) {
        for (const auto& link : links) {
            if (link.to_node != frontier[i] || !seen.insert(link.from_node).second) continue;
            const auto it = std::find_if(nodes.begin(), nodes.end(),
                                         [&](const MLNode& n) { return n.id == link.from_node; });
            if (it == nodes.end()) continue;
            if (it->type == NodeType::DataInput) {
                input = &*it;
                break;
            }
            frontier.push_back(it->id);
        }
    }
    if (!input) return found;
    const auto name = input->parameters.find("dataset_name");
    if (name == input->parameters.end()) return found;
    found.name = name->second;
    const auto dataset = cyxwiz::GraphArrowDataset(found.name);
    if (!dataset || !dataset->GetSchema()) return found;
    const auto metadata = dataset->GetSchema()->metadata();
    if (!metadata) return found;
    const auto get = [&](const char* key) {
        const auto value = metadata->Get(key);
        return value.ok() ? *value : std::string();
    };
    if (get("cyxwiz.token_windows.version") != "1") return found;
    found.found = true;
    found.supervision = get("cyxwiz.token_windows.supervision");
    try {
        found.context = std::stoi(get("cyxwiz.token_windows.context"));
    } catch (...) {
        found.context = 0;
    }
    const auto type = get("cyxwiz.token_windows.tokenizer_type");
    if (type.size() == 1 && type[0] >= '0' && type[0] <= '3') {
        auto tokenizer = std::make_shared<cyxwiz::Tokenizer>(static_cast<cyxwiz::TokenizerType>(type[0] - '0'));
        std::istringstream artifact(get("cyxwiz.token_windows.vocabulary"));
        if (tokenizer->GetVocabulary().LoadFromStream(artifact)) {
            tokenizer->SetLowercase(get("cyxwiz.token_windows.lowercase") == "true");
            tokenizer->SetPadding(false);
            tokenizer->SetTruncation(false);
            tokenizer->SetAddBos(false);
            tokenizer->SetAddEos(false);
            found.tokenizer = std::move(tokenizer);
        }
    }
    return found;
}

}  // namespace gui
