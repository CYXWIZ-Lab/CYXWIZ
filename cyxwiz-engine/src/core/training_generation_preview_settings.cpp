#include "training_generation_preview_settings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace cyxwiz {
namespace {

constexpr const char* kPrefix = "generation_preview_";

std::string Key(const char* name) { return std::string(kPrefix) + name; }

const std::string* Find(const std::map<std::string, std::string>& params, const char* name) {
    const auto it = params.find(Key(name));
    return it == params.end() ? nullptr : &it->second;
}

[[noreturn]] void Fail(const std::string& message) { throw std::invalid_argument(message); }

long long ParseInteger(const std::string& text, const char* name, long long low, long long high) {
    long long value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < low || value > high)
        Fail(std::string("Generation previews: ") + name + " must be a whole number from " +
             std::to_string(low) + " to " + std::to_string(high) + " (got \"" + text + "\")");
    return value;
}

float ParseFloat(const std::string& text, const char* name) {
    // strtof, not from_chars: Apple's libc++ only has float from_chars from macOS 26.
    char* end = nullptr;
    errno = 0;
    const float value = text.empty() || std::isspace(static_cast<unsigned char>(text[0]))
                            ? 0.0f : std::strtof(text.c_str(), &end);
    if (end != text.c_str() + text.size() || errno == ERANGE || !std::isfinite(value))
        Fail(std::string("Generation previews: ") + name + " must be a number (got \"" + text + "\")");
    return value;
}

bool ParseBool(const std::string& text, const char* name) {
    if (text == "true") return true;
    if (text == "false") return false;
    Fail(std::string("Generation previews: ") + name + " must be true or false");
}

template <typename Enum, size_t N>
Enum ParseChoice(const std::string& text, const char* name, const std::pair<const char*, Enum> (&choices)[N]) {
    for (const auto& [label, value] : choices)
        if (text == label) return value;
    std::string allowed;
    for (const auto& choice : choices) allowed += (allowed.empty() ? "" : ", ") + std::string(choice.first);
    Fail(std::string("Generation previews: ") + name + " must be one of " + allowed + " (got \"" + text + "\")");
}

const std::pair<const char*, GenerationPreviewMode> kModes[] = {
    {"continuation", GenerationPreviewMode::Continuation}, {"chat", GenerationPreviewMode::Chat}};
const std::pair<const char*, GenerationPreviewTemplate> kTemplates[] = {
    {"role_lines", GenerationPreviewTemplate::RoleLines},
    {"chatml", GenerationPreviewTemplate::ChatML},
    {"llama3", GenerationPreviewTemplate::Llama3}};
const std::pair<const char*, GenerationPreviewDecoding> kDecodings[] = {
    {"greedy", GenerationPreviewDecoding::Greedy}, {"sampling", GenerationPreviewDecoding::Sampling},
    {"beam", GenerationPreviewDecoding::Beam}};
const std::pair<const char*, GenerationPreviewCadence> kCadences[] = {
    {"epochs", GenerationPreviewCadence::Epochs}, {"steps", GenerationPreviewCadence::Steps}};
const std::pair<const char*, GenerationPreviewWeights> kWeights[] = {
    {"current", GenerationPreviewWeights::Current}, {"best", GenerationPreviewWeights::Best}};
const std::pair<const char*, GenerationPreviewCheckKind> kCheckKinds[] = {
    {"none", GenerationPreviewCheckKind::None},
    {"contains", GenerationPreviewCheckKind::Contains},
    {"not_contains", GenerationPreviewCheckKind::NotContains},
    {"regex", GenerationPreviewCheckKind::Regex},
    {"ends_with_stop", GenerationPreviewCheckKind::EndsWithStop}};

template <typename Enum, size_t N>
const char* NameOf(Enum value, const std::pair<const char*, Enum> (&choices)[N]) {
    for (const auto& [label, choice] : choices)
        if (choice == value) return label;
    return "unknown";
}

std::string FormatFloat(float value) {
    std::ostringstream out;
    out << std::setprecision(6) << value;
    return out.str();
}

std::vector<GenerationPreviewPrompt> ParsePromptSet(const std::string& text) {
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception&) {
        Fail("Generation previews: the prompt list is not valid JSON");
    }
    if (!document.is_array()) Fail("Generation previews: the prompt list must be a JSON array");
    std::vector<GenerationPreviewPrompt> prompts;
    for (const auto& item : document) {
        if (!item.is_object() || !item.contains("text") || !item.at("text").is_string())
            Fail("Generation previews: each prompt needs a \"text\" string");
        GenerationPreviewPrompt prompt;
        prompt.text = item.at("text").get<std::string>();
        if (item.contains("check") && !item.at("check").is_null()) {
            const auto& check = item.at("check");
            if (!check.is_object() || !check.contains("kind") || !check.at("kind").is_string())
                Fail("Generation previews: a prompt check needs a \"kind\"");
            prompt.check.kind = ParseChoice(check.at("kind").get<std::string>(), "check kind", kCheckKinds);
            if (check.contains("value") && check.at("value").is_string())
                prompt.check.value = check.at("value").get<std::string>();
        }
        prompts.push_back(std::move(prompt));
    }
    return prompts;
}

std::string PromptSetJson(const std::vector<GenerationPreviewPrompt>& prompts) {
    nlohmann::json document = nlohmann::json::array();
    for (const auto& prompt : prompts) {
        nlohmann::json item = {{"text", prompt.text}};
        if (prompt.check.kind != GenerationPreviewCheckKind::None)
            item["check"] = {{"kind", NameOf(prompt.check.kind, kCheckKinds)}, {"value", prompt.check.value}};
        document.push_back(std::move(item));
    }
    return document.dump();
}

void AddSegment(RenderedPreviewPrompt& rendered, std::string text, bool from_template) {
    if (text.empty()) return;
    rendered.text += text;
    rendered.segments.push_back({std::move(text), from_template});
}

}  // namespace

const char* GenerationPreviewModeName(GenerationPreviewMode value) { return NameOf(value, kModes); }
const char* GenerationPreviewTemplateName(GenerationPreviewTemplate value) { return NameOf(value, kTemplates); }
const char* GenerationPreviewTemplateLabel(GenerationPreviewTemplate value) {
    switch (value) {
        case GenerationPreviewTemplate::RoleLines: return "Role lines";
        case GenerationPreviewTemplate::ChatML: return "ChatML";
        case GenerationPreviewTemplate::Llama3: return "Llama 3";
    }
    return "Unknown";
}
const char* GenerationPreviewDecodingName(GenerationPreviewDecoding value) { return NameOf(value, kDecodings); }
const char* GenerationPreviewCadenceName(GenerationPreviewCadence value) { return NameOf(value, kCadences); }
const char* GenerationPreviewWeightsName(GenerationPreviewWeights value) { return NameOf(value, kWeights); }
const char* GenerationPreviewCheckKindName(GenerationPreviewCheckKind value) { return NameOf(value, kCheckKinds); }
const char* GenerationPreviewDecodingLabel(GenerationPreviewDecoding value) {
    switch (value) {
        case GenerationPreviewDecoding::Greedy: return "Greedy";
        case GenerationPreviewDecoding::Sampling: return "Sampling";
        case GenerationPreviewDecoding::Beam: return "Beam search";
    }
    return "";
}

void AddGenerationDecodingFields(const TrainingGenerationPreviewSettings& s, nlohmann::json& report) {
    const bool sampling = s.decoding == GenerationPreviewDecoding::Sampling;
    const bool beam = s.decoding == GenerationPreviewDecoding::Beam;
    report["decoding"] = GenerationPreviewDecodingName(s.decoding);
    report["temperature"] = sampling ? s.temperature : 0.0f;
    report["top_k"] = sampling ? s.top_k : 0;
    report["top_p"] = sampling ? s.top_p : 1.0f;
    report["min_p"] = sampling ? s.min_p : 0.0f;
    report["num_beams"] = beam ? s.num_beams : 1;
    report["length_penalty"] = beam ? s.length_penalty : 1.0f;
    report["repetition_penalty"] = s.repetition_penalty;
    report["frequency_penalty"] = s.frequency_penalty;
    report["presence_penalty"] = s.presence_penalty;
    report["no_repeat_ngram"] = s.no_repeat_ngram;
}

namespace {
// Every stored setting but the on/off switch, without validating the whole.
void ParseGenerationFields(const std::map<std::string, std::string>& params, TrainingGenerationPreviewSettings& value) {
    if (const auto* text = Find(params, "mode")) value.mode = ParseChoice(*text, "mode", kModes);
    if (const auto* text = Find(params, "template")) value.chat_template = ParseChoice(*text, "template", kTemplates);
    if (const auto* text = Find(params, "system_prompt")) value.system_prompt = *text;
    if (const auto* text = Find(params, "label_system")) value.label_system = *text;
    if (const auto* text = Find(params, "label_user")) value.label_user = *text;
    if (const auto* text = Find(params, "label_assistant")) value.label_assistant = *text;
    if (const auto* text = Find(params, "stop_texts"); text && !text->empty()) {
        nlohmann::json document;
        try {
            document = nlohmann::json::parse(*text);
        } catch (const nlohmann::json::exception&) {
            Fail("Generation previews: the stop texts are not valid JSON");
        }
        if (!document.is_array()) Fail("Generation previews: the stop texts must be a JSON array");
        for (const auto& item : document) {
            if (!item.is_string()) Fail("Generation previews: each stop text must be a string");
            value.stop_texts.push_back(item.get<std::string>());
        }
    }
    if (const auto* text = Find(params, "max_new_tokens"))
        value.max_new_tokens = static_cast<int>(ParseInteger(*text, "max new tokens", 1, kGenerationPreviewMaxNewTokens));
    if (const auto* text = Find(params, "decoding")) value.decoding = ParseChoice(*text, "decoding", kDecodings);
    if (const auto* text = Find(params, "temperature")) value.temperature = ParseFloat(*text, "temperature");
    if (const auto* text = Find(params, "top_k")) value.top_k = static_cast<int>(ParseInteger(*text, "top-k", 0, 100000));
    if (const auto* text = Find(params, "top_p")) value.top_p = ParseFloat(*text, "top-p");
    if (const auto* text = Find(params, "seed"))
        value.seed = static_cast<unsigned>(ParseInteger(*text, "seed", 0, 4294967295LL));
    if (const auto* text = Find(params, "samples_per_prompt"))
        value.samples_per_prompt = static_cast<int>(
            ParseInteger(*text, "samples per prompt", 1, kGenerationPreviewMaxSamplesPerPrompt));
    if (const auto* text = Find(params, "repetition_penalty"))
        value.repetition_penalty = ParseFloat(*text, "repetition penalty");
    if (const auto* text = Find(params, "min_p")) value.min_p = ParseFloat(*text, "min-p");
    if (const auto* text = Find(params, "frequency_penalty"))
        value.frequency_penalty = ParseFloat(*text, "frequency penalty");
    if (const auto* text = Find(params, "presence_penalty"))
        value.presence_penalty = ParseFloat(*text, "presence penalty");
    if (const auto* text = Find(params, "no_repeat_ngram"))
        value.no_repeat_ngram = static_cast<int>(ParseInteger(*text, "no-repeat n-gram size", 0, 10));
    if (const auto* text = Find(params, "num_beams"))
        value.num_beams = static_cast<int>(ParseInteger(*text, "beams", 2, 8));
    if (const auto* text = Find(params, "length_penalty")) value.length_penalty = ParseFloat(*text, "length penalty");
    if (const auto* text = Find(params, "every"))
        value.every = static_cast<int>(ParseInteger(*text, "every", 1, 1000000));
    if (const auto* text = Find(params, "every_unit")) value.cadence = ParseChoice(*text, "every unit", kCadences);
    if (const auto* text = Find(params, "baseline")) value.baseline = ParseBool(*text, "baseline");
    if (const auto* text = Find(params, "after_last")) value.after_last = ParseBool(*text, "after last");
    if (const auto* text = Find(params, "weights")) value.weights = ParseChoice(*text, "weights", kWeights);
    if (const auto* text = Find(params, "prompt_set")) value.prompts = ParsePromptSet(*text);
}
}  // namespace

TrainingGenerationPreviewSettings ParseTrainingGenerationPreview(
    const std::map<std::string, std::string>& params) {
    TrainingGenerationPreviewSettings value;
    const auto* enabled = Find(params, "enabled");
    if (!enabled || !ParseBool(*enabled, "enabled")) return value;
    value.enabled = true;
    if (Find(params, "prompts") || Find(params, "every_epochs"))
        Fail("Generation previews: this Data Loader was saved before the Generation tab existed. Open the "
             "Data Loader, set the prompts on its Generation tab and apply");
    ParseGenerationFields(params, value);
    if (const auto error = ValidateTrainingGenerationPreview(value); !error.empty()) Fail(error);
    return value;
}

TrainingGenerationPreviewSettings ParseGenerationSettings(const std::map<std::string, std::string>& params) {
    TrainingGenerationPreviewSettings value;
    if (const auto* enabled = Find(params, "enabled")) value.enabled = ParseBool(*enabled, "enabled");
    ParseGenerationFields(params, value);
    return value;
}

std::map<std::string, std::string> SerializeTrainingGenerationPreview(
    const TrainingGenerationPreviewSettings& s) {
    nlohmann::json stops = s.stop_texts;
    return {
        {Key("enabled"), s.enabled ? "true" : "false"},
        {Key("mode"), GenerationPreviewModeName(s.mode)},
        {Key("template"), GenerationPreviewTemplateName(s.chat_template)},
        {Key("system_prompt"), s.system_prompt},
        {Key("label_system"), s.label_system},
        {Key("label_user"), s.label_user},
        {Key("label_assistant"), s.label_assistant},
        {Key("stop_texts"), stops.dump()},
        {Key("max_new_tokens"), std::to_string(s.max_new_tokens)},
        {Key("decoding"), GenerationPreviewDecodingName(s.decoding)},
        {Key("temperature"), FormatFloat(s.temperature)},
        {Key("top_k"), std::to_string(s.top_k)},
        {Key("top_p"), FormatFloat(s.top_p)},
        {Key("seed"), std::to_string(s.seed)},
        {Key("samples_per_prompt"), std::to_string(s.samples_per_prompt)},
        {Key("repetition_penalty"), FormatFloat(s.repetition_penalty)},
        {Key("min_p"), FormatFloat(s.min_p)},
        {Key("frequency_penalty"), FormatFloat(s.frequency_penalty)},
        {Key("presence_penalty"), FormatFloat(s.presence_penalty)},
        {Key("no_repeat_ngram"), std::to_string(s.no_repeat_ngram)},
        {Key("num_beams"), std::to_string(s.num_beams)},
        {Key("length_penalty"), FormatFloat(s.length_penalty)},
        {Key("every"), std::to_string(s.every)},
        {Key("every_unit"), GenerationPreviewCadenceName(s.cadence)},
        {Key("baseline"), s.baseline ? "true" : "false"},
        {Key("after_last"), s.after_last ? "true" : "false"},
        {Key("weights"), GenerationPreviewWeightsName(s.weights)},
        {Key("prompt_set"), PromptSetJson(s.prompts)},
    };
}

std::string ValidateTrainingGenerationPreview(const TrainingGenerationPreviewSettings& s) {
    if (!s.enabled) return {};
    if (s.prompts.empty() || s.prompts.size() > static_cast<size_t>(kGenerationPreviewMaxPrompts))
        return "Generation previews need 1 to 8 prompts";
    for (size_t i = 0; i < s.prompts.size(); ++i) {
        const auto& prompt = s.prompts[i];
        const std::string which = "prompt " + std::to_string(i + 1);
        if (prompt.text.find_first_not_of(" \t\r\n") == std::string::npos) return "Generation previews: " + which + " is empty";
        if (prompt.text.size() > kGenerationPreviewMaxPromptBytes || prompt.text.find('\0') != std::string::npos)
            return "Generation previews: " + which + " is longer than 4096 bytes";
        switch (prompt.check.kind) {
            case GenerationPreviewCheckKind::Contains:
            case GenerationPreviewCheckKind::NotContains:
                if (prompt.check.value.empty()) return "Generation previews: the check on " + which + " needs its text";
                break;
            case GenerationPreviewCheckKind::Regex:
                if (prompt.check.value.empty()) return "Generation previews: the check on " + which + " needs a pattern";
                try {
                    std::regex pattern(prompt.check.value);
                } catch (const std::regex_error& error) {
                    return "Generation previews: the pattern on " + which + " is not valid (" + error.what() + ")";
                }
                break;
            default:
                break;
        }
    }
    if (s.system_prompt.size() > kGenerationPreviewMaxSystemBytes)
        return "Generation previews: the system prompt is longer than 4096 bytes";
    if (s.mode == GenerationPreviewMode::Chat && s.chat_template == GenerationPreviewTemplate::RoleLines) {
        for (const auto* label : {&s.label_system, &s.label_user, &s.label_assistant})
            if (label->empty() || label->size() > 64 || label->find('\n') != std::string::npos)
                return "Generation previews: role labels must be 1 to 64 characters on one line";
    }
    if (s.stop_texts.size() > static_cast<size_t>(kGenerationPreviewMaxStopTexts))
        return "Generation previews: at most 8 stop texts";
    for (const auto& stop : s.stop_texts)
        if (stop.empty() || stop.size() > 128) return "Generation previews: each stop text must be 1 to 128 bytes";
    if (s.max_new_tokens < 1 || s.max_new_tokens > kGenerationPreviewMaxNewTokens)
        return "Generation previews: max new tokens must be 1 to 1024";
    if (s.decoding == GenerationPreviewDecoding::Sampling) {
        if (!(s.temperature > 0.0f && s.temperature <= 5.0f)) return "Generation previews: temperature must be above 0 and at most 5";
        if (s.top_k < 0) return "Generation previews: top-k must be 0 (off) or more";
        if (!(s.top_p > 0.0f && s.top_p <= 1.0f)) return "Generation previews: top-p must be above 0 and at most 1";
    }
    if (s.samples_per_prompt < 1 || s.samples_per_prompt > kGenerationPreviewMaxSamplesPerPrompt)
        return "Generation previews: samples per prompt must be 1 to 4";
    if (s.decoding != GenerationPreviewDecoding::Sampling && s.samples_per_prompt != 1)
        return "Generation previews: greedy decoding and beam search give one sample per prompt";
    if (!(s.min_p >= 0.0f && s.min_p < 1.0f)) return "Generation previews: min-p must be 0 (off) to below 1";
    if (!(s.frequency_penalty >= 0.0f && s.frequency_penalty <= 2.0f) ||
        !(s.presence_penalty >= 0.0f && s.presence_penalty <= 2.0f))
        return "Generation previews: frequency and presence penalties must be 0 (off) to 2";
    if (s.no_repeat_ngram < 0 || s.no_repeat_ngram > 10)
        return "Generation previews: the no-repeat n-gram size must be 0 (off) to 10";
    if (s.num_beams < 2 || s.num_beams > 8) return "Generation previews: beam search needs 2 to 8 beams";
    if (!(s.length_penalty >= 0.0f && s.length_penalty <= 2.0f))
        return "Generation previews: the length penalty must be 0 to 2";
    if (!(s.repetition_penalty >= 1.0f && s.repetition_penalty <= 2.0f))
        return "Generation previews: repetition penalty must be 1 (off) to 2";
    if (s.every < 1) return "Generation previews: every must be 1 or more";
    if (s.cadence == GenerationPreviewCadence::Epochs && s.every > 10000)
        return "Generation previews: every must be at most 10000 epochs";
    return {};
}

RenderedPreviewPrompt RenderGenerationPreviewPrompt(
    const TrainingGenerationPreviewSettings& s, const std::string& prompt) {
    return RenderGenerationConversation(s, {}, prompt);
}

RenderedPreviewPrompt RenderGenerationConversation(
    const TrainingGenerationPreviewSettings& s,
    const std::vector<GenerationConversationTurn>& history, const std::string& message) {
    RenderedPreviewPrompt rendered;
    if (s.mode == GenerationPreviewMode::Continuation) {
        AddSegment(rendered, message, false);
        return rendered;
    }
    const bool has_system = !s.system_prompt.empty();
    // A user turn and the assistant's cue; `reply` (an earlier turn's answer)
    // follows the cue and closes the turn.
    const auto add_turn = [&](const std::string& user, const std::string* reply) {
        switch (s.chat_template) {
            case GenerationPreviewTemplate::RoleLines:
                AddSegment(rendered, s.label_user + " ", true);
                AddSegment(rendered, user, false);
                AddSegment(rendered, "\n" + s.label_assistant + "\n", true);
                if (reply) {
                    AddSegment(rendered, *reply, false);
                    AddSegment(rendered, "\n", true);
                }
                break;
            case GenerationPreviewTemplate::ChatML:
                AddSegment(rendered, "<|im_start|>user\n", true);
                AddSegment(rendered, user, false);
                AddSegment(rendered, "<|im_end|>\n<|im_start|>assistant\n", true);
                if (reply) {
                    AddSegment(rendered, *reply, false);
                    AddSegment(rendered, "<|im_end|>\n", true);
                }
                break;
            case GenerationPreviewTemplate::Llama3:
                AddSegment(rendered, "<|start_header_id|>user<|end_header_id|>\n\n", true);
                AddSegment(rendered, user, false);
                AddSegment(rendered, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", true);
                if (reply) {
                    AddSegment(rendered, *reply, false);
                    AddSegment(rendered, "<|eot_id|>", true);
                }
                break;
        }
    };
    if (has_system) {
        switch (s.chat_template) {
            case GenerationPreviewTemplate::RoleLines:
                AddSegment(rendered, s.label_system + " ", true);
                AddSegment(rendered, s.system_prompt, false);
                AddSegment(rendered, "\n", true);
                break;
            case GenerationPreviewTemplate::ChatML:
                AddSegment(rendered, "<|im_start|>system\n", true);
                AddSegment(rendered, s.system_prompt, false);
                AddSegment(rendered, "<|im_end|>\n", true);
                break;
            case GenerationPreviewTemplate::Llama3:
                AddSegment(rendered, "<|start_header_id|>system<|end_header_id|>\n\n", true);
                AddSegment(rendered, s.system_prompt, false);
                AddSegment(rendered, "<|eot_id|>", true);
                break;
        }
    }
    for (const auto& turn : history) add_turn(turn.user, &turn.assistant);
    add_turn(message, nullptr);
    return rendered;
}

std::vector<std::string> GenerationPreviewTemplateStopTexts(const TrainingGenerationPreviewSettings& s) {
    if (s.mode == GenerationPreviewMode::Continuation) return {};
    switch (s.chat_template) {
        case GenerationPreviewTemplate::RoleLines: return {"\n" + s.label_user};
        case GenerationPreviewTemplate::ChatML: return {"<|im_end|>"};
        case GenerationPreviewTemplate::Llama3: return {"<|eot_id|>"};
    }
    return {};
}

std::vector<std::string> GenerationPreviewStopTexts(const TrainingGenerationPreviewSettings& s) {
    auto stops = GenerationPreviewTemplateStopTexts(s);
    for (const auto& stop : s.stop_texts)
        if (std::find(stops.begin(), stops.end(), stop) == stops.end()) stops.push_back(stop);
    return stops;
}

bool ShouldRunPreviewAfterEpoch(const TrainingGenerationPreviewSettings& s, int completed_epoch, int total_epochs) {
    if (!s.enabled || completed_epoch < 1) return false;
    if (s.after_last && completed_epoch == total_epochs) return true;
    return s.cadence == GenerationPreviewCadence::Epochs && completed_epoch % s.every == 0;
}

bool ShouldRunPreviewAfterStep(const TrainingGenerationPreviewSettings& s, long long completed_step) {
    return s.enabled && s.cadence == GenerationPreviewCadence::Steps && completed_step > 0 &&
           completed_step % s.every == 0;
}

GenerationPreviewMode SuggestedGenerationPreviewMode(const std::string& supervision) {
    return supervision == "response_only" ? GenerationPreviewMode::Chat : GenerationPreviewMode::Continuation;
}

}  // namespace cyxwiz
