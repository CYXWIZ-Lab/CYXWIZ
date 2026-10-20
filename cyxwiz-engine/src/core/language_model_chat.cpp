#include "language_model_chat.h"

#include "executable_model.h"
#include "generation_preview_presentation.h"
#include "../inference/text_inference_input.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tokenizer.h>
#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace cyxwiz {
namespace {

const char* TokenizerKindName(TokenizerType type) {
    switch (type) {
        case TokenizerType::Whitespace: return "Whitespace";
        case TokenizerType::Word: return "Word";
        case TokenizerType::Character: return "Character";
        case TokenizerType::ByteBPE: return "Byte BPE";
        case TokenizerType::WordPiece: return "WordPiece";
        case TokenizerType::SentencePieceBPE: return "SentencePiece BPE";
        case TokenizerType::SentencePieceUnigram: return "SentencePiece Unigram";
    }
    return "Tokenizer";
}

// Evaluation mode while generating; a layer stack gets each module's own mode
// back (frozen or mixed modules), any other model is left in evaluation mode.
class EvaluationScope {
public:
    explicit EvaluationScope(IExecutableModel& model) : sequential_(model.AsSequentialModel()) {
        if (sequential_)
            for (size_t i = 0; i < sequential_->Size(); ++i) modes_.push_back(sequential_->GetModule(i)->IsTraining());
        model.SetTraining(false);
    }
    ~EvaluationScope() {
        if (!sequential_) return;
        for (size_t i = 0; i < modes_.size(); ++i) sequential_->GetModule(i)->SetTraining(modes_[i]);
    }

private:
    SequentialModel* sequential_ = nullptr;
    std::vector<bool> modes_;
};

GenerationPreviewSample AsSample(const LanguageModelReply& reply) {
    GenerationPreviewSample sample;
    sample.stop_reason = LanguageModelGenerationStopReasonName(reply.stop_reason);
    sample.stop_text = reply.stop_text;
    sample.new_tokens = static_cast<int>(reply.new_token_ids.size());
    return sample;
}

nlohmann::json SettingsJson(const TrainingGenerationPreviewSettings& s) {
    nlohmann::json json = {
        {"mode", GenerationPreviewModeName(s.mode)},
        {"template", s.mode == GenerationPreviewMode::Chat ? GenerationPreviewTemplateName(s.chat_template) : ""},
        {"stop_texts", GenerationPreviewStopTexts(s)}};
    AddGenerationDecodingFields(s, json);
    return json;
}

bool WriteText(const std::filesystem::path& path, const std::string& text, std::string& error) {
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream output(path, std::ios::binary);
    if (output) output << text;
    if (!output) {
        error = "cannot write " + path.string();
        return false;
    }
    return true;
}

}  // namespace

GenerationTokenizer::GenerationTokenizer() : package_(std::make_unique<TextTokenizerPackage>()) {}
GenerationTokenizer::~GenerationTokenizer() = default;

std::shared_ptr<GenerationTokenizer> GenerationTokenizer::Load(const std::string& config_json,
                                                              const std::string& vocabulary,
                                                              const std::string& model_artifact, std::string& error) {
    std::shared_ptr<GenerationTokenizer> loaded(new GenerationTokenizer());
    if (!LoadTextTokenizerPackage(config_json, vocabulary, model_artifact, *loaded->package_, error) ||
        !loaded->package_->tokenizer || !loaded->package_->has_vocabulary) {
        if (error.empty()) error = "the tokenizer has no vocabulary";
        return nullptr;
    }
    // Prompts are encoded as in training windows and generation previews:
    // no padding, truncation or added markers.
    auto& tokenizer = *loaded->package_->tokenizer;
    tokenizer.SetPadding(false);
    tokenizer.SetTruncation(false);
    tokenizer.SetAddBos(false);
    tokenizer.SetAddEos(false);
    loaded->kind_ = TokenizerKindName(tokenizer.GetType());
    const auto hash = Utilities::HashText(vocabulary, "sha256");
    if (hash.success) loaded->sha256_ = hash.sha256_hash;
    return loaded;
}

std::vector<int64_t> GenerationTokenizer::Encode(const std::string& text) const {
    const auto encoded = package_->tokenizer->Encode(text);
    std::vector<int64_t> ids(encoded.begin(), encoded.end());
    const int pad = package_->tokenizer->GetPadId();
    if (pad >= 0 && std::find(ids.begin(), ids.end(), pad) != ids.end())
        throw std::invalid_argument("The text encodes to the padding token");
    return ids;
}

std::string GenerationTokenizer::Decode(const std::vector<int64_t>& ids) const {
    const int64_t eos = EosId();
    const int64_t size = static_cast<int64_t>(VocabularySize());
    std::vector<int> values;
    values.reserve(ids.size());
    for (const int64_t id : ids) {
        if (id < 0 || id >= size || id > static_cast<int64_t>((std::numeric_limits<int>::max)()))
            throw std::runtime_error("The model returned an invalid token ID");
        if (id != eos) values.push_back(static_cast<int>(id));
    }
    return package_->tokenizer->Decode(values);
}

std::string GenerationTokenizer::TokenText(int64_t id) const {
    const auto& words = package_->tokenizer->GetVocabulary().GetWords();
    if (id < 0 || static_cast<size_t>(id) >= words.size()) return "?";
    if (id == EosId()) return "<end>";
    return Decode({id});
}

int64_t GenerationTokenizer::EosId() const { return package_->tokenizer->GetVocabulary().EosIndex(); }
size_t GenerationTokenizer::VocabularySize() const { return package_->tokenizer->GetVocabulary().Size(); }

LanguageModelPrompt BuildLanguageModelPrompt(
    const TrainingGenerationPreviewSettings& settings,
    const std::vector<GenerationConversationTurn>& history, bool send_history, const std::string& message,
    const std::function<std::vector<int64_t>(const std::string&)>& encode, size_t context) {
    LanguageModelPrompt prompt;
    const bool chat = settings.mode == GenerationPreviewMode::Chat;
    size_t first = chat && send_history ? 0 : history.size();
    const size_t wanted = static_cast<size_t>(std::max(1, settings.max_new_tokens));
    // Leave out the oldest turns while the reply's tokens do not fit.
    while (true) {
        const std::vector<GenerationConversationTurn> sent(history.begin() + static_cast<std::ptrdiff_t>(first),
                                                           history.end());
        prompt.rendered = RenderGenerationConversation(settings, sent, message);
        prompt.ids = encode(prompt.rendered.text);
        if (first >= history.size() || context == 0 || prompt.ids.size() + wanted <= context) break;
        ++first;
    }
    prompt.turns_sent = history.size() - first;
    prompt.turns_left_out = chat && send_history ? first : 0;
    if (prompt.ids.empty()) {
        prompt.error = "Type a message first";
    } else if (context > 0 && prompt.ids.size() >= context) {
        prompt.error = "The prompt is " + std::to_string(prompt.ids.size()) + " tokens; it must leave room in the " +
                       std::to_string(context) + "-token context";
    } else {
        prompt.new_tokens_fit = context == 0 ? wanted : std::min(wanted, context - prompt.ids.size());
    }
    return prompt;
}

LanguageModelReply GenerateLanguageModelReply(
    IExecutableModel& model, const GenerationTokenizer& tokenizer,
    const TrainingGenerationPreviewSettings& s, const std::string& message, LanguageModelPrompt prompt,
    const LanguageModelRunOptions& options, const std::function<bool()>& cancel,
    const std::function<void(const std::string& text, size_t tokens)>& on_text) {
    if (!prompt.error.empty()) throw std::invalid_argument(prompt.error);
    LanguageModelReply reply;
    reply.message = message;
    reply.max_new_tokens = static_cast<int>(prompt.new_tokens_fit);
    reply.seed = options.seed;
    reply.sampled = s.decoding == GenerationPreviewDecoding::Sampling;

    LanguageModelGenerationConfig config;
    config.max_new_tokens = prompt.new_tokens_fit;
    config.max_context_tokens = static_cast<size_t>(std::max(0, s.context));
    config.eos_token_id = options.ignore_stops ? -1 : tokenizer.EosId();
    config.include_prompt = false;
    config.should_cancel = cancel;
    config.use_kv_cache = options.use_kv_cache;
    config.reuse_cache = options.reuse_cache;
    ApplyGenerationDecoding(s, config);
    const auto stops = options.ignore_stops ? std::vector<std::string>() : GenerationPreviewStopTexts(s);
    std::vector<int64_t> so_far;
    config.on_step = [&](const LanguageModelGenerationStep& step) {
        so_far.push_back(step.token_id);
        if (on_text) on_text(tokenizer.Decode(so_far), so_far.size());
    };
    if (!stops.empty()) {
        config.stop_check = [&](const std::vector<int64_t>& ids) {
            const auto text = tokenizer.Decode(ids);
            for (const auto& stop : stops)
                if (text.find(stop) != std::string::npos) {
                    reply.stop_text = stop;
                    return true;
                }
            return false;
        };
    }

    const auto start = std::chrono::steady_clock::now();
    LanguageModelGenerationResult result;
    {
        EvaluationScope modes(model);
        result = GenerateTokenIdsWithReport(model, prompt.ids, config, options.seed);
    }
    reply.used_kv_cache = result.used_kv_cache;
    reply.reused_cache_tokens = result.reused_cache_tokens;
    reply.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    reply.new_token_ids = std::move(result.new_token_ids);
    reply.steps = std::move(result.steps);
    // Each step keeps its five most likely candidates (the token inspector);
    // the full distribution is the whole vocabulary.
    for (auto& step : reply.steps) {
        auto& candidates = step.candidates;
        const size_t keep = std::min<size_t>(5, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep), candidates.end(),
                          [](const NextTokenCandidate& a, const NextTokenCandidate& b) {
                              return a.probability > b.probability;
                          });
        candidates.resize(keep);
        candidates.shrink_to_fit();
    }
    reply.stop_reason = result.stop_reason;
    reply.text = tokenizer.Decode(reply.new_token_ids);
    if (reply.stop_reason == LanguageModelGenerationStopReason::StopText && !reply.stop_text.empty())
        reply.text = reply.text.substr(0, reply.text.find(reply.stop_text));
    else
        reply.stop_text.clear();
    reply.repeated_phrase = DetectRepeatedPhrase(reply.text);
    reply.prompt = std::move(prompt);
    return reply;
}

LanguageModelSpeed MeasureLanguageModelSpeed(IExecutableModel& model, const GenerationTokenizer& tokenizer,
                                             const TrainingGenerationPreviewSettings& settings,
                                             const LanguageModelPrompt& prompt, const std::function<bool()>& cancel) {
    auto greedy = settings;
    greedy.decoding = GenerationPreviewDecoding::Greedy;
    LanguageModelSpeed speed;
    const auto* sequential = model.AsSequentialModel();
    speed.cache_supported = sequential && sequential->SupportsIncrementalDecoding();
    const auto run = [&](bool cache) {
        LanguageModelRunOptions options;
        options.use_kv_cache = cache;
        options.ignore_stops = true;
        const auto reply = GenerateLanguageModelReply(model, tokenizer, greedy, "", prompt, options, cancel, {});
        if (reply.stop_reason == LanguageModelGenerationStopReason::UserCancelled)
            throw std::runtime_error("The speed measurement was cancelled");
        speed.tokens = reply.new_token_ids.size();
        return reply.seconds > 0.0 ? reply.new_token_ids.size() / reply.seconds : 0.0;
    };
    if (speed.cache_supported) speed.cached_tokens_per_second = run(true);
    speed.uncached_tokens_per_second = run(false);
    return speed;
}

std::string LanguageModelSpeedText(const LanguageModelSpeed& speed) {
    char buffer[160];
    if (!speed.cache_supported) {
        std::snprintf(buffer, sizeof(buffer), "%.0f tokens/s (%zu tokens; this model has no KV cache)",
                      speed.uncached_tokens_per_second, speed.tokens);
    } else {
        const double ratio = speed.uncached_tokens_per_second > 0.0
                                 ? speed.cached_tokens_per_second / speed.uncached_tokens_per_second
                                 : 0.0;
        std::snprintf(buffer, sizeof(buffer),
                      "With the KV cache %.0f tokens/s \xC2\xB7 without %.0f tokens/s \xC2\xB7 %.1fx (%zu tokens)",
                      speed.cached_tokens_per_second, speed.uncached_tokens_per_second, ratio, speed.tokens);
    }
    return buffer;
}

std::string LanguageModelReplyStopLabel(const LanguageModelReply& reply) {
    const auto sample = AsSample(reply);
    if (sample.stop_reason == "eos") return "Stopped by itself";
    if (sample.stop_reason == "user_cancel") return "Stopped";
    const auto label = GenerationPreviewStopLabel(sample, reply.max_new_tokens);
    const auto dot = label.find(" \xC2\xB7 ");
    return dot == std::string::npos ? label : label.substr(0, dot);
}

std::string LanguageModelReplySpeedLabel(const LanguageModelReply& reply) {
    const size_t tokens = reply.new_token_ids.size();
    char buffer[96];
    if (reply.seconds > 0.0 && tokens > 0)
        std::snprintf(buffer, sizeof(buffer), "%zu %s \xC2\xB7 %.1f s \xC2\xB7 %.0f tokens/s", tokens,
                      tokens == 1 ? "token" : "tokens", reply.seconds, tokens / reply.seconds);
    else
        std::snprintf(buffer, sizeof(buffer), "%zu %s", tokens, tokens == 1 ? "token" : "tokens");
    return buffer;
}

std::vector<GenerationConversationTurn> LanguageModelHistory(const std::vector<LanguageModelReply>& replies) {
    std::vector<GenerationConversationTurn> history;
    history.reserve(replies.size());
    for (const auto& reply : replies) history.push_back({reply.message, reply.text});
    return history;
}

bool WriteLanguageModelChatReport(const std::filesystem::path& path, const std::vector<LanguageModelReply>& replies,
                                  const TrainingGenerationPreviewSettings& settings,
                                  const LanguageModelReportSource& source, std::string& error) {
    nlohmann::json report = SettingsJson(settings);
    report.update({{"schema", kGenerationPreviewReportSchema}, {"run_id", source.run_id}, {"point", "chat"},
                   {"epoch", source.epoch}, {"total_epochs", 0}, {"step", source.step},
                   {"weights", source.weights}, {"seed", settings.seed},
                   {"max_new_tokens", settings.max_new_tokens}, {"context", source.context},
                   {"tokenizer_sha256", source.tokenizer_sha256}, {"cancelled", false},
                   {"samples", nlohmann::json::array()}});
    if (source.validation_loss) report["validation_loss"] = *source.validation_loss;
    double elapsed_ms = 0.0;
    for (size_t i = 0; i < replies.size(); ++i) {
        const auto& reply = replies[i];
        elapsed_ms += reply.seconds * 1000.0;
        report["samples"].push_back({{"prompt_index", i}, {"sample_index", 0}, {"prompt", reply.message},
                                     {"rendered_prompt", reply.prompt.rendered.text},
                                     {"prompt_token_ids", reply.prompt.ids}, {"new_token_ids", reply.new_token_ids},
                                     {"text", reply.text},
                                     {"stop_reason", LanguageModelGenerationStopReasonName(reply.stop_reason)},
                                     {"stop_text", reply.stop_text}});
    }
    report["elapsed_ms"] = elapsed_ms;
    return WriteText(path, report.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) + "\n", error);
}

bool ExportLanguageModelChatJsonl(const std::filesystem::path& path, const std::vector<LanguageModelReply>& replies,
                                  const TrainingGenerationPreviewSettings& settings, std::string& error) {
    std::string text;
    for (size_t i = 0; i < replies.size(); ++i) {
        const auto& reply = replies[i];
        nlohmann::json line = SettingsJson(settings);
        line.update({{"turn", i + 1}, {"message", reply.message}, {"rendered_prompt", reply.prompt.rendered.text},
                     {"output", reply.text}, {"stop_reason", LanguageModelGenerationStopReasonName(reply.stop_reason)},
                     {"prompt_tokens", reply.prompt.ids.size()}, {"new_tokens", reply.new_token_ids.size()},
                     {"seconds", reply.seconds}, {"seed", reply.seed}});
        if (!reply.stop_text.empty()) line["stop_text"] = reply.stop_text;
        text += line.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) + "\n";
    }
    return WriteText(path, text, error);
}

}  // namespace cyxwiz
