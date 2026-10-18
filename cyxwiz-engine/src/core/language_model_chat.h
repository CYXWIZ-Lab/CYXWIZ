#pragma once

#include "language_model_generation.h"
#include "training_generation_preview_settings.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class IExecutableModel;
struct TextTokenizerPackage;

// Using a trained language model directly (TOFIX159, LM Generation panel,
// design approved 2026-10-18): chat with a fine-tuned model or let a base
// model continue text. Generic for any causal language model the Engine
// trains or imports; the template, labels and stop texts are the Data
// Loader's generation settings. Pure: no ImGui.

// Encodes prompts and decodes new tokens with a model's tokenizer.
class GenerationTokenizer {
public:
    // tokenizer/config.json text, the vocabulary text and the optional model
    // artifact (a .cyxmodel's tokenizer assets, or a training run's
    // sequence_batch tokenizer fields). Nullptr with `error` when unusable.
    static std::shared_ptr<GenerationTokenizer> Load(const std::string& config_json, const std::string& vocabulary,
                                                     const std::string& model_artifact, std::string& error);
    ~GenerationTokenizer();

    std::vector<int64_t> Encode(const std::string& text) const;
    std::string Decode(const std::vector<int64_t>& ids) const;  // End-of-sequence left out.
    std::string TokenText(int64_t id) const;                    // One token, for the token inspector.
    int64_t EosId() const;
    size_t VocabularySize() const;
    const std::string& Kind() const { return kind_; }      // "Byte BPE", "Word", ...
    const std::string& Sha256() const { return sha256_; }  // Of the vocabulary text.

private:
    GenerationTokenizer();
    std::unique_ptr<TextTokenizerPackage> package_;
    std::string kind_;
    std::string sha256_;
};

// The prompt for one message: the conversation rendered and encoded, the
// oldest turns left out until the reply's tokens fit the context.
struct LanguageModelPrompt {
    RenderedPreviewPrompt rendered;
    std::vector<int64_t> ids;
    size_t turns_sent = 0;
    size_t turns_left_out = 0;
    size_t new_tokens_fit = 0;  // min(max new tokens, context - prompt).
    std::string error;          // Empty when the message fits.
};

LanguageModelPrompt BuildLanguageModelPrompt(
    const TrainingGenerationPreviewSettings& settings,
    const std::vector<GenerationConversationTurn>& history, bool send_history, const std::string& message,
    const std::function<std::vector<int64_t>(const std::string&)>& encode, size_t context);

struct LanguageModelReply {
    std::string message;  // As entered (a user message, or the text to continue).
    std::string text;     // The reply, a matched stop text cut off.
    LanguageModelPrompt prompt;
    std::vector<int64_t> new_token_ids;
    std::vector<LanguageModelGenerationStep> steps;
    LanguageModelGenerationStopReason stop_reason = LanguageModelGenerationStopReason::MaxTokens;
    std::string stop_text;
    int max_new_tokens = 0;
    unsigned seed = 0;
    double seconds = 0.0;
    std::string repeated_phrase;
};

// Generates on the calling thread (model errors throw). `on_text` gets the
// decoded reply so far and its token count after each token.
LanguageModelReply GenerateLanguageModelReply(
    IExecutableModel& model, const GenerationTokenizer& tokenizer,
    const TrainingGenerationPreviewSettings& settings, const std::string& message, LanguageModelPrompt prompt,
    unsigned seed, const std::function<bool()>& cancel,
    const std::function<void(const std::string& text, size_t tokens)>& on_text);

// "Stopped by itself", "Hit the 128-token limit", "Stopped at \"</answer>\"", "Stopped".
std::string LanguageModelReplyStopLabel(const LanguageModelReply& reply);
// "128 tokens · 3.4 s · 37 tokens/s".
std::string LanguageModelReplySpeedLabel(const LanguageModelReply& reply);
// The earlier turns of a conversation, as the next prompt sends them.
std::vector<GenerationConversationTurn> LanguageModelHistory(const std::vector<LanguageModelReply>& replies);

// Facts about the loaded model that a conversation report records.
struct LanguageModelReportSource {
    std::string run_id;   // The training run, or the model's name.
    int epoch = 0;
    long long step = 0;
    std::string weights;  // "best", "latest", "session", "package"
    std::optional<double> validation_loss;
    int context = 0;
    std::string tokenizer_sha256;
};

// A conversation as a schema 2 generation report (point "chat", one sample
// per reply), readable by the Generations view. Rewritten after each reply.
bool WriteLanguageModelChatReport(const std::filesystem::path& path, const std::vector<LanguageModelReply>& replies,
                                  const TrainingGenerationPreviewSettings& settings,
                                  const LanguageModelReportSource& source, std::string& error);
// One JSON object per reply.
bool ExportLanguageModelChatJsonl(const std::filesystem::path& path, const std::vector<LanguageModelReply>& replies,
                                  const TrainingGenerationPreviewSettings& settings, std::string& error);

}  // namespace cyxwiz
