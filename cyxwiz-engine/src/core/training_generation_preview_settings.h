#pragma once
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

// Generation previews during training (TOFIX159, design approved 2026-10-18):
// sample the model on a few prompts at a cadence and keep the outputs per run.
// Generic for any language model: plain continuation for base models, or a chat
// template for instruction-tuned ones.

enum class GenerationPreviewMode { Continuation, Chat };
enum class GenerationPreviewTemplate { RoleLines, ChatML, Llama3 };
enum class GenerationPreviewDecoding { Greedy, Sampling };
enum class GenerationPreviewCadence { Epochs, Steps };
enum class GenerationPreviewWeights { Current, Best };
enum class GenerationPreviewCheckKind { None, Contains, NotContains, Regex, EndsWithStop };

struct GenerationPreviewCheck {
    GenerationPreviewCheckKind kind = GenerationPreviewCheckKind::None;
    std::string value;  // Text for Contains / NotContains, the pattern for Regex.
};

struct GenerationPreviewPrompt {
    std::string text;  // The whole prompt (continuation) or the user message (chat).
    GenerationPreviewCheck check;
};

constexpr int kGenerationPreviewMaxPrompts = 8;
constexpr size_t kGenerationPreviewMaxPromptBytes = 4096;
constexpr size_t kGenerationPreviewMaxSystemBytes = 4096;
constexpr int kGenerationPreviewMaxNewTokens = 1024;
constexpr int kGenerationPreviewMaxStopTexts = 8;
constexpr int kGenerationPreviewMaxSamplesPerPrompt = 4;

struct TrainingGenerationPreviewSettings {
    bool enabled = false;
    GenerationPreviewMode mode = GenerationPreviewMode::Continuation;
    GenerationPreviewTemplate chat_template = GenerationPreviewTemplate::RoleLines;
    std::string system_prompt;
    std::string label_system = "System:";
    std::string label_user = "User:";
    std::string label_assistant = "Assistant:";
    std::vector<std::string> stop_texts;  // Added by the user; the template adds its own.
    int max_new_tokens = 64;
    GenerationPreviewDecoding decoding = GenerationPreviewDecoding::Greedy;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    unsigned seed = 1234;
    int samples_per_prompt = 1;
    float repetition_penalty = 1.0f;  // 1 = off.
    int every = 1;
    GenerationPreviewCadence cadence = GenerationPreviewCadence::Epochs;
    bool baseline = false;    // Before the first update.
    bool after_last = true;   // After the last epoch even off the cadence.
    GenerationPreviewWeights weights = GenerationPreviewWeights::Current;
    std::vector<GenerationPreviewPrompt> prompts;

    // Resolved from the selected training dataset, never entered independently.
    std::string vocabulary_artifact;
    int tokenizer_type = -1;
    bool lowercase = false;
    int context = 0;
};

const char* GenerationPreviewModeName(GenerationPreviewMode mode);
const char* GenerationPreviewTemplateName(GenerationPreviewTemplate value);
const char* GenerationPreviewTemplateLabel(GenerationPreviewTemplate value);
const char* GenerationPreviewDecodingName(GenerationPreviewDecoding value);
const char* GenerationPreviewCadenceName(GenerationPreviewCadence value);
const char* GenerationPreviewWeightsName(GenerationPreviewWeights value);
const char* GenerationPreviewCheckKindName(GenerationPreviewCheckKind kind);

// Reads the Data Loader's generation_preview_* parameters. Throws
// std::invalid_argument with a message for the user when a value is out of
// range, and when the graph still carries the pre-TOFIX159 keys.
TrainingGenerationPreviewSettings ParseTrainingGenerationPreview(
    const std::map<std::string, std::string>& params);

// The parameters that ParseTrainingGenerationPreview reads back (the resolved
// dataset fields are not stored).
std::map<std::string, std::string> SerializeTrainingGenerationPreview(
    const TrainingGenerationPreviewSettings& settings);

// Checks prompts, stop texts, checks and ranges; returns "" when valid.
std::string ValidateTrainingGenerationPreview(const TrainingGenerationPreviewSettings& settings);

// A prompt as the model sees it. Segments mark which text the template added,
// so the dialog can show it in another tone.
struct RenderedPreviewSegment {
    std::string text;
    bool from_template = false;
};
struct RenderedPreviewPrompt {
    std::string text;
    std::vector<RenderedPreviewSegment> segments;
};
RenderedPreviewPrompt RenderGenerationPreviewPrompt(
    const TrainingGenerationPreviewSettings& settings, const std::string& prompt);

// A conversation as the model sees it: the system prompt once, each earlier
// turn (user message and the reply), then `message` and the assistant's cue.
// Continuation mode ignores the history and gives `message` as it is.
struct GenerationConversationTurn {
    std::string user;
    std::string assistant;
};
RenderedPreviewPrompt RenderGenerationConversation(
    const TrainingGenerationPreviewSettings& settings,
    const std::vector<GenerationConversationTurn>& history, const std::string& message);

// Stop texts the chat template implies (the next user turn or its end marker).
std::vector<std::string> GenerationPreviewTemplateStopTexts(
    const TrainingGenerationPreviewSettings& settings);
// Template stop texts followed by the user's, without duplicates.
std::vector<std::string> GenerationPreviewStopTexts(
    const TrainingGenerationPreviewSettings& settings);

// Cadence. completed_epoch counts from 1; total_epochs is the run's epoch count.
bool ShouldRunPreviewAfterEpoch(const TrainingGenerationPreviewSettings& settings,
                                int completed_epoch, int total_epochs);
bool ShouldRunPreviewAfterStep(const TrainingGenerationPreviewSettings& settings,
                               long long completed_step);

// The dataset metadata value cyxwiz.token_windows.supervision suggests a mode:
// "response_only" -> Chat, anything else -> Continuation.
GenerationPreviewMode SuggestedGenerationPreviewMode(const std::string& supervision);

} // namespace cyxwiz
