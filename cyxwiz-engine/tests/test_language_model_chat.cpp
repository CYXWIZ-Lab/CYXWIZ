// LM Generation panel core (TOFIX159): conversation rendering, the
// tokenizer wrapper, prompts trimmed to the context, replies and their
// reports.
#include "core/executable_model.h"
#include "core/generation_preview_presentation.h"
#include "core/language_model_chat.h"
#include "core/language_model_checkpoints.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tokenizer.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

// Vocabulary [PAD] [UNK] [BOS] [EOS] one two: emits `script` in order.
class ScriptedLogits : public cyxwiz::Module {
public:
    std::vector<int> script;
    size_t produced = 0;
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        Require(!IsTraining(), "generation runs in evaluation mode");
        const size_t length = input.Shape()[1];
        std::vector<float> logits(length * 6, -10.0f);
        const int next = script[std::min(script.size() - 1, produced++)];
        logits[(length - 1) * 6 + next] = 10.0f;
        return cyxwiz::Tensor({1, length, 6}, logits.data());
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& g) override { return g; }
    std::string GetName() const override { return "ScriptedLogits"; }
};

cyxwiz::TrainingGenerationPreviewSettings ChatSettings() {
    cyxwiz::TrainingGenerationPreviewSettings s;
    s.mode = cyxwiz::GenerationPreviewMode::Chat;
    s.system_prompt = "Be brief.";
    s.label_assistant = "Bot:";
    s.max_new_tokens = 4;
    s.context = 20;
    return s;
}

cyxwiz::LanguageModelRunOptions Seed(unsigned seed) {
    cyxwiz::LanguageModelRunOptions options;
    options.seed = seed;
    return options;
}

std::vector<int64_t> Words(const std::string& text) {  // One ID per space-separated word.
    std::vector<int64_t> ids;
    std::istringstream in(text);
    for (std::string word; in >> word;) ids.push_back(4);
    return ids;
}

void TestConversationRendering() {
    auto s = ChatSettings();
    const std::vector<cyxwiz::GenerationConversationTurn> history = {{"hi", "hello"}};
    Require(cyxwiz::RenderGenerationConversation(s, history, "and you?").text ==
                "System: Be brief.\nUser: hi\nBot:\nhello\nUser: and you?\nBot:\n",
            "role lines: system once, each earlier turn closed, then the cue");
    Require(cyxwiz::RenderGenerationPreviewPrompt(s, "hi").text == "System: Be brief.\nUser: hi\nBot:\n",
            "a single prompt renders as before");
    s.chat_template = cyxwiz::GenerationPreviewTemplate::ChatML;
    Require(cyxwiz::RenderGenerationConversation(s, history, "x").text ==
                "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                "hello<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n",
            "ChatML conversation");
    s.chat_template = cyxwiz::GenerationPreviewTemplate::Llama3;
    s.system_prompt.clear();
    Require(cyxwiz::RenderGenerationConversation(s, history, "x").text ==
                "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                "\n\nhello<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nx<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "Llama 3 conversation without a system prompt");
    s.mode = cyxwiz::GenerationPreviewMode::Continuation;
    const auto continuation = cyxwiz::RenderGenerationConversation(s, history, "In the beginning");
    Require(continuation.text == "In the beginning" && continuation.segments.size() == 1 &&
                !continuation.segments[0].from_template,
            "continuation gives the text as it is, without history");
}

std::shared_ptr<cyxwiz::GenerationTokenizer> TrainingTokenizer() {
    cyxwiz::Vocabulary vocab;
    vocab.SetVocabulary({"[PAD]", "[UNK]", "[BOS]", "[EOS]", "one", "two"});
    std::ostringstream serialized;
    Require(vocab.SaveToStream(serialized), "vocabulary serialize");
    // The tokenizer fields a training run of token windows records.
    const std::string config = R"({"version":"1.0","source":"training_token_windows","effective":)"
                               R"({"tokenizer_type":"0","lowercase":"false","max_length":20,)"
                               R"("vocab_file_packaged":"tokenizer/vocab.txt"}})";
    std::string error;
    auto tokenizer = cyxwiz::GenerationTokenizer::Load(config, serialized.str(), "", error);
    Require(tokenizer != nullptr, "training tokenizer loads: " + error);
    return tokenizer;
}

void TestTokenizer() {
    const auto tokenizer = TrainingTokenizer();
    Require(tokenizer->Encode("one two one") == std::vector<int64_t>({4, 5, 4}), "encode without markers or padding");
    Require(tokenizer->Decode({4, 3, 5}) == "one two", "decode leaves end-of-sequence out");
    Require(tokenizer->EosId() == 3 && tokenizer->VocabularySize() == 6, "eos and size");
    Require(tokenizer->TokenText(5) == "two" && tokenizer->TokenText(3) == "<end>", "token text");
    Require(tokenizer->Kind() == "Whitespace" && tokenizer->Sha256().size() == 64, "kind and hash");
    std::string error;
    Require(cyxwiz::GenerationTokenizer::Load(R"({"method":"word"})", "", "", error) == nullptr && !error.empty(),
            "a tokenizer without a vocabulary is refused with a reason");
}

void TestPromptTrimming() {
    const auto s = ChatSettings();  // max new tokens 4
    std::vector<cyxwiz::GenerationConversationTurn> history = {
        {"a b c", "d e f"}, {"g h", "i j"}, {"k", "l"}};
    // "System: Be brief." = 3 words; a turn "User: <u> Bot: <a>"; the message turn "User: m Bot:".
    auto all = cyxwiz::BuildLanguageModelPrompt(s, history, true, "m", Words, 100);
    Require(all.turns_sent == 3 && all.turns_left_out == 0 && all.new_tokens_fit == 4, "everything fits");
    const size_t full = all.ids.size();
    auto trimmed = cyxwiz::BuildLanguageModelPrompt(s, history, true, "m", Words, full + 4 - 1);
    Require(trimmed.turns_left_out == 1 && trimmed.turns_sent == 2 && trimmed.error.empty(),
            "the oldest turn is left out when the reply would not fit");
    auto none = cyxwiz::BuildLanguageModelPrompt(s, history, false, "m", Words, 100);
    Require(none.turns_sent == 0 && none.turns_left_out == 0, "history off sends the message alone");
    auto tight = cyxwiz::BuildLanguageModelPrompt(s, {}, true, "m", Words, 9);
    Require(tight.error.empty() && tight.new_tokens_fit == 9 - tight.ids.size(),
            "new tokens are capped by the context the prompt leaves");
    auto too_long = cyxwiz::BuildLanguageModelPrompt(s, {}, true, "one two three four five six", Words, 8);
    Require(!too_long.error.empty() && too_long.new_tokens_fit == 0, "a prompt over the context is refused");
}

void TestReplies() {
    const auto tokenizer = TrainingTokenizer();
    auto stack = std::make_shared<cyxwiz::SequentialModel>();
    stack->Add<ScriptedLogits>();
    auto* logits = static_cast<ScriptedLogits*>(stack->GetModule(0));
    stack->SetTraining(true);
    cyxwiz::SequentialExecutableModel model(stack);
    const auto encode = [&](const std::string& text) { return tokenizer->Encode(text); };

    auto s = ChatSettings();
    s.mode = cyxwiz::GenerationPreviewMode::Continuation;
    s.stop_texts = {"two"};
    logits->script = {4, 4, 5, 4};
    size_t streamed = 0;
    auto prompt = cyxwiz::BuildLanguageModelPrompt(s, {}, true, "one", encode, 20);
    const auto stopped = cyxwiz::GenerateLanguageModelReply(model, *tokenizer, s, "one", prompt, Seed(7), {},
        [&](const std::string&, size_t tokens) { streamed = tokens; });
    Require(stopped.text == "one one " || stopped.text == "one one", "stop text cut from the reply: " + stopped.text);
    Require(stopped.stop_text == "two" && streamed == 3 && stopped.new_token_ids.size() == 3, "stopped at the stop text");
    Require(cyxwiz::LanguageModelReplyStopLabel(stopped) == "Stopped at \"two\"", "stop label");
    Require(stack->GetModule(0)->IsTraining(), "the model's own mode comes back");

    s.stop_texts.clear();
    logits->script = {5, 3};
    logits->produced = 0;
    const auto eos = cyxwiz::GenerateLanguageModelReply(model, *tokenizer, s, "one", prompt, Seed(7), {}, {});
    Require(eos.text == "two" && cyxwiz::LanguageModelReplyStopLabel(eos) == "Stopped by itself", "end of sequence");

    logits->script = {4};
    logits->produced = 0;
    const auto limit = cyxwiz::GenerateLanguageModelReply(model, *tokenizer, s, "one", prompt, Seed(7), {}, {});
    Require(cyxwiz::LanguageModelReplyStopLabel(limit) == "Hit the 4-token limit" &&
                cyxwiz::LanguageModelReplySpeedLabel(limit).rfind("4 tokens", 0) == 0,
            "token limit and speed labels");
    Require(cyxwiz::LanguageModelHistory({eos, limit}).size() == 2 &&
                cyxwiz::LanguageModelHistory({eos})[0].assistant == "two",
            "history carries each message and reply");

    // The conversation as a report the Generations view reads.
    const auto dir = std::filesystem::temp_directory_path() /
                     ("lm_chat_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto path = cyxwiz::GenerationPreviewReportPath(dir, "chat", 0, 0);
    Require(path.filename() == "chat.json", "chat report name");
    cyxwiz::LanguageModelReportSource source;
    source.run_id = "train-1";
    source.epoch = 2;
    source.weights = "best";
    source.context = 20;
    source.tokenizer_sha256 = tokenizer->Sha256();
    std::string error;
    Require(cyxwiz::WriteLanguageModelChatReport(path, {stopped, eos}, s, source, error), error);
    const auto report = cyxwiz::LoadGenerationPreviewReport(path, error);
    Require(report && report->point == "chat" && report->samples.size() == 2 && report->samples[1].text == "two" &&
                report->samples[1].stop_reason == "eos",
            "the chat report loads as a generation report: " + error);
    Require(cyxwiz::GenerationPreviewPointLabel(*report) == "Chat: epoch 2 best weights", "chat point label");
    Require(cyxwiz::ExportLanguageModelChatJsonl(dir / "chat.jsonl", {stopped, eos}, s, error), error);
    std::ifstream jsonl(dir / "chat.jsonl");
    int lines = 0;
    for (std::string line; std::getline(jsonl, line);) {
        Require(nlohmann::json::parse(line).contains("output"), "jsonl line shape");
        ++lines;
    }
    Require(lines == 2, "one JSONL line per reply");
    jsonl.close();
    std::filesystem::remove_all(dir);
}

void TestCheckpointHelpers() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() /
                     ("lm_ckpt_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto write = [](const fs::path& path, const std::string& text) {
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << text;
    };
    write(dir / "train-1" / "best" / "metadata.json",
          R"({"epoch":4,"val_loss":3.42,"timestamp":"2026-10-14T13:26:55Z"})");
    write(dir / "train-1" / "best" / "model" / "w.bin", "x");
    write(dir / "train-2" / "latest" / "metadata.json", R"({"epoch":1,"val_loss":0,"timestamp":"2026-10-15T09:00:00Z"})");
    write(dir / "notes" / "readme.txt", "not a run");
    fs::last_write_time(dir / "train-1" / "best" / "metadata.json",
                        fs::last_write_time(dir / "train-2" / "latest" / "metadata.json") - std::chrono::hours(1));
    const auto rows = cyxwiz::ListCheckpointRuns(dir);
    Require(rows.size() == 2 && rows[0].run == "train-2" && rows[0].weights == "latest" && !rows[0].validation_loss,
            "runs newest first; a zero validation loss is not shown");
    Require(rows[1].run == "train-1" && rows[1].epoch == 4 && rows[1].validation_loss &&
                *rows[1].validation_loss > 3.41 && rows[1].saved == "2026-10-14 13:26" &&
                rows[1].folder == dir / "train-1" / "best",
            "run row facts");
    Require(cyxwiz::GraphCheckpointFolder((dir / "missing.cyxgraph").string(), "", "fallback") == "fallback",
            "an unreadable graph falls back to the project's checkpoints");
    Require(cyxwiz::LanguageModelSummary(4, 3.4203, 768, 8192, "Byte BPE") ==
                "Epoch 4 \xC2\xB7 validation loss 3.42 \xC2\xB7 context 768 \xC2\xB7 vocabulary 8,192 (Byte BPE)",
            "summary line");
    Require(cyxwiz::LanguageModelSummary(0, std::nullopt, 0, 50, "Word") == "Vocabulary 50 (Word)",
            "summary without training facts");

    // Saving the panel's settings keeps the graph's own prompts and switch.
    write(dir / "g.cyxgraph", R"({"nodes":[{"id":3,"parameters":{"generation_preview_enabled":"true",)"
                              R"("generation_preview_prompt_set":"[{\"text\":\"hi\"}]","learning_rate":"0.001"}}],"links":[]})");
    auto settings = ChatSettings();
    settings.enabled = false;
    settings.stop_texts = {"</answer>"};
    settings.max_new_tokens = 128;
    std::string error;
    Require(cyxwiz::SaveGenerationSettingsToGraph((dir / "g.cyxgraph").string(), 3, settings, error), error);
    nlohmann::json saved;
    std::ifstream(dir / "g.cyxgraph") >> saved;
    std::map<std::string, std::string> params;
    for (auto& [key, value] : saved["nodes"][0]["parameters"].items()) params[key] = value.get<std::string>();
    const auto back = cyxwiz::ParseTrainingGenerationPreview(params);
    Require(back.enabled && back.prompts.size() == 1 && back.prompts[0].text == "hi", "prompts and the switch kept");
    Require(back.mode == cyxwiz::GenerationPreviewMode::Chat && back.label_assistant == "Bot:" &&
                back.max_new_tokens == 128 && back.stop_texts == std::vector<std::string>({"</answer>"}) &&
                params["learning_rate"] == "0.001",
            "the panel's settings written, other parameters untouched");
    // With previews off, the graph's own prompts survive a save and its
    // settings read back (they are not reset to the defaults).
    write(dir / "off.cyxgraph", R"({"nodes":[{"id":3,"parameters":{"generation_preview_enabled":"false",)"
                                R"("generation_preview_prompt_set":"[{\"text\":\"kept\"}]"}}],"links":[]})");
    Require(cyxwiz::SaveGenerationSettingsToGraph((dir / "off.cyxgraph").string(), 3, settings, error), error);
    nlohmann::json off;
    std::ifstream(dir / "off.cyxgraph") >> off;
    std::map<std::string, std::string> off_params;
    for (auto& [key, value] : off["nodes"][0]["parameters"].items()) off_params[key] = value.get<std::string>();
    const auto off_back = cyxwiz::ParseGenerationSettings(off_params);
    Require(!off_back.enabled && off_back.prompts.size() == 1 && off_back.prompts[0].text == "kept" &&
                off_back.max_new_tokens == 128 && off_back.label_assistant == "Bot:",
            "previews off: prompts kept, settings read back");
    Require(!cyxwiz::ParseTrainingGenerationPreview(off_params).enabled, "training still sees previews off");

    Require(!cyxwiz::SaveGenerationSettingsToGraph((dir / "g.cyxgraph").string(), 9, settings, error) &&
                error.find("Data Loader") != std::string::npos,
            "an unknown Data Loader is refused");

    // A graph an older Engine saved (prompts in the old keys): its settings are
    // replaced in the current form, previews off, other parameters untouched.
    write(dir / "old.cyxgraph", R"({"nodes":[{"id":3,"parameters":{"generation_preview_enabled":"true",)"
                                R"("generation_preview_prompts":"In the beginning","generation_preview_every_epochs":"1",)"
                                R"("learning_rate":"0.001"}}],"links":[]})");
    bool replaced = false;
    Require(cyxwiz::SaveGenerationSettingsToGraph((dir / "old.cyxgraph").string(), 3, settings, error, &replaced) &&
                replaced,
            "an older Engine's settings are replaced: " + error);
    nlohmann::json rewritten;
    std::ifstream(dir / "old.cyxgraph") >> rewritten;
    std::map<std::string, std::string> fresh;
    for (auto& [key, value] : rewritten["nodes"][0]["parameters"].items()) fresh[key] = value.get<std::string>();
    Require(!fresh.count("generation_preview_prompts") && !fresh.count("generation_preview_every_epochs") &&
                fresh["learning_rate"] == "0.001",
            "the old keys are gone, the rest kept");
    const auto now = cyxwiz::ParseGenerationSettings(fresh);
    Require(!now.enabled && now.prompts.empty() && now.max_new_tokens == 128 &&
                now.mode == cyxwiz::GenerationPreviewMode::Chat,
            "the graph reads in the current form: the panel's settings, previews off");
    fs::remove_all(dir);
}

}  // namespace

int main() {
    TestConversationRendering();
    TestTokenizer();
    TestPromptTrimming();
    TestReplies();
    TestCheckpointHelpers();
    std::cout << "Language model chat test passed\n";
    return 0;
}
