#pragma once

#include "../panel.h"
#include "language_model_load_dialog.h"
#include "../../core/active_model_use.h"
#include "../../core/language_model_loader.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

// Language Model Generation (TOFIX159, design approved 2026-10-18): load a
// trained language model and use it directly. Chat with a fine-tuned model
// or let a base model continue text, with the Data Loader's generation
// settings (template, system prompt, stop texts, decoding). Generic for any
// causal language model the Engine trains or imports.
class LanguageModelGenerationPanel : public Panel {
public:
    LanguageModelGenerationPanel();
    ~LanguageModelGenerationPanel() override;

    void Render() override;
    const char* GetIcon() const override { return "LM"; }

private:
    // One generation in the background; the panel polls it each frame.
    struct RunState {
        std::atomic<bool> done{false};
        bool ok = false;
        std::mutex mutex;
        std::string partial;  // Guarded by mutex.
        size_t tokens = 0;
        LanguageModelReply reply;
        std::string error;
        uint64_t task_id = 0;
        int max_new_tokens = 0;
        std::shared_ptr<ActiveModelUse> use;  // This session's model only; released when done.
    };

    void RenderEmpty();
    void RenderModelBar();
    void RenderDetails();
    void RenderRawTokenIds();
    void RenderModeSwitch(const char* note);
    void ExportReplies(const char* default_name);
    void RenderConversation(float composer_height);
    void RenderContinuation();
    void RenderReplyFooter(size_t index, bool latest);
    void RenderTokens(size_t index);
    void RenderComposer();
    void RenderSettingsRail();
    void RenderSpeed();
    void StartSpeedMeasurement();
    void PollRun();
    // Starts a reply to `message`; `replace_last` sends the history without the
    // last reply and removes it once the prompt is built (Regenerate). False,
    // with the reason in the note, when it could not start.
    bool Start(const std::string& message, bool replace_last = false);
    void SaveConversation();
    void SetModel(std::shared_ptr<LoadedLanguageModel> model);
    bool Chat() const;
    std::string SendBlocker() const;
    // A run on this session's model claims it from Train, Test and the other
    // jobs; null for another model, or with `why` set when it is busy.
    std::shared_ptr<ActiveModelUse> ClaimModel(std::string& why) const;

    LanguageModelLoadDialog load_dialog_;
    std::shared_ptr<LoadedLanguageModel> model_;
    TrainingGenerationPreviewSettings settings_;
    bool send_history_ = true;
    std::vector<LanguageModelReply> replies_;
    std::string message_;           // The chat composer, or the text to continue.
    std::string new_stop_;
    std::shared_ptr<RunState> run_;
    std::shared_ptr<RunState> raw_run_;  // Generate from IDs (Details).
    bool scroll_to_end_ = false;
    std::filesystem::path chat_report_;  // This conversation's report file.
    std::string note_;                   // The outcome of the last action (saved, exported, failed).
    bool note_is_error_ = false;
    bool details_open_ = false;
    std::set<size_t> saw_open_;          // Replies showing "What the model saw".
    int tokens_reply_ = -1;              // The reply whose tokens are shown.
    int tokens_step_ = 0;

    // The composer's context meter, recomputed when its inputs change.
    std::string meter_key_;
    LanguageModelPrompt meter_prompt_;

    // Seeds and the KV cache.
    bool random_seed_ = true;   // A new seed for each sampled reply (else the settings' seed).
    bool use_kv_cache_ = true;
    // What the model's KV cache holds after the last chat reply, so the next
    // turn feeds only what is new. Written by the generation task (one at a
    // time); cleared whenever something else runs the model.
    std::shared_ptr<std::vector<int64_t>> cache_ids_ = std::make_shared<std::vector<int64_t>>();
    struct SpeedState {
        std::atomic<bool> done{false};
        std::mutex mutex;
        std::string text;  // Guarded by mutex.
        bool failed = false;
        uint64_t task_id = 0;
        std::shared_ptr<ActiveModelUse> use;  // As RunState::use.
    };
    std::shared_ptr<SpeedState> speed_run_;
    std::string speed_text_;
    bool speed_failed_ = false;

    // Raw token IDs (Details): for checking a tokenizer or the model.
    std::string raw_ids_;
    std::string raw_result_;
};

}  // namespace cyxwiz
