#include "language_model_generation_panel.h"

#include "../generation_settings_widgets.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/async_task_manager.h"
#include "../../core/file_dialogs.h"
#include "../../core/generation_preview_presentation.h"
#include "../../core/training_manager.h"
// Last: it opens cyxwiz::gui, which would hide ::gui from the headers above.
#include "../editor_fonts.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <sstream>

namespace cyxwiz {
namespace {

constexpr float kRailWidth = 360.0f;

// The role's name without its trailing colon ("Berean:" -> "Berean").
std::string RoleName(const std::string& label, const char* fallback) {
    std::string name = label;
    while (!name.empty() && (name.back() == ':' || name.back() == ' ')) name.pop_back();
    return name.empty() ? fallback : name;
}

// Wrapped text on a tinted box in the code font; `cursor` adds the
// streaming caret.
void TextBox(const char* id, const std::string& text, const ImVec4& background, const ImVec4& colour,
             bool cursor = false) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, background);
    ImGui::BeginChild(id, ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImFont* code = gui::GetCodeFont();
    if (code) ImGui::PushFont(code);
    ::gui::DrawGenerationSegments({{text + (cursor ? "_" : ""), colour}});
    if (code) ImGui::PopFont();
    ImGui::EndChild();
}

// Links laid out from the right edge of the current line, in their own ID
// scope `id` (two rows may both have "Copy"); `clicked` gets the index of
// the one clicked.
bool RightLinks(const char* id, const std::vector<const char*>& labels, int* clicked) {
    float width = 0.0f;
    for (const char* label : labels) width += ImGui::CalcTextSize(label).x + 14.0f;
    ui::SameLineRight(width);
    ImGui::PushID(id);
    bool any = false;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i) ImGui::SameLine(0.0f, 14.0f);
        if (ui::LinkButton(labels[i])) {
            *clicked = static_cast<int>(i);
            any = true;
        }
    }
    ImGui::PopID();
    return any;
}

std::vector<int64_t> ParseIds(const std::string& text) {
    std::vector<int64_t> ids;
    std::string token;
    for (const char c : text + " ") {
        if (c >= '0' && c <= '9') {
            token += c;
        } else if (!token.empty()) {
            ids.push_back(std::stoll(token));
            token.clear();
        }
    }
    return ids;
}

std::string JoinIds(const std::vector<int64_t>& ids) {
    std::ostringstream joined;
    for (size_t i = 0; i < ids.size(); ++i) joined << (i ? " " : "") << ids[i];
    return joined.str();
}

// A token as a chip label: spaces as middle dots, newlines as \n.
std::string TokenLabel(const std::string& text) {
    std::string out;
    for (const char c : text) out += c == ' ' ? std::string("\xC2\xB7") : c == '\n' ? std::string("\\n") : std::string(1, c);
    return out.empty() ? std::string("\"\"") : out;
}

}  // namespace

LanguageModelGenerationPanel::LanguageModelGenerationPanel() : Panel("Language Model Generation", false) {}

LanguageModelGenerationPanel::~LanguageModelGenerationPanel() {
    if (run_) AsyncTaskManager::Instance().Cancel(run_->task_id);
    if (raw_run_) AsyncTaskManager::Instance().Cancel(raw_run_->task_id);
    if (speed_run_) AsyncTaskManager::Instance().Cancel(speed_run_->task_id);
}

bool LanguageModelGenerationPanel::Chat() const { return settings_.mode == GenerationPreviewMode::Chat; }

void LanguageModelGenerationPanel::SetModel(std::shared_ptr<LoadedLanguageModel> model) {
    model_ = std::move(model);
    settings_ = model_->settings;
    replies_.clear();
    saw_open_.clear();
    chat_report_.clear();
    tokens_reply_ = -1;
    meter_key_.clear();
    raw_result_.clear();
    speed_text_.clear();
    cache_ids_->clear();
    note_ = "Loaded " + model_->name + " \xC2\xB7 " + std::to_string(model_->checks.size()) + " checks passed";
    if (!model_->warnings.empty()) note_ += " \xC2\xB7 " + model_->warnings.front();
    note_is_error_ = false;
}

std::string LanguageModelGenerationPanel::SendBlocker() const {
    if (!model_) return "Load a model first";
    if (run_ || raw_run_ || speed_run_) return "Generating; Stop first";
    if (model_->source == "This session's model" && TrainingManager::Instance().IsTrainingActive())
        return "Training is running on this model; generate when it ends";
    auto check = settings_;
    check.enabled = true;
    check.prompts = {{"x", {}}};
    if (const auto problem = ValidateTrainingGenerationPreview(check); !problem.empty()) return problem;
    return {};
}

void LanguageModelGenerationPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(1240.0f, 820.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Language Model Generation###LanguageModelGeneration", &visible_)) {
        PollRun();
        if (auto loaded = load_dialog_.Render()) SetModel(std::move(loaded));
        const auto& t = ui::CurrentTokens();
        if (!model_) {
            RenderEmpty();
        } else {
            RenderModelBar();
            if (details_open_) RenderDetails();
            ImGui::PushStyleColor(ImGuiCol_TableBorderLight, t.border_soft);
            ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, t.border_soft);
            if (ImGui::BeginTable("##lm_layout", 2,
                                  ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_BordersInnerV,
                                  ImGui::GetContentRegionAvail())) {
                ImGui::TableSetupColumn("main", ImGuiTableColumnFlags_WidthStretch);
                // The rail narrows with the panel, so the conversation keeps room.
                const float rail = std::clamp(ImGui::GetContentRegionAvail().x * 0.36f, 260.0f, kRailWidth);
                ImGui::TableSetupColumn("rail", ImGuiTableColumnFlags_WidthFixed, rail);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::BeginChild("##lm_main", ImVec2(0.0f, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);
                if (Chat()) RenderConversation(ImGui::GetFrameHeight() * 2.6f + ImGui::GetTextLineHeight() * 6.0f);
                else RenderContinuation();
                ImGui::EndChild();
                ImGui::TableSetColumnIndex(1);
                RenderSettingsRail();
                ImGui::EndTable();
            }
            ImGui::PopStyleColor(2);
        }
    }
    ImGui::End();
}

void LanguageModelGenerationPanel::RenderEmpty() {
    const auto& t = ui::CurrentTokens();
    using Source = LanguageModelLoadRequest::Source;
    const float width = std::min(460.0f, ImGui::GetContentRegionAvail().x);
    // The heading, the three sources and the Deploy line, centred as one block.
    const char* hint = "Load a model to chat with it or let it continue text. Any causal language model the Engine "
                       "trained works: a base model or a fine-tuned (chat) model.";
    const float line = ImGui::GetTextLineHeightWithSpacing();
    const float hint_height = ImGui::CalcTextSize(hint, nullptr, false, width).y;
    const float row_height = line * 2.0f + ImGui::GetStyle().WindowPadding.y * 2.0f + ImGui::GetStyle().ItemSpacing.y;
    const float block = line * 3.0f + hint_height + row_height * 3.0f + line * 2.0f;
    ImGui::Dummy(ImVec2(0.0f, std::max(12.0f, (ImGui::GetContentRegionAvail().y - block) * 0.4f)));
    const float left = std::max(0.0f, (ImGui::GetContentRegionAvail().x - width) * 0.5f);
    const auto centred = [&](const char* text, const ImVec4& colour) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x -
                                                                       ImGui::CalcTextSize(text).x) * 0.5f));
        ImGui::TextColored(colour, "%s", text);
    };
    ImGui::SetWindowFontScale(2.0f);
    centred(ICON_FA_CUBE, t.text_faint);
    ImGui::SetWindowFontScale(1.0f);
    centred("No language model loaded", t.text_dim);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + left);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
    ImGui::TextColored(t.text_faint, "%s", hint);
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    auto& training = TrainingManager::Instance();
    const bool has_session = training.GetActiveModel() != nullptr && !training.IsTrainingActive();
    struct Row {
        const char* id;
        const char* title;
        const char* caption;
        const char* action;
        Source source;
        bool enabled;
        const char* reason;
    };
    const Row rows[] = {
        {"session", "This session's model",
         has_session ? "Trained or loaded in this session" : "Nothing trained or loaded in this session", "Use",
         Source::Session, has_session, "Train a model, or use Train > Test > Load Checkpoint"},
        {"checkpoint", "A training checkpoint", "A run of a graph in this project", "Choose...", Source::Checkpoint,
         true, nullptr},
        {"package", "A model package (.cyxmodel)", "Carries its tokenizer", "Open...", Source::Package, true, nullptr},
    };
    for (const auto& row : rows) {
        ImGui::PushID(row.id);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + left);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
        ImGui::BeginChild("##row", ImVec2(width, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        ImGui::BeginGroup();
        ImGui::TextColored(row.enabled ? t.text_bright : t.text_dim, "%s", row.title);
        ImGui::TextColored(t.text_faint, "%s", row.caption);
        ImGui::EndGroup();
        ui::SameLineRight(ui::ButtonWidth(row.action, ui::ButtonSize::Small));
        if (ui::SecondaryButton(row.action, row.enabled, row.reason)) load_dialog_.Open(row.source);
        ImGui::EndChild();
        ImGui::PopID();
    }
    ImGui::Spacing();
    const char* deploy = "Serving a model to other apps over HTTP is under Deploy.";
    ImGui::SetCursorPosX(std::max(0.0f, (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(deploy).x) * 0.5f));
    ImGui::TextColored(t.text_faint, "%s", deploy);
}

void LanguageModelGenerationPanel::RenderModelBar() {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##model_bar", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImGui::TextColored(t.accent_text, ICON_FA_CUBE);
    ImGui::SameLine();
    ImGui::TextColored(t.text_bright, "%s", model_->name.c_str());
    ::gui::GenerationFlowSameLine(ImGui::CalcTextSize(model_->subtitle.c_str()).x);
    ImGui::TextColored(t.text_dim, "%s", model_->subtitle.c_str());

    const bool busy = run_ || raw_run_ || speed_run_;
    const std::string status = busy ? std::string("Generating") : "Ready on " + model_->device_label;
    const char* change = "Change model...";
    const char* details = details_open_ ? "Hide details" : "Details";
    const float actions = ui::ChipWidth(status.c_str()) + ui::ButtonWidth(change, ui::ButtonSize::Small) +
                          ImGui::CalcTextSize(details).x + 24.0f;
    ui::SameLineRight(actions);
    ui::Chip(status.c_str(), busy ? t.running : t.success);
    ImGui::SameLine(0.0f, 12.0f);
    if (ui::SecondaryButton(change, !busy, "Generating; Stop first")) load_dialog_.Open();
    ImGui::SameLine(0.0f, 12.0f);
    if (ui::LinkButton(details)) details_open_ = !details_open_;

    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", model_->summary.c_str());
    if (!note_.empty()) ImGui::TextColored(note_is_error_ ? t.error : t.text_faint, "%s", note_.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::Spacing();
}

void LanguageModelGenerationPanel::RenderDetails() {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##lm_details", ImVec2(0.0f, std::min(420.0f, ImGui::GetContentRegionAvail().y * 0.5f)),
                      ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    std::vector<ui::KeyValue> rows;
    for (const auto& fact : model_->details) rows.push_back({fact.key, fact.value});
    ui::KeyValueTable("##lm_facts", rows, 2, true);
    ImGui::PushTextWrapPos(0.0f);
    for (const auto& check : model_->checks) {
        ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK);
        ImGui::SameLine();
        ImGui::TextColored(t.text, "%s", check.c_str());
    }
    for (const auto& warning : model_->warnings) {
        ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::TextColored(t.text, "%s", warning.c_str());
    }
    ImGui::PopTextWrapPos();
    if (!model_->location.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", model_->location.c_str());
        ImGui::PopTextWrapPos();
        if (ui::LinkButton("Copy path")) ImGui::SetClipboardText(model_->location.c_str());
    }
    if (!replies_.empty()) {
        const auto& last = replies_.back();
        ImGui::Spacing();
        ImGui::TextColored(t.text_bright, "Last reply");
        ui::KeyValueTable("##lm_last", {{"Stopped", LanguageModelReplyStopLabel(last)},
                                        {"Prompt", std::to_string(last.prompt.ids.size()) + " tokens"},
                                        {"New tokens", LanguageModelReplySpeedLabel(last)},
                                        {"Decoding", GenerationPreviewDecodingName(settings_.decoding)},
                                        {"Seed", std::to_string(last.seed)}},
                          2);
        if (ui::LinkButton("Copy generated IDs")) ImGui::SetClipboardText(JoinIds(last.new_token_ids).c_str());
    }
    RenderRawTokenIds();
    ImGui::EndChild();
    ImGui::Spacing();
}

void LanguageModelGenerationPanel::RenderRawTokenIds() {
    const auto& t = ui::CurrentTokens();
    if (raw_run_ && raw_run_->done.load()) {
        std::lock_guard<std::mutex> lock(raw_run_->mutex);
        raw_result_ = raw_run_->ok ? "Generated IDs: " + JoinIds(raw_run_->reply.new_token_ids) + "\nDecodes to: " +
                                         raw_run_->reply.text + "\n" + LanguageModelReplyStopLabel(raw_run_->reply)
                                   : raw_run_->error;
        raw_run_.reset();
    }
    ImGui::Spacing();
    ImGui::TextColored(t.text_bright, "Raw token IDs");
    ImGui::SameLine();
    ImGui::TextColored(t.text_faint, "for checking a tokenizer or the model");
    ui::FieldLabel("Prompt token IDs (spaces, commas or new lines)");
    ImGui::InputTextMultiline("##raw_ids", &raw_ids_, ImVec2(-1.0f, ImGui::GetTextLineHeight() * 2.6f));
    std::string decoded;
    std::vector<int64_t> ids;
    try {
        ids = ParseIds(raw_ids_);
        decoded = ids.empty() ? std::string() : model_->tokenizer->Decode(ids);
    } catch (const std::exception& error) {
        decoded = error.what();
        ids.clear();
    }
    ImGui::TextColored(t.text_dim, "Decodes to");
    ImGui::SameLine();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text, "%s", decoded.c_str());
    ImGui::PopTextWrapPos();
    const size_t context = static_cast<size_t>(std::max(0, settings_.context));
    std::string blocker = ids.empty() ? "Type valid token IDs first" : SendBlocker();
    if (blocker.empty() && context > 0 && ids.size() >= context) blocker = "The IDs fill the context";
    if (ui::SecondaryButton("Generate from IDs", blocker.empty(), blocker.c_str())) {
        LanguageModelPrompt prompt;
        prompt.ids = ids;
        prompt.rendered.text = decoded;
        const size_t wanted = static_cast<size_t>(settings_.max_new_tokens);
        prompt.new_tokens_fit = context == 0 ? wanted : std::min(wanted, context - ids.size());
        auto model = model_;
        auto settings = settings_;
        auto state = std::make_shared<RunState>();
        std::weak_ptr<const void> owner = state;
        raw_run_ = state;
        raw_result_.clear();
        cache_ids_->clear();  // This run empties the model's cache.
        state->task_id = AsyncTaskManager::Instance().RunAsync(
            "Generate from token IDs",
            [model, settings, prompt, state](LambdaTask& task) {
                if (const auto error = ActivateLanguageModelDevice(*model); !error.empty()) throw std::runtime_error(error);
                LanguageModelRunOptions options;
                options.seed = settings.seed;
                auto reply = GenerateLanguageModelReply(*model->model, *model->tokenizer, settings, "", prompt, options,
                                                        [&task] { return task.IsCancelRequested(); }, {});
                std::lock_guard<std::mutex> lock(state->mutex);
                state->reply = std::move(reply);
                state->ok = true;
            },
            nullptr,
            [state](bool, const std::string& error) {
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->error = error;
                }
                state->done.store(true);
            },
            owner);
    }
    if (raw_run_) {
        ImGui::SameLine();
        ImGui::TextColored(t.running, "Generating...");
    }
    if (!raw_result_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text, "%s", raw_result_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void LanguageModelGenerationPanel::RenderModeSwitch(const char* note) {
    const auto& t = ui::CurrentTokens();
    const char* modes[] = {"Chat", "Continuation"};
    int mode = Chat() ? 0 : 1;
    ImGui::BeginDisabled(run_ != nullptr);
    if (ui::SegmentedControl("##lm_mode", modes, 2, &mode)) {
        const auto chosen = mode == 0 ? GenerationPreviewMode::Chat : GenerationPreviewMode::Continuation;
        if (chosen != settings_.mode) {
            settings_.mode = chosen;  // A new conversation in the other mode.
            replies_.clear();
            saw_open_.clear();
            chat_report_.clear();
            tokens_reply_ = -1;
        }
    }
    ImGui::EndDisabled();
    ::gui::GenerationFlowSameLine(ImGui::CalcTextSize(note).x, 12.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "%s", note);
}

void LanguageModelGenerationPanel::RenderConversation(float composer_height) {
    const auto& t = ui::CurrentTokens();
    char note[96];
    std::snprintf(note, sizeof(note), "Conversation \xC2\xB7 %zu %s", replies_.size(),
                  replies_.size() == 1 ? "turn" : "turns");
    RenderModeSwitch(note);
    int clicked = -1;
    if (RightLinks("##chat_links", {"Copy all", "Export (JSONL)", "New conversation"}, &clicked)) {
        if (clicked == 0) {
            std::string all = settings_.system_prompt.empty() ? "" : "System: " + settings_.system_prompt + "\n\n";
            for (const auto& reply : replies_)
                all += RoleName(settings_.label_user, "User") + ": " + reply.message + "\n" +
                       RoleName(settings_.label_assistant, "Assistant") + ": " + reply.text + "\n\n";
            ImGui::SetClipboardText(all.c_str());
            note_ = "Copied the conversation";
            note_is_error_ = false;
        } else if (clicked == 1) {
            ExportReplies("conversation.jsonl");
        } else if (clicked == 2 && !run_) {
            replies_.clear();
            saw_open_.clear();
            chat_report_.clear();
            tokens_reply_ = -1;
        }
    }

    if (!settings_.system_prompt.empty()) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
        ImGui::BeginChild("##system", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        ImGui::TextColored(t.accent_text, "System");
        ImGui::SameLine();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", settings_.system_prompt.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
    }

    const float height = std::max(120.0f, ImGui::GetContentRegionAvail().y - composer_height);
    ImGui::BeginChild("##turns", ImVec2(0.0f, height), ImGuiChildFlags_None);
    const std::string user = RoleName(settings_.label_user, "User");
    const std::string assistant = RoleName(settings_.label_assistant, "Assistant");
    if (replies_.empty() && !run_)
        ui::EmptyState("", "No messages yet", "Type a message below. Each reply uses the settings on the right.");
    for (size_t i = 0; i < replies_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::TextColored(t.text_dim, "%s", user.c_str());
        TextBox("##user", replies_[i].message, t.bg_raised, t.text);
        ImGui::TextColored(t.accent_text, "%s", assistant.c_str());
        TextBox("##reply", replies_[i].text, t.bg_window, t.text_bright);
        RenderReplyFooter(i, i + 1 == replies_.size());
        ImGui::PopID();
        ImGui::Spacing();
    }
    if (run_) {
        std::string partial, message;
        size_t tokens = 0;
        {
            std::lock_guard<std::mutex> lock(run_->mutex);
            partial = run_->partial;
            message = run_->reply.message;
            tokens = run_->tokens;
        }
        ImGui::TextColored(t.text_dim, "%s", user.c_str());
        TextBox("##user_running", message, t.bg_raised, t.text);
        ImGui::TextColored(t.accent_text, "%s", assistant.c_str());
        TextBox("##reply_running", partial, t.bg_window, t.text_bright, true);
        ImGui::TextColored(t.running, "Generating \xC2\xB7 %zu of up to %d tokens", tokens, run_->max_new_tokens);
        ImGui::SetScrollHereY(1.0f);
    } else if (scroll_to_end_) {
        ImGui::SetScrollHereY(1.0f);
        scroll_to_end_ = false;
    }
    ImGui::EndChild();
    RenderComposer();
}

void LanguageModelGenerationPanel::ExportReplies(const char* default_name) {
    if (replies_.empty()) {
        note_ = "Nothing to export yet";
        note_is_error_ = false;
        return;
    }
    if (auto path = FileDialogs::SaveFile("Export (JSONL)", {{"JSON Lines", "jsonl"}}, nullptr, default_name)) {
        std::string error;
        note_is_error_ = !ExportLanguageModelChatJsonl(*path, replies_, settings_, error);
        note_ = note_is_error_ ? error : "Exported " + std::to_string(replies_.size()) + " replies to " + *path;
    }
}

void LanguageModelGenerationPanel::RenderReplyFooter(size_t index, bool latest) {
    const auto& t = ui::CurrentTokens();
    const auto& reply = replies_[index];
    const auto reason = LanguageModelGenerationStopReasonName(reply.stop_reason);
    const ImVec4 colour = reason == "max_tokens" ? t.warning : reason == "user_cancel" ? t.text_dim : t.success;
    ImGui::TextColored(colour, "%s", LanguageModelReplyStopLabel(reply).c_str());
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::TextColored(t.text_dim, "%s", LanguageModelReplySpeedLabel(reply).c_str());
    if (reply.sampled) {
        const std::string seed = "seed " + std::to_string(reply.seed);
        ::gui::GenerationFlowSameLine(ImGui::CalcTextSize(seed.c_str()).x + 90.0f, 12.0f);
        ImGui::TextColored(t.text_dim, "%s", seed.c_str());
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::PushID(static_cast<int>(index));
        if (ui::LinkButton("Use this seed")) {
            settings_.seed = reply.seed;
            random_seed_ = false;
            note_ = "Fixed seed " + std::to_string(reply.seed) + ": sampling repeats this reply";
            note_is_error_ = false;
        }
        ImGui::PopID();
    }
    if (!reply.repeated_phrase.empty()) {
        const std::string repeats = "Repeats \"" + reply.repeated_phrase + "\"";
        ::gui::GenerationFlowSameLine(ui::ChipWidth(repeats.c_str()), 12.0f);
        ui::Chip(repeats.c_str(), t.warning);
    }
    const bool saw = saw_open_.count(index) > 0;
    const bool tokens = tokens_reply_ == static_cast<int>(index);
    std::vector<const char*> links = {"Copy"};
    if (latest && !run_) links.push_back(Chat() ? "Regenerate" : "Generate again");
    if (latest && !run_ && !Chat()) links.push_back("Continue further");
    links.push_back(saw ? "Hide what the model saw" : "What the model saw");
    links.push_back(tokens ? "Hide tokens" : "Tokens");
    int clicked = -1;
    if (RightLinks("##reply_links", links, &clicked)) {
        const std::string label = links[static_cast<size_t>(clicked)];
        if (label == "Copy") {
            ImGui::SetClipboardText(reply.text.c_str());
        } else if (label == "Regenerate" || label == "Generate again") {
            const std::string message = reply.message;
            replies_.pop_back();
            saw_open_.erase(index);
            if (tokens_reply_ == static_cast<int>(index)) tokens_reply_ = -1;
            Start(message);
        } else if (label == "Continue further") {
            message_ = reply.message + reply.text;
            const std::string message = message_;
            Start(message);
        } else if (label == "What the model saw" || label == "Hide what the model saw") {
            if (saw) saw_open_.erase(index);
            else saw_open_.insert(index);
        } else {
            tokens_reply_ = tokens ? -1 : static_cast<int>(index);
            tokens_step_ = 0;
        }
        return;
    }
    if (saw) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
        ImGui::BeginChild("##saw", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleColor();
        std::vector<std::pair<std::string, ImVec4>> segments;
        for (const auto& segment : reply.prompt.rendered.segments)
            segments.push_back({segment.text, segment.from_template ? t.accent_text : t.text});
        if (segments.empty()) segments.push_back({reply.prompt.rendered.text, t.text});
        ImFont* code = gui::GetCodeFont();
        if (code) ImGui::PushFont(code);
        ::gui::DrawGenerationSegments(segments);
        if (code) ImGui::PopFont();
        std::string facts = std::to_string(reply.prompt.ids.size()) + " tokens";
        if (settings_.context > 0) facts += " of the " + std::to_string(settings_.context) + "-token context";
        if (reply.prompt.turns_sent) facts += " \xC2\xB7 " + std::to_string(reply.prompt.turns_sent) + " earlier turns sent";
        if (reply.prompt.turns_left_out)
            facts += " \xC2\xB7 " + std::to_string(reply.prompt.turns_left_out) + " oldest left out to fit";
        if (reply.reused_cache_tokens)
            facts += " \xC2\xB7 " + std::to_string(reply.reused_cache_tokens) +
                     " taken from the KV cache of the previous turn";
        else if (!reply.used_kv_cache)
            facts += " \xC2\xB7 no KV cache";
        ImGui::TextColored(t.text_faint, "%s", facts.c_str());
        ImGui::EndChild();
    }
    if (tokens) RenderTokens(index);
}

void LanguageModelGenerationPanel::RenderTokens(size_t index) {
    const auto& t = ui::CurrentTokens();
    const auto& reply = replies_[index];
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    ImGui::BeginChild("##tokens", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImGui::TextColored(t.text_bright, "Tokens");
    ::gui::GenerationFlowSameLine(ImGui::CalcTextSize("click a generated token to see what else the model considered").x);
    ImGui::TextColored(t.text_faint, "click a generated token to see what else the model considered");
    ui::SameLineRight(ImGui::CalcTextSize("Copy token IDs").x);
    if (ui::LinkButton("Copy token IDs")) ImGui::SetClipboardText(JoinIds(reply.new_token_ids).c_str());
    // The last prompt tokens for context, then the generated ones.
    const size_t lead = std::min<size_t>(6, reply.prompt.ids.size());
    for (size_t i = reply.prompt.ids.size() - lead; i < reply.prompt.ids.size(); ++i) {
        const std::string label = TokenLabel(model_->tokenizer->TokenText(reply.prompt.ids[i]));
        if (i > reply.prompt.ids.size() - lead) ::gui::GenerationFlowSameLine(ui::ChipWidth(label.c_str()), 4.0f);
        ui::Chip(label.c_str(), t.text_dim);
    }
    const size_t limit = std::min<size_t>(reply.steps.size(), 96);
    for (size_t i = 0; i < limit; ++i) {
        ImGui::PushID(static_cast<int>(i));
        const std::string label = TokenLabel(model_->tokenizer->TokenText(reply.steps[i].token_id));
        ::gui::GenerationFlowSameLine(ui::FilterChipWidth(label.c_str(), nullptr), 4.0f);
        if (ui::FilterChip("##step", label.c_str(), nullptr, tokens_step_ == static_cast<int>(i)))
            tokens_step_ = static_cast<int>(i);
        ImGui::PopID();
    }
    if (reply.steps.size() > limit) {
        const std::string more = "... " + std::to_string(reply.steps.size() - limit) + " more";
        ::gui::GenerationFlowSameLine(ImGui::CalcTextSize(more.c_str()).x, 6.0f);
        ImGui::TextColored(t.text_faint, "%s", more.c_str());
    }
    if (tokens_step_ >= 0 && tokens_step_ < static_cast<int>(reply.steps.size())) {
        const auto& step = reply.steps[static_cast<size_t>(tokens_step_)];
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "Step %d \xC2\xB7 chose \"%s\"", tokens_step_ + 1,
                           TokenLabel(model_->tokenizer->TokenText(step.token_id)).c_str());
        if (ImGui::BeginTable("##candidates", 4, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Token", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableSetupColumn("Probability", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("##p", ImGuiTableColumnFlags_WidthFixed, 64.0f);
            ImGui::TableHeadersRow();
            for (const auto& candidate : step.candidates) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%lld", static_cast<long long>(candidate.token_id));
                ImGui::TableSetColumnIndex(1);
                const bool chosen = candidate.token_id == step.token_id;
                ImGui::TextColored(chosen ? t.text_bright : t.text, "%s",
                                   TokenLabel(model_->tokenizer->TokenText(candidate.token_id)).c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, chosen ? t.caution : t.text_faint);
                ImGui::ProgressBar(candidate.probability, ImVec2(-1.0f, 8.0f), "");
                ImGui::PopStyleColor();
                ImGui::TableSetColumnIndex(3);
                ImGui::TextColored(t.text_dim, "%.1f%%", candidate.probability * 100.0f);
            }
            ImGui::EndTable();
        }
        ::gui::GenerationHint("Probabilities after temperature, top-k and top-p; the five most likely.");
    }
    ImGui::EndChild();
}

void LanguageModelGenerationPanel::RenderComposer() {
    const auto& t = ui::CurrentTokens();
    // The context meter, recomputed only when what it measures changes.
    std::ostringstream key;
    key << message_ << '\x1f' << replies_.size() << '\x1f' << send_history_ << '\x1f'
        << static_cast<int>(settings_.mode) << static_cast<int>(settings_.chat_template) << '\x1f'
        << settings_.system_prompt << '\x1f' << settings_.label_system << settings_.label_user
        << settings_.label_assistant << '\x1f' << settings_.max_new_tokens;
    if (key.str() != meter_key_) {
        meter_key_ = key.str();
        try {
            meter_prompt_ = BuildLanguageModelPrompt(
                settings_, LanguageModelHistory(replies_), send_history_, message_,
                [this](const std::string& text) { return model_->tokenizer->Encode(text); },
                static_cast<size_t>(std::max(0, settings_.context)));
        } catch (const std::exception& error) {
            meter_prompt_ = {};
            meter_prompt_.error = error.what();
        }
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##composer", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    if (Chat())
        ImGui::TextColored(t.text_dim, "Message (sent as the %s turn; Ctrl+Enter for a new line)",
                           RoleName(settings_.label_user, "User").c_str());
    else
        ImGui::TextColored(t.text_dim, "Text to continue (Ctrl+Enter for a new line)");
    ImFont* code = gui::GetCodeFont();
    if (code) ImGui::PushFont(code);
    const bool entered = ImGui::InputTextMultiline(
        "##message", &message_, ImVec2(-1.0f, ImGui::GetTextLineHeight() * (Chat() ? 3.6f : 6.0f)),
        ImGuiInputTextFlags_CtrlEnterForNewLine | ImGuiInputTextFlags_EnterReturnsTrue);
    if (code) ImGui::PopFont();
    const size_t context = static_cast<size_t>(std::max(0, settings_.context));
    const bool empty = message_.find_first_not_of(" \t\r\n") == std::string::npos;
    const bool too_long = !meter_prompt_.error.empty() && !empty;
    const float used = context ? std::min(1.0f, static_cast<float>(meter_prompt_.ids.size()) / context) : 0.0f;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, too_long ? t.error : t.accent);
    ImGui::ProgressBar(used, ImVec2(std::min(220.0f, ImGui::GetContentRegionAvail().x), 6.0f), "");
    ImGui::PopStyleColor();
    std::string meter;
    if (too_long) {
        meter = meter_prompt_.error;
    } else if (context) {
        meter = std::to_string(meter_prompt_.ids.size()) + " / " + std::to_string(context) + " tokens in context \xC2\xB7 " +
                std::to_string(meter_prompt_.new_tokens_fit) + " new tokens fit";
        if (meter_prompt_.turns_left_out)
            meter += " \xC2\xB7 " + std::to_string(meter_prompt_.turns_left_out) + " oldest turns left out";
    } else {
        meter = std::to_string(meter_prompt_.ids.size()) + " tokens";
    }
    // Next to the bar when it fits, else under it; wrapped, so a narrow
    // panel shows all of it.
    ::gui::GenerationFlowSameLine(ImGui::CalcTextSize(meter.c_str()).x, 12.0f);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(too_long ? t.error : t.text_dim, "%s", meter.c_str());
    ImGui::PopTextWrapPos();
    const char* primary = Chat() ? "Send" : "Generate";
    std::string blocker = empty ? (Chat() ? "Type a message first" : "Type the text to continue first") : SendBlocker();
    if (blocker.empty() && too_long) blocker = meter_prompt_.error;
    const float buttons = ui::ButtonWidth(primary, ui::ButtonSize::Regular) +
                          (run_ ? ui::ButtonWidth("Stop", ui::ButtonSize::Regular) + 12.0f : 0.0f);
    ui::SameLineRight(buttons);
    if (run_) {
        if (ui::DangerButton("Stop", true, nullptr, ui::ButtonSize::Regular))
            AsyncTaskManager::Instance().Cancel(run_->task_id);
        ImGui::SameLine(0.0f, 12.0f);
    }
    if ((ui::PrimaryButton(primary, blocker.empty(), blocker.c_str()) || entered) && blocker.empty()) {
        const std::string message = message_;
        if (Chat()) message_.clear();
        Start(message);
    }
    ImGui::EndChild();
}

void LanguageModelGenerationPanel::RenderContinuation() {
    const auto& t = ui::CurrentTokens();
    RenderModeSwitch("The model continues your text as it is");
    int clicked = -1;
    if (RightLinks("##document_links", {"Copy", "Export (JSONL)", "Clear"}, &clicked)) {
        if (clicked == 0 && !replies_.empty()) {
            ImGui::SetClipboardText((replies_.back().message + replies_.back().text).c_str());
        } else if (clicked == 1) {
            ExportReplies("generations.jsonl");
        } else if (clicked == 2 && !run_) {
            replies_.clear();
            saw_open_.clear();
            chat_report_.clear();
            tokens_reply_ = -1;
        }
    }

    // The newest generation: your text, then what the model wrote.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_raised);
    ImGui::BeginChild("##document", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImFont* code = gui::GetCodeFont();
    if (code) ImGui::PushFont(code);
    if (run_) {
        std::lock_guard<std::mutex> lock(run_->mutex);
        ::gui::DrawGenerationSegments({{run_->reply.message, t.text}, {run_->partial + "_", t.caution}});
    } else if (!replies_.empty()) {
        ::gui::DrawGenerationSegments({{replies_.back().message, t.text}, {replies_.back().text, t.caution}});
    } else {
        ::gui::GenerationHint("The generated text appears here, after your text.");
    }
    if (code) ImGui::PopFont();
    ImGui::EndChild();
    ui::Chip("your text", t.text);
    ImGui::SameLine(0.0f, 6.0f);
    ui::Chip("generated", t.caution);
    if (run_) {
        size_t tokens = 0;
        {
            std::lock_guard<std::mutex> lock(run_->mutex);
            tokens = run_->tokens;
        }
        ImGui::SameLine(0.0f, 12.0f);
        ImGui::TextColored(t.running, "Generating \xC2\xB7 %zu of up to %d tokens", tokens, run_->max_new_tokens);
    } else if (!replies_.empty()) {
        RenderReplyFooter(replies_.size() - 1, true);
    }
    ImGui::Spacing();
    RenderComposer();
}

void LanguageModelGenerationPanel::RenderSettingsRail() {
    const auto& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##lm_rail", ImVec2(0.0f, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImGui::TextColored(t.text_bright, "Generation settings");
    ::gui::GenerationHint(model_->settings_from_graph
                            ? "The same settings as the Data Loader's Generation tab, loaded from the model's graph."
                            : "This model carries no generation settings; these are the defaults.");
    if (ui::LinkButton(model_->settings_from_graph ? "Reset to the graph's" : "Reset to the defaults")) {
        const auto mode = settings_.mode;
        settings_ = model_->settings;
        if (settings_.mode != mode) {
            replies_.clear();
            saw_open_.clear();
            chat_report_.clear();
            tokens_reply_ = -1;
        }
    }
    if (!model_->graph_path.empty() && model_->loader_node_id >= 0) {
        ImGui::SameLine(0.0f, 14.0f);
        if (ui::LinkButton("Save to the graph")) {
            std::string error;
            note_is_error_ = !SaveGenerationSettingsToGraph(model_->graph_path, model_->loader_node_id, settings_, error);
            note_ = note_is_error_ ? "Not saved: " + error
                                   : "Saved to " + std::filesystem::path(model_->graph_path).filename().string() +
                                         " (reopen it on the canvas to see the change)";
            if (!note_is_error_) model_->settings = settings_;
        }
    }
    ImGui::Spacing();
    if (Chat()) {
        ::gui::RenderGenerationTemplateSettings(settings_, "Template", "sent once, before the conversation");
        ImGui::Spacing();
    }
    ::gui::RenderGenerationStopSettings(settings_, "Stop at", new_stop_);
    ImGui::Spacing();
    ::gui::RenderGenerationDecodingSettings(settings_, "Decoding", false, &random_seed_);
    ImGui::Spacing();
    RenderSpeed();
    if (Chat()) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_bright, "History");
        ImGui::Checkbox("Send earlier turns with each message", &send_history_);
        ::gui::GenerationHint(("The oldest turns are left out when they no longer fit the " +
                             std::to_string(settings_.context) + "-token context; the system prompt always stays.")
                                .c_str());
    }
    ImGui::Spacing();
    if (!model_->chats_folder.empty()) {
        ::gui::GenerationHint(("Each conversation is kept as a generation report in " +
                             model_->chats_folder.parent_path().filename().string() + "/generation_chats/.")
                                .c_str());
    } else {
        ::gui::GenerationHint("Conversations with this model are not saved; use Export (JSONL).");
    }
    ImGui::EndChild();
}

void LanguageModelGenerationPanel::RenderSpeed() {
    const auto& t = ui::CurrentTokens();
    if (speed_run_ && speed_run_->done.load()) {
        std::lock_guard<std::mutex> lock(speed_run_->mutex);
        speed_text_ = speed_run_->text;
        speed_failed_ = speed_run_->failed;
        speed_run_.reset();
    }
    ImGui::TextColored(t.text_bright, "Speed");
    const auto* sequential = model_->model->AsSequentialModel();
    const bool cache = sequential && sequential->SupportsIncrementalDecoding();
    ImGui::BeginDisabled(!cache || run_ != nullptr);
    ImGui::Checkbox("KV cache", &use_kv_cache_);
    ImGui::EndDisabled();
    ui::Tooltip(cache ? "Reuses each layer's keys and values: every new token feeds only itself. In a chat, the "
                        "earlier turns are kept between messages too. Off recomputes the whole sequence per token."
                      : "This model has no KV cache (a graph with branches, or a layer without incremental "
                        "decoding): every token recomputes the sequence.");
    ImGui::SameLine(0.0f, 12.0f);
    const std::string blocker = message_.find_first_not_of(" \t\r\n") == std::string::npos && replies_.empty()
                                    ? "Type a message first: it is the prompt measured"
                                    : SendBlocker();
    if (ui::SecondaryButton("Measure speed", blocker.empty(), blocker.c_str())) StartSpeedMeasurement();
    if (speed_run_) {
        ImGui::TextColored(t.running, "Measuring (with and without the cache)...");
    } else if (!speed_text_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(speed_failed_ ? t.error : t.text, "%s", speed_text_.c_str());
        ImGui::PopTextWrapPos();
    }
    ::gui::GenerationHint("Runs the message (or the last one) greedily for the full max new tokens, with the KV "
                          "cache and without, and shows tokens per second.");
}

void LanguageModelGenerationPanel::StartSpeedMeasurement() {
    const bool fresh = message_.find_first_not_of(" \t\r\n") != std::string::npos;
    const std::string message = fresh ? message_ : replies_.back().message;
    LanguageModelPrompt prompt;
    try {
        prompt = BuildLanguageModelPrompt(settings_, LanguageModelHistory(replies_), send_history_, message,
                                          [this](const std::string& text) { return model_->tokenizer->Encode(text); },
                                          static_cast<size_t>(std::max(0, settings_.context)));
    } catch (const std::exception& error) {
        prompt.error = error.what();
    }
    if (!prompt.error.empty()) {
        speed_text_ = prompt.error;
        speed_failed_ = true;
        return;
    }
    auto model = model_;
    auto settings = settings_;
    auto state = std::make_shared<SpeedState>();
    std::weak_ptr<const void> owner = state;
    speed_run_ = state;
    speed_text_.clear();
    cache_ids_->clear();  // The measurement empties the model's cache.
    state->task_id = AsyncTaskManager::Instance().RunAsync(
        "Measure generation speed",
        [model, settings, prompt, state](LambdaTask& task) {
            if (const auto error = ActivateLanguageModelDevice(*model); !error.empty()) throw std::runtime_error(error);
            const auto speed = MeasureLanguageModelSpeed(*model->model, *model->tokenizer, settings, prompt,
                                                         [&task] { return task.IsCancelRequested(); });
            std::lock_guard<std::mutex> lock(state->mutex);
            state->text = LanguageModelSpeedText(speed);
        },
        nullptr,
        [state](bool success, const std::string& error) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!success) {
                    state->text = error.empty() ? "The speed measurement failed" : error;
                    state->failed = true;
                }
            }
            state->done.store(true);
        },
        owner);
}

void LanguageModelGenerationPanel::Start(const std::string& message) {
    if (!model_ || run_) return;
    LanguageModelPrompt prompt;
    try {
        prompt = BuildLanguageModelPrompt(settings_, LanguageModelHistory(replies_), send_history_, message,
                                          [this](const std::string& text) { return model_->tokenizer->Encode(text); },
                                          static_cast<size_t>(std::max(0, settings_.context)));
    } catch (const std::exception& error) {
        prompt.error = error.what();
    }
    if (!prompt.error.empty()) {
        note_ = prompt.error;
        note_is_error_ = true;
        return;
    }
    auto state = std::make_shared<RunState>();
    state->reply.message = message;
    state->max_new_tokens = static_cast<int>(prompt.new_tokens_fit);
    run_ = state;
    note_.clear();
    auto model = model_;
    auto settings = settings_;
    LanguageModelRunOptions options;
    const bool sampled = settings_.decoding == GenerationPreviewDecoding::Sampling;
    options.seed = sampled && random_seed_ ? std::random_device{}() : settings_.seed;
    options.use_kv_cache = use_kv_cache_;
    // A chat keeps the cache between turns, for a model only this panel runs
    // (the session's model is shared with Test and Inference).
    if (Chat() && use_kv_cache_ && model_->source != "This session's model") options.reuse_cache = cache_ids_.get();
    else cache_ids_->clear();
    auto cache_ids = cache_ids_;  // Keeps the vector alive for the task.
    std::weak_ptr<const void> owner = state;
    state->task_id = AsyncTaskManager::Instance().RunAsync(
        "Generate text",
        [model, settings, message, prompt, state, options, cache_ids](LambdaTask& task) {
            if (const auto error = ActivateLanguageModelDevice(*model); !error.empty()) throw std::runtime_error(error);
            auto reply = GenerateLanguageModelReply(
                *model->model, *model->tokenizer, settings, message, prompt, options,
                [&task] { return task.IsCancelRequested(); },
                [&state](const std::string& text, size_t tokens) {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->partial = text;
                    state->tokens = tokens;
                });
            std::lock_guard<std::mutex> lock(state->mutex);
            state->reply = std::move(reply);
            state->ok = true;
        },
        nullptr,
        [state](bool, const std::string& error) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->error = error;
            }
            state->done.store(true);
        },
        owner);
}

void LanguageModelGenerationPanel::PollRun() {
    if (!run_ || !run_->done.load()) return;
    bool added = false;
    {
        std::lock_guard<std::mutex> lock(run_->mutex);
        if (run_->ok) {
            replies_.push_back(std::move(run_->reply));
            added = true;
        } else {
            note_ = run_->error.empty() ? "Generation stopped" : run_->error;
            note_is_error_ = !run_->error.empty();
            // An unanswered chat message goes back to the composer.
            if (Chat() && message_.empty()) message_ = run_->reply.message;
        }
    }
    run_.reset();
    if (added) {
        scroll_to_end_ = true;
        SaveConversation();
    }
}

void LanguageModelGenerationPanel::SaveConversation() {
    if (!model_ || model_->chats_folder.empty() || replies_.empty()) return;
    if (chat_report_.empty()) chat_report_ = GenerationPreviewReportPath(model_->chats_folder, "chat", 0, 0);
    std::string error;
    if (!WriteLanguageModelChatReport(chat_report_, replies_, settings_, model_->report, error)) {
        note_ = "The conversation was not saved: " + error;
        note_is_error_ = true;
    }
}

}  // namespace cyxwiz
