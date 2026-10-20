#include "data_loader_generation_tab.h"

#include "generation_settings_widgets.h"

#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/arrow_dataset.h"
#include "../core/async_task_manager.h"
#include "../core/file_dialogs.h"
#include "../core/graph_compiler.h"
#include "../core/graph_compiler_dataset_hooks.h"
#include "../core/project_manager.h"
#include "../core/training_generation_preview.h"
#include "../core/training_manager.h"

#include <cyxwiz/sequential.h>

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

// The shared card (ui::BeginCard), closed at the end of the scope.
struct Card {
    explicit Card(const char* id) { cyxwiz::ui::BeginCard(id); }
    ~Card() {
        cyxwiz::ui::EndCard();
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

bool DataLoaderGenerationTab::Render(const GenerationTabDataset& dataset, NodeEditor* editor, const MLNode* loader) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    bool changed = false;
    if (ImGui::Checkbox("Generate samples while training", &settings_.enabled)) changed = true;
    ImGui::BeginDisabled(!settings_.enabled);
    const char* modes[] = {"Continuation · base model", "Chat / instruction · fine-tuned model"};
    GenerationFlowSameLine(SegmentedWidth(modes, 2), 20.0f);
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
        GenerationFlowSameLine(ImGui::CalcTextSize(note).x, 12.0f);
        ImGui::TextColored(matches ? t.text_dim : t.warning, "%s", note);
    }
    if (!old_format_note_.empty()) ImGui::TextColored(t.warning, "%s", old_format_note_.c_str());
    ImGui::Spacing();

    if (ImGui::GetContentRegionAvail().x < 900.0f) {
        changed |= RenderPrompts();
        changed |= RenderPromptEditor();
        RenderModelSees(dataset);
        RenderTryNow(dataset, editor, loader);
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
        RenderTryNow(dataset, editor, loader);
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
    GenerationHint("Saved with the graph. Each run writes generation_previews/ under its checkpoint folder (baseline, "
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
    if (prompts.empty()) GenerationHint("No prompts yet. Add one, then type what the model should continue or answer.");
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
    if (!prompts.empty()) GenerationHint("Right-click a prompt to move or remove it.");
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
        GenerationFlowSameLine(180.0f);
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::InputTextWithHint("##check_value", prompt.check.kind == GenerationPreviewCheckKind::Regex
                                                          ? "pattern, e.g. Psalm \\d+:\\d+" : "text",
                                     &prompt.check.value))
            changed = true;
    }
    ImGui::PopID();
    GenerationHint("Checks are optional. Each output shows pass or fail next to it in the Generations view.");
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
    DrawGenerationSegments(segments);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::TextColored(t.accent_text, "from the template");
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::TextColored(t.text, "your prompt");
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::TextColored(t.success, "| generation starts here");
}

bool DataLoaderGenerationTab::RenderTemplate() {
    Card card("##gen_template");
    return RenderGenerationTemplateSettings(settings_, "Chat template", "shared by every prompt");
}

bool DataLoaderGenerationTab::RenderStops() {
    Card card("##gen_stops");
    return RenderGenerationStopSettings(settings_, "Stop when the model writes", new_stop_);
}

bool DataLoaderGenerationTab::RenderDecoding() {
    Card card("##gen_decoding");
    return RenderGenerationDecodingSettings(settings_, "Decoding", true);
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
        GenerationHint("Uses the best validation checkpoint so far; until one exists, the current weights.");
    return changed;
}

namespace {

// The newest training run under `root` with a best/ checkpoint.
std::filesystem::path NewestBestCheckpoint(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path newest;
    fs::file_time_type newest_time{};
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        const auto metadata = it->path() / "best" / "metadata.json";
        std::error_code inner;
        if (!it->is_directory(inner) || !fs::exists(metadata, inner)) continue;
        const auto time = fs::last_write_time(metadata, inner);
        if (newest.empty() || time > newest_time) {
            newest = it->path() / "best";
            newest_time = time;
        }
    }
    return newest;
}

std::string Param(const MLNode* node, const char* key) {
    if (!node) return {};
    const auto it = node->parameters.find(key);
    return it == node->parameters.end() ? std::string() : it->second;
}

}  // namespace

void DataLoaderGenerationTab::RenderTryNow(const GenerationTabDataset& dataset, NodeEditor* editor,
                                          const MLNode* loader) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    Card card("##gen_try");
    Title("Try now", "the settings above on a saved checkpoint, without training");

    // Which checkpoint: the one chosen, else the newest run's best under the
    // checkpoint folder, else the Data Loader's Start from.
    std::filesystem::path checkpoint = try_checkpoint_;
    std::string source = "chosen";
    if (checkpoint.empty()) {
        const std::string folder = Param(loader, "checkpoint_dir");
        checkpoint = NewestBestCheckpoint(folder.empty() ? cyxwiz::ProjectManager::Instance().GetCheckpointsPath()
                                                         : folder);
        source = "newest run";
        if (checkpoint.empty() && !Param(loader, "initialize_from_checkpoint").empty()) {
            checkpoint = Param(loader, "initialize_from_checkpoint");
            source = "Start from";
        }
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Checkpoint");
    ImGui::SameLine();
    ImGui::PushTextWrapPos(0.0f);
    if (checkpoint.empty())
        ImGui::TextColored(t.warning, "none yet: train once, or choose a checkpoint folder");
    else
        ImGui::TextColored(t.text, "%s  (%s)", checkpoint.string().c_str(), source.c_str());
    ImGui::PopTextWrapPos();
    if (cyxwiz::ui::LinkButton("Choose...")) {
        const std::string start = checkpoint.empty() ? std::string() : checkpoint.parent_path().string();
        if (auto chosen = cyxwiz::FileDialogs::SelectFolder("Checkpoint folder (a run, or its best)",
                                                             start.empty() ? nullptr : start.c_str()))
            try_checkpoint_ = *chosen;
    }
    if (!try_checkpoint_.empty()) {
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Use the newest")) try_checkpoint_.clear();
    }

    const bool running = try_ && !try_->done.load();
    std::string blocker;
    if (cyxwiz::TrainingManager::Instance().IsTrainingActive()) blocker = "Training is running; try when it ends";
    else if (!editor || !loader) blocker = "Open the Data Loader from the canvas";
    else if (!dataset.found) blocker = "Load the training data first (apply its Data Input)";
    else if (checkpoint.empty()) blocker = "No checkpoint to try";
    else if (settings_.prompts.empty()) blocker = "Add a prompt first";
    else if (const auto problem = Problem(); !problem.empty() && settings_.enabled) blocker = problem;
    if (running) {
        ImGui::TextColored(t.running, "Generating...");
        ImGui::SameLine();
        if (cyxwiz::ui::DangerButton("Cancel")) cyxwiz::AsyncTaskManager::Instance().Cancel(try_->task_id);
    } else if (cyxwiz::ui::SecondaryButton("Try now", blocker.empty(), blocker.c_str())) {
        StartTry(editor, loader, checkpoint);
    }

    if (try_ && try_->done.load()) {
        try_failed_ = !try_->ok;
        if (try_->ok) {
            try_view_.SetFolder(try_->folder);
            try_note_ = "Saved in " + try_->folder.string();
        } else {
            try_note_ = try_->error.empty() ? "Try now failed" : try_->error;
        }
        try_.reset();
    }
    if (!try_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(try_failed_ ? t.error : t.text_faint, "%s", try_note_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!try_failed_ && !try_view_.Folder().empty()) try_view_.Render();
}

void DataLoaderGenerationTab::StartTry(NodeEditor* editor, const MLNode* loader,
                                      const std::filesystem::path& checkpoint) {
    // The graph as it is, with this tab's settings (applied or not, and on).
    auto nodes = editor->GetNodes();
    for (auto& node : nodes) {
        if (node.id != loader->id) continue;
        auto settings = settings_;
        settings.enabled = true;
        for (const char* key : {"generation_preview_prompts", "generation_preview_every_epochs"})
            node.parameters.erase(key);
        for (const auto& [key, value] : cyxwiz::SerializeTrainingGenerationPreview(settings))
            node.parameters[key] = value;
    }
    cyxwiz::GraphCompiler compiler;
    auto config = compiler.Compile(nodes, editor->GetLinks(), false);
    try_failed_ = true;
    try_view_.SetFolder({});
    if (!config.is_valid) {
        try_note_ = "The graph does not compile";
        for (const auto& issue : config.issues)
            if (issue.level == cyxwiz::IssueLevel::Error) {
                try_note_ += ": " + issue.message;
                break;
            }
        return;
    }
    if (!config.generation_preview.enabled) {
        try_note_ = "Generation previews need a causal language model trained on token windows";
        return;
    }
    try_note_.clear();
    try_failed_ = false;
    auto state = std::make_shared<TryState>();
    try_ = state;
    std::weak_ptr<const void> owner = state;
    state->task_id = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Try generation",
        [state, config, checkpoint](cyxwiz::LambdaTask& task) {
            if (cyxwiz::TrainingManager::Instance().IsTrainingActive())
                throw std::runtime_error("Training started; try when it ends");
            const auto cancelled = [&task] { return task.IsCancelRequested(); };
            auto built = cyxwiz::BuildModelFromCheckpoint(config, cyxwiz::GraphArrowDataset(config.dataset_name),
                                                          checkpoint.string(), cancelled);
            if (!built.model) throw std::runtime_error(built.error_message);
            auto* sequential = built.model->AsSequentialModel();
            if (!sequential) throw std::runtime_error("Try now needs a layer-stack language model");
            cyxwiz::TrainingGenerationPreview preview(built.prepared.generation_preview, built.prepared.output_size);
            const std::filesystem::path resolved(built.resolved_checkpoint_path);
            cyxwiz::GenerationPreviewPoint point;
            point.kind = "try";
            point.epoch = built.metadata.epoch;
            point.step = built.metadata.global_step;
            point.weights = resolved.filename().string();
            if (built.metadata.val_loss > 0.0f) point.validation_loss = built.metadata.val_loss;
            state->folder = resolved.parent_path() / "generation_tries";
            if (!preview.Run(*sequential, point, resolved.parent_path().filename().string(), state->folder, cancelled))
                throw std::runtime_error("Try now was cancelled");
        },
        nullptr,
        [state](bool success, const std::string& error) {
            state->ok = success;
            state->error = error;
            state->done.store(true);
        },
        owner);
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
