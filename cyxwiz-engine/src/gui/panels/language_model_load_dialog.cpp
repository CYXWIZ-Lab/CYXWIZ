#include "language_model_load_dialog.h"

#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/async_task_manager.h"
#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/training_manager.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <cstdio>

namespace cyxwiz {
namespace {

constexpr const char* kTitle = "Load a model";

// One row of the source list: a title, a muted description and an optional
// status line, on the selection tone when chosen. Returns true when clicked.
bool SourceRow(const char* id, const char* title, const char* description, const char* status, bool chosen) {
    const auto& t = ui::CurrentTokens();
    ImGui::PushID(id);
    const float width = ImGui::GetContentRegionAvail().x;
    const float pad = 12.0f;
    const float wrap = width - pad * 2.0f;
    const float height = pad * 2.0f + ImGui::GetTextLineHeight() +
                         ImGui::CalcTextSize(description, nullptr, false, wrap).y + 4.0f +
                         (status && *status ? ImGui::GetTextLineHeightWithSpacing() : 0.0f);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##row", ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (chosen || hovered)
        draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height),
                            ImGui::GetColorU32(chosen ? t.selection : t.hover), 8.0f);
    ImVec2 at(origin.x + pad, origin.y + pad);
    draw->AddText(at, ImGui::GetColorU32(t.text_bright), title);
    at.y += ImGui::GetTextLineHeight() + 4.0f;
    draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), at, ImGui::GetColorU32(t.text_dim), description, nullptr, wrap);
    if (status && *status) {
        at.y += ImGui::CalcTextSize(description, nullptr, false, wrap).y + 2.0f;
        draw->AddText(at, ImGui::GetColorU32(t.text_faint), status);
    }
    ImGui::PopID();
    return clicked;
}

std::string SessionStatus() {
    auto& training = TrainingManager::Instance();
    if (training.IsTrainingActive()) return "Training is running";
    if (!training.GetActiveModel()) return "None in this session";
    const auto info = training.GetActiveModelInfo();
    if (info.origin == ActiveModelOrigin::LoadedCheckpoint) {
        const std::filesystem::path path = info.checkpoint_path;
        return "Loaded checkpoint " + path.parent_path().filename().string() + " \xC2\xB7 " + path.filename().string();
    }
    return "Trained in this session";
}

}  // namespace

void LanguageModelLoadDialog::Open(LanguageModelLoadRequest::Source source) {
    source_ = source;
    open_request_ = true;
    error_.clear();
    // Read once: a device query logs.
    device_label_ = "the default device";
    if (const auto selected = Device::GetProcessDevice()) {
        const auto info = Device(selected->type, selected->device_id).GetInfo();
        device_label_ = info.name.empty() ? "device " + std::to_string(selected->device_id) : info.name;
    }
    RefreshGraphs();
}

void LanguageModelLoadDialog::RefreshGraphs() {
    namespace fs = std::filesystem;
    const fs::path previous = graph_index_ >= 0 && graph_index_ < static_cast<int>(graphs_.size())
                                  ? graphs_[static_cast<size_t>(graph_index_)]
                                  : fs::path();
    graphs_.clear();
    graph_index_ = -1;
    const auto& root = ProjectManager::Instance().GetProjectRoot();
    std::error_code ec;
    if (root.empty() || !fs::is_directory(root, ec)) return;
    std::vector<std::pair<fs::file_time_type, fs::path>> found;
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        std::error_code inner;
        const auto name = it->path().filename().string();
        if (it->is_directory(inner)) {
            // Folders that hold data or results, not graphs.
            if (name == "checkpoints" || name == "datasets" || name == "build" || name == ".git" ||
                name == "node_modules" || name == "__pycache__")
                it.disable_recursion_pending();
            continue;
        }
        if (it->path().extension() == ".cyxgraph") found.push_back({it->last_write_time(inner), it->path()});
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (auto& [time, path] : found) graphs_.push_back(std::move(path));
    for (size_t i = 0; i < graphs_.size(); ++i)
        if (graphs_[i] == previous) graph_index_ = static_cast<int>(i);
    if (graph_index_ < 0) {
        // The newest graph with a saved run here (a graph trained elsewhere
        // has none), else the newest graph.
        auto& project = ProjectManager::Instance();
        for (size_t i = 0; i < graphs_.size() && graph_index_ < 0; ++i) {
            const auto folder = GraphCheckpointFolder(graphs_[i].string(), project.GetProjectRoot(),
                                                      project.GetCheckpointsPath());
            if (!ListCheckpointRuns(folder).empty()) graph_index_ = static_cast<int>(i);
        }
        if (graph_index_ < 0 && !graphs_.empty()) graph_index_ = 0;
    }
    runs_folder_chosen_ = false;
    RefreshRuns();
}

void LanguageModelLoadDialog::RefreshRuns() {
    auto& project = ProjectManager::Instance();
    if (!runs_folder_chosen_) {
        runs_folder_ = graph_index_ >= 0
                           ? GraphCheckpointFolder(graphs_[static_cast<size_t>(graph_index_)].string(),
                                                   project.GetProjectRoot(), project.GetCheckpointsPath())
                           : std::filesystem::path(project.GetCheckpointsPath());
    }
    runs_ = ListCheckpointRuns(runs_folder_);
    run_index_ = runs_.empty() ? -1 : 0;
}

std::string LanguageModelLoadDialog::Blocker() const {
    if (load_) return "Loading";
    switch (source_) {
        case LanguageModelLoadRequest::Source::Session:
            if (TrainingManager::Instance().IsTrainingActive()) return "Training is running; use the model when it ends";
            if (!TrainingManager::Instance().GetActiveModel()) return "This session has no trained or loaded model";
            return {};
        case LanguageModelLoadRequest::Source::Checkpoint:
            if (graph_index_ < 0) return "Choose the graph the checkpoint was trained from";
            if (run_index_ < 0) return "Choose a run (none under this folder)";
            return {};
        case LanguageModelLoadRequest::Source::Package:
            if (package_path_.empty()) return "Choose a .cyxmodel package";
            return {};
    }
    return {};
}

std::shared_ptr<LoadedLanguageModel> LanguageModelLoadDialog::Render() {
    if (open_request_) {
        ImGui::OpenPopup(kTitle);
        open_request_ = false;
    }
    std::shared_ptr<LoadedLanguageModel> loaded;
    if (load_ && load_->done.load()) {
        if (load_->ok && load_->result) {
            loaded = load_->result;
        } else {
            std::lock_guard<std::mutex> lock(load_->mutex);
            error_ = load_->error.empty() ? "Loading failed" : load_->error;
        }
        load_.reset();
    }

    ui::DialogOptions options;
    options.size = ImVec2(1040.0f, 720.0f);
    options.min_size = ImVec2(760.0f, 520.0f);
    if (!ui::BeginDialog(kTitle, options)) return loaded;
    if (loaded) ImGui::CloseCurrentPopup();
    const auto& t = ui::CurrentTokens();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim,
                       "Use a trained language model here, without training. To serve it to other apps over HTTP, "
                       "use Deploy.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    if (ImGui::BeginTable("##load_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("sources", ImGuiTableColumnFlags_WidthFixed, 290.0f);
        ImGui::TableSetupColumn("body", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        RenderSources();
        ImGui::TableSetColumnIndex(1);
        ImGui::BeginChild("##load_body", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        switch (source_) {
            case LanguageModelLoadRequest::Source::Session: RenderSession(); break;
            case LanguageModelLoadRequest::Source::Checkpoint: RenderCheckpoint(); break;
            case LanguageModelLoadRequest::Source::Package: RenderPackage(); break;
        }
        ImGui::Spacing();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Device");
        ImGui::SameLine();
        ImGui::TextColored(t.text, "%s", device_label_.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "(the selected device; change it in Preferences > Devices)");
        RenderProgress();
        ImGui::EndChild();
        ImGui::EndTable();
    }

    const std::string blocker = Blocker();
    ImGui::TextColored(t.text_faint, "The session's active model (Test, Inference) is not replaced.");
    const auto result = ui::EndDialog("Load", load_ ? "Cancel loading" : "Cancel", blocker.empty(),
                                      blocker.c_str(), false, false);
    if (result == ui::DialogResult::Primary) {
        Start();
    } else if (result == ui::DialogResult::Secondary && load_) {
        AsyncTaskManager::Instance().Cancel(load_->task_id);
    }
    return loaded;
}

void LanguageModelLoadDialog::RenderSources() {
    const std::string session = SessionStatus();
    using Source = LanguageModelLoadRequest::Source;
    ImGui::BeginDisabled(load_ != nullptr);
    if (SourceRow("session", "This session's model", "The model trained now, or opened with Load checkpoint.",
                  session.c_str(), source_ == Source::Session))
        source_ = Source::Session;
    ImGui::Spacing();
    if (SourceRow("checkpoint", "A training checkpoint",
                  "A run of a graph in this project. The graph gives the model and its tokenizer.", nullptr,
                  source_ == Source::Checkpoint))
        source_ = Source::Checkpoint;
    ImGui::Spacing();
    if (SourceRow("package", "A model package (.cyxmodel)",
                  "An exported model with its tokenizer, from this machine or another.", nullptr,
                  source_ == Source::Package))
        source_ = Source::Package;
    ImGui::EndDisabled();
}

void LanguageModelLoadDialog::RenderSession() {
    const auto& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_bright, "This session's model");
    auto& training = TrainingManager::Instance();
    if (!training.GetActiveModel()) {
        ui::EmptyState("", "No model in this session", "Train a graph, or use Train > Test > Load Checkpoint.");
        return;
    }
    const auto info = training.GetActiveModelInfo();
    std::vector<ui::KeyValue> facts = {{"Model", SessionStatus()},
                                       {"Data", info.effective_dataset_name},
                                       {"Checkpoint", info.checkpoint_path}};
    if (info.checkpoint_metadata.epoch > 0) facts.push_back({"Epoch", std::to_string(info.checkpoint_metadata.epoch)});
    ui::FactsTable("##session_facts", facts);
}

void LanguageModelLoadDialog::RenderCheckpoint() {
    const auto& t = ui::CurrentTokens();
    const auto& root = ProjectManager::Instance().GetProjectRoot();
    ui::FieldLabel("Graph");
    if (graphs_.empty()) {
        ImGui::TextColored(t.warning, root.empty() ? "Open a project to choose one of its graphs"
                                                   : "This project has no .cyxgraph files");
    } else {
        const auto shown = [&](int index) {
            std::error_code ec;
            const auto relative = std::filesystem::relative(graphs_[static_cast<size_t>(index)], root, ec);
            return ec ? graphs_[static_cast<size_t>(index)].filename().string() : relative.generic_string();
        };
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::BeginCombo("##graph", shown(graph_index_).c_str())) {
            for (int i = 0; i < static_cast<int>(graphs_.size()); ++i) {
                if (ImGui::Selectable(shown(i).c_str(), i == graph_index_) && i != graph_index_) {
                    graph_index_ = i;
                    runs_folder_chosen_ = false;
                    RefreshRuns();
                }
            }
            ImGui::EndCombo();
        }
    }
    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    std::error_code ec;
    const auto folder_shown = std::filesystem::relative(runs_folder_, root, ec);
    ImGui::TextColored(t.text_dim, "Runs in %s", (ec || root.empty() ? runs_folder_ : folder_shown).generic_string().c_str());
    ui::SameLineRight(ImGui::CalcTextSize("Choose a folder...").x + 8.0f);
    if (ui::LinkButton("Choose a folder...")) {
        const std::string start = runs_folder_.string();
        if (auto chosen = FileDialogs::SelectFolder("Checkpoint folder", start.empty() ? nullptr : start.c_str())) {
            std::filesystem::path picked = *chosen;
            // A run or a checkpoint itself: list its parent's runs.
            if (std::filesystem::exists(picked / "metadata.json", ec)) picked = picked.parent_path();
            if (std::filesystem::exists(picked / "best" / "metadata.json", ec) ||
                std::filesystem::exists(picked / "latest" / "metadata.json", ec))
                picked = picked.parent_path();
            runs_folder_ = picked;
            runs_folder_chosen_ = true;
            RefreshRuns();
        }
    }
    if (runs_.empty()) {
        ImGui::TextColored(t.text_faint, "No runs with a saved checkpoint here. Train the graph, or choose a folder.");
        return;
    }
    const ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY;
    const float height = std::min(8, static_cast<int>(runs_.size()) + 1) * ImGui::GetFrameHeightWithSpacing() + 8.0f;
    if (ImGui::BeginTable("##runs", 5, flags, ImVec2(0.0f, height))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthStretch, 2.2f);
        ImGui::TableSetupColumn("Weights", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Epoch", ImGuiTableColumnFlags_WidthStretch, 0.7f);
        ImGui::TableSetupColumn("Val. loss", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Saved", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableHeadersRow();
        for (int i = 0; i < static_cast<int>(runs_.size()); ++i) {
            const auto& row = runs_[static_cast<size_t>(i)];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushID(i);
            if (ImGui::Selectable(row.run.c_str(), i == run_index_, ImGuiSelectableFlags_SpanAllColumns))
                run_index_ = i;
            ImGui::PopID();
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(row.weights.c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%d", row.epoch);
            ImGui::TableSetColumnIndex(3);
            if (row.validation_loss) ImGui::Text("%.2f", *row.validation_loss);
            else ImGui::TextColored(t.text_faint, "-");
            ImGui::TableSetColumnIndex(4);
            ImGui::TextUnformatted(row.saved.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint,
                       "The model is built from the graph, the tokenizer is read from the graph's training data "
                       "(loaded in the background), then the weights are loaded.");
    ImGui::PopTextWrapPos();
}

void LanguageModelLoadDialog::RenderPackage() {
    const auto& t = ui::CurrentTokens();
    ui::FieldLabel("Package");
    const float browse = ui::ButtonWidth("Browse...", ui::ButtonSize::Small);
    ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x - browse - 8.0f));
    ImGui::InputTextWithHint("##package", "A .cyxmodel file or folder", &package_path_);
    ImGui::SameLine();
    if (ui::SecondaryButton("Browse...")) {
        const FileDialogs::FilterList filters = {{"CyxWiz model", "cyxmodel"}};
        if (auto chosen = FileDialogs::OpenFile("Open a model package", filters)) package_path_ = *chosen;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint,
                       "A package carries its tokenizer. Packages exported before chat settings were stored start "
                       "in Continuation; set the template in the panel.");
    ImGui::PopTextWrapPos();
}

void LanguageModelLoadDialog::RenderProgress() {
    const auto& t = ui::CurrentTokens();
    if (load_) {
        std::string step;
        int index = 0, count = 0;
        {
            std::lock_guard<std::mutex> lock(load_->mutex);
            step = load_->step;
            index = load_->index;
            count = load_->count;
        }
        ImGui::Spacing();
        ImGui::TextColored(t.running, "%s...", step.empty() ? "Loading" : step.c_str());
        if (count > 0) {
            char note[32];
            std::snprintf(note, sizeof(note), "step %d of %d", index, count);
            ui::SameLineRight(ImGui::CalcTextSize(note).x);
            ImGui::TextColored(t.text_dim, "%s", note);
        }
        ImGui::ProgressBar(count > 0 ? (index - 0.5f) / static_cast<float>(count) : -1.0f * static_cast<float>(ImGui::GetTime()),
                           ImVec2(-1.0f, 6.0f), "");
        ImGui::TextColored(t.text_faint, "The Engine stays usable while it loads.");
    } else if (!error_.empty()) {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", error_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void LanguageModelLoadDialog::Start() {
    LanguageModelLoadRequest request;
    request.source = source_;
    request.project_root = ProjectManager::Instance().GetProjectRoot();
    if (source_ == LanguageModelLoadRequest::Source::Checkpoint) {
        request.graph_path = graphs_[static_cast<size_t>(graph_index_)].string();
        request.checkpoint_path = runs_[static_cast<size_t>(run_index_)].folder.string();
    }
    request.package_path = package_path_;
    error_.clear();
    auto state = std::make_shared<LoadState>();
    load_ = state;
    std::weak_ptr<const void> owner = state;
    state->task_id = AsyncTaskManager::Instance().RunAsync(
        "Load a language model",
        [state, request](LambdaTask& task) {
            const auto cancelled = [&task] { return task.IsCancelRequested(); };
            auto loaded = LoadLanguageModel(request, cancelled, [&state](const std::string& step, int index, int count) {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->step = step;
                state->index = index;
                state->count = count;
            });
            if (task.IsCancelRequested()) throw std::runtime_error("Loading was cancelled");
            state->result = std::make_shared<LoadedLanguageModel>(std::move(loaded));
        },
        nullptr,
        [state](bool success, const std::string& error) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->ok = success && state->result;
                state->error = error;
            }
            state->done.store(true);
        },
        owner);
}

}  // namespace cyxwiz
