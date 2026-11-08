#include "nas_panel.h"

#include "../graph_training_launcher.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/live_graph_compile.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace cyxwiz {

namespace np = nas_presentation;

namespace {

void Heading(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_bright);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void Wrapped(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

// A thin bar of `fraction` (0..1) across `width`.
void Bar(float fraction, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float height = 6.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float y = at.y + (ImGui::GetTextLineHeight() - height) * 0.5f;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + width, y + height), ui::ToU32(t.bg_input), 3.0f);
    if (fraction > 0.0f)
        draw->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + width * std::clamp(fraction, 0.0f, 1.0f), y + height),
                            ui::ToU32(t.accent), 3.0f);
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
}

// A label and a typed whole-number field on one grid row.
bool IntField(const char* label, const char* id, int* value, int min, int max, const char* unit = nullptr) {
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    const bool changed = ui::NumberField(id, value, min, max, 1, 60.0f);
    if (unit) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", unit);
    }
    return changed;
}

}  // namespace

bool ArchitectureSearchFields(NASSearchConfig& config, int* train_best) {
    bool changed = false;
    if (!ImGui::BeginTable("##nas_search_fields", 4, ImGuiTableFlags_SizingFixedFit)) return false;
    ImGui::TableSetupColumn("a", ImGuiTableColumnFlags_WidthFixed, 170.0f);
    ImGui::TableSetupColumn("b", ImGuiTableColumnFlags_WidthFixed, 160.0f);
    ImGui::TableSetupColumn("c", ImGuiTableColumnFlags_WidthFixed, 170.0f);
    ImGui::TableSetupColumn("d", ImGuiTableColumnFlags_WidthFixed, 160.0f);
    ImGui::TableNextRow();
    changed |= IntField("Versions per round", "##versions", &config.population_size, 2,
                        core::kArchitectureSearchMaxVersions);
    changed |= IntField("Rounds", "##rounds", &config.generations, 1, core::kArchitectureSearchMaxRounds);
    ImGui::TableNextRow();
    changed |= IntField("Kept unchanged", "##kept", &config.elite_count, 0, std::max(0, config.population_size - 1));
    int changed_share = static_cast<int>(std::lround(config.mutation_rate * 100.0));
    if (IntField("Changed per round", "##changed", &changed_share, 0, 100, "%")) {
        config.mutation_rate = changed_share / 100.0;
        changed = true;
    }
    ImGui::TableNextRow();
    int mixed = static_cast<int>(std::lround(config.crossover_rate * 100.0));
    if (IntField("Made from two", "##mixed", &mixed, 0, 100, "%")) {
        config.crossover_rate = mixed / 100.0;
        changed = true;
    }
    if (train_best)
        changed |= IntField("Train the best", "##train_best_default", train_best, 1, core::kArchitectureSearchMaxTrain);
    ImGui::EndTable();
    config.elite_count = std::clamp(config.elite_count, 0, std::max(0, config.population_size - 1));
    return changed;
}

NASPanel::NASPanel() = default;

NASPanel::~NASPanel() {
    stop_search_ = true;
    stop_training_ = true;
    if (search_thread_ && search_thread_->joinable()) search_thread_->join();
    if (train_thread_ && train_thread_->joinable()) train_thread_->join();
}

void NASPanel::Render() {
    PollWork();
    if (!visible_) return;
    // Nothing could be read: read again when a compile finishes (the data
    // loaded, an error fixed).
    if (read_ && !read_problem_.empty() && node_editor_ && node_editor_->GetLiveCompile() &&
        node_editor_->GetLiveCompile()->ResultSerial() != read_serial_)
        read_ = false;
    if (!read_) ReadGraph();
    if (const LiveGraphCompile* live = node_editor_ ? node_editor_->GetLiveCompile() : nullptr) {
        if (rebase_frames_ > 0) {
            --rebase_frames_;
            read_revision_ = live->GraphRevision();
        }
        canvas_changed_ = read_problem_.empty() && live->GraphRevision() != read_revision_;
    }

    ImGui::SetNextWindowSize(ImVec2(760, 1000), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_NETWORK_WIRED " Neural Architecture Search###NAS", &visible_)) {
        RenderHeader();
        ImGui::Spacing();
        ImGui::BeginChild("##nas_body", ImVec2(0, 0), ImGuiChildFlags_None);
        if (!read_problem_.empty()) {
            ui::EmptyState(ICON_FA_NETWORK_WIRED, "Nothing to score yet", read_problem_.c_str());
        } else {
            RenderScoreCard();
            ImGui::Spacing();
            ImGui::Spacing();
            RenderChange();
            ImGui::Spacing();
            ImGui::Spacing();
            RenderSearch();
            ImGui::Spacing();
            ImGui::Spacing();
            RenderSuggestions();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

// ===== The graph =====

void NASPanel::ReadGraph() {
    read_ = true;
    read_problem_.clear();
    steps_.clear();
    change_note_.clear();
    suggestions_.clear();
    if (!IsTraining() && !searching_) {
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = {};
        has_result_ = false;
        trained_.clear();
        to_train_.clear();
        versions_note_.clear();
        not_started_.clear();
    }
    if (!node_editor_) {
        read_problem_ = "No graph is open.";
        return;
    }
    const LiveGraphCompile* live = node_editor_->GetLiveCompile();
    read_serial_ = live ? live->ResultSerial() : 0;
    read_revision_ = live ? live->GraphRevision() : 0;
    const std::string& path = node_editor_->GetCurrentFilePath();
    graph_name_ = path.empty() ? "Untitled graph" : std::filesystem::path(path).stem().string();
    const TrainingConfiguration* compiled = live ? live->CleanConfig() : nullptr;
    if (!compiled) {
        read_problem_ = "Compile the graph first (it has errors or is compiling): the input and output sizes come "
                        "from the compile. Then press Read the graph again.";
        return;
    }
    shape_ = np::ReadShape(*compiled);
    if (!shape_.ok) {
        read_problem_ = shape_.why;
        return;
    }
    graph_nodes_ = node_editor_->GetNodes();
    graph_links_ = node_editor_->GetLinks();
    nodes_ = graph_nodes_;
    links_ = graph_links_;
    Rescore();
}

void NASPanel::Rescore() {
    score_ = NASEvaluator::ScoreArchitecture(nodes_, links_, shape_.input_shape, search_config_);
}

void NASPanel::Mutate(MutationType type) {
    auto [nodes, links] = NASEvaluator::MutateArchitecture(nodes_, links_, type);
    const std::string change = np::DescribeChange(nodes_, links_, nodes, links);
    if (change.empty()) {
        change_note_ = "Nothing changed: no layer fitted that change. Try another.";
        return;
    }
    nodes_ = std::move(nodes);
    links_ = std::move(links);
    Rescore();
    change_note_.clear();
    steps_.push_back({change, score_.overall_score});
}

void NASPanel::ReplaceCanvas(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links) {
    if (!node_editor_) return;
    node_editor_->ReplaceWithEditedGraph(nodes, links);
    rebase_frames_ = 3;  // the canvas as replaced is the graph read, once the compile sees it
    graph_nodes_ = nodes;
    graph_links_ = links;
    nodes_ = nodes;
    links_ = links;
    steps_.clear();
    Rescore();
}

// ===== Work on threads =====

void NASPanel::PollWork() {
    if (!searching_ && search_thread_ && search_thread_->joinable()) search_thread_->join();
    if (training_finished_) {
        if (train_thread_ && train_thread_->joinable()) train_thread_->join();
        training_finished_ = false;
        train_stage_ = TrainStage::Idle;
    }
    // The launch can stop after it started (the data preparation failed):
    // nothing is preparing or waiting any more and no version began.
    if (train_stage_ == TrainStage::Preparing && !(launch_pending_ && launch_pending_()) &&
        !::gui::HasActiveGraphTrainingPreparation()) {
        NotStarted("Not started: the data preparation stopped before the training (the Tasks panel says why).");
    }
    // The saved defaults, while nothing runs.
    if (!searching_ && !IsTraining()) {
        const auto saved = core::EngineConfig::Instance().GetArchitectureSearchDefaults();
        if (followed_defaults_ != saved) {
            np::ApplyDefaults(search_config_, train_best_, saved);
            followed_defaults_ = saved;
        }
    }
}

void NASPanel::StartSearch() {
    if (searching_ || IsTraining() || nodes_.empty()) return;
    if (search_thread_ && search_thread_->joinable()) search_thread_->join();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = {};
        has_result_ = false;
        trained_.clear();
        to_train_.clear();
    }
    versions_note_.clear();
    not_started_.clear();
    searching_ = true;
    stop_search_ = false;
    search_round_ = 0;
    search_best_ = 0.0;
    search_thread_ = std::make_unique<std::thread>(
        [this, nodes = nodes_, links = links_, input_shape = shape_.input_shape, config = search_config_]() {
            NASSearchResult result;
            try {
                result = NASEvaluator::EvolveArchitecture(nodes, links, input_shape, config,
                                                          [this](int round, const ArchitectureScore& best) {
                                                              search_round_ = round + 1;
                                                              search_best_ = best.overall_score;
                                                              return !stop_search_.load();
                                                          });
            } catch (const std::exception& e) {
                result.success = false;
                result.error_message = e.what();
            }
            if (!result.success) spdlog::warn("Architecture search: {}", result.error_message);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                result_ = std::move(result);
                has_result_ = true;
            }
            searching_ = false;
        });
}

void NASPanel::StartTraining() {
    if (IsTraining() || searching_ || !start_training_) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_result_ || !result_.success) return;
        const size_t count = std::min(result_.ranked.size(), static_cast<size_t>(train_best_));
        to_train_.assign(result_.ranked.begin(), result_.ranked.begin() + static_cast<std::ptrdiff_t>(count));
        trained_.assign(count, Trained{});
    }
    not_started_.clear();
    versions_note_.clear();
    train_stage_ = TrainStage::Preparing;
    spdlog::info("Architecture search: preparing the data as Train does to train the best {}", to_train_.size());
    start_training_(graph_nodes_, graph_links_);
}

void NASPanel::BeginTraining(TrialSetup setup) {
    if (train_stage_ != TrainStage::Preparing) return;
    if (train_thread_ && train_thread_->joinable()) train_thread_->join();
    train_stage_ = TrainStage::Training;
    stop_training_ = false;
    training_finished_ = false;
    train_thread_ = std::make_unique<std::thread>([this, setup = std::move(setup), versions = to_train_]() {
        for (size_t i = 0; i < versions.size() && !stop_training_; ++i) {
            TrialSetup version = setup;
            version.nodes = versions[i].nodes;
            version.links = versions[i].links;
            version.other_architecture = true;
            TrialCallbacks callbacks;
            callbacks.should_cancel = [this] { return stop_training_.load(); };
            callbacks.on_epoch = [this, i](const TrialProgress& p) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (i < trained_.size()) {
                    trained_[i].epoch = p.epoch;
                    trained_[i].epochs = p.epochs;
                }
            };
            TrialOutcome outcome = RunTrial(version, {}, callbacks);
            if (outcome.ok)
                spdlog::info("Architecture search: version {} trained, validation loss {:.4g}", i + 1,
                             outcome.score.val_loss);
            else
                spdlog::warn("Architecture search: version {} {}", i + 1,
                             outcome.cancelled ? "stopped" : "failed: " + outcome.error);
            std::lock_guard<std::mutex> lock(mutex_);
            if (i < trained_.size()) {
                trained_[i].done = true;
                trained_[i].ok = outcome.ok;
                trained_[i].score = outcome.score;
                trained_[i].error = outcome.cancelled ? "Stopped" : outcome.error;
            }
        }
        training_finished_ = true;
    });
}

void NASPanel::NotStarted(const std::string& why) {
    if (train_stage_ != TrainStage::Preparing) return;
    train_stage_ = TrainStage::Idle;
    not_started_ = why;
    std::lock_guard<std::mutex> lock(mutex_);
    trained_.clear();
    to_train_.clear();
    spdlog::warn("Architecture search: {}", why);
}

// ===== Drawing =====

void NASPanel::RenderHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.accent_text, ICON_FA_NETWORK_WIRED);
    ImGui::SameLine();
    Heading("Neural Architecture Search");
    if (read_problem_.empty() && shape_.ok) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s (from the graph)", graph_name_.c_str(), shape_.text.c_str());
    }
    const char* label = "Read the graph again";
    ui::SameLineRight(ImGui::CalcTextSize(label).x + 8.0f);
    const bool busy = searching_ || IsTraining();
    if (ui::LinkButton(label, !busy)) ReadGraph();
    if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Wait for the search or the training to finish");
    if (canvas_changed_)
        ui::NoticeLine("The canvas changed since the panel read it (an edit, an undo): Read the graph again to work "
                       "on it as it is now.",
                       t.caution);
}

void NASPanel::RenderScoreCard() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::BeginCard("##nas_score");
    {
        ui::FontScope big(ui::Font::Heading);
        ImGui::TextUnformatted(np::ScoreText(score_.overall_score).c_str());
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "quick score from the graph's shape, no training");
    if (ImGui::BeginTable("##score_parts", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("part", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("bar", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.000").x);
        for (const auto& part : np::ScoreParts(score_)) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", part.label.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", part.tip.c_str());
            ImGui::TableNextColumn();
            Bar(static_cast<float>(part.value), ImGui::GetContentRegionAvail().x);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(np::ScoreText(part.value).c_str());
        }
        ImGui::EndTable();
    }
    Wrapped(np::ScoreSummary(score_, nodes_, links_), t.text_dim);
    ui::EndCard();
}

void NASPanel::RenderChange() {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("Change the model");
    ImGui::Spacing();
    // The change buttons wrap to the panel's width.
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    bool first = true;
    for (const auto& mutation : np::Mutations()) {
        const float width = ui::ButtonWidth(mutation.label, ui::ButtonSize::Regular);
        if (!first) {
            ImGui::SameLine();
            if (ImGui::GetCursorScreenPos().x + width > right) ImGui::NewLine();
        }
        first = false;
        if (ui::SecondaryButton(mutation.label, true, nullptr, ui::ButtonSize::Regular)) Mutate(mutation.type);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", mutation.tip);
    }

    if (!steps_.empty() && ImGui::BeginTable("##nas_steps", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("n", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("00").x);
        ImGui::TableSetupColumn("change", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("score", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.000").x);
        for (size_t i = 0; i < steps_.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%zu", i + 1);
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(steps_[i].text.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", np::ScoreText(steps_[i].score).c_str());
        }
        ImGui::EndTable();
    }
    if (!change_note_.empty()) ui::NoticeLine(change_note_, t.text_dim);

    ImGui::Spacing();
    const bool changed = !steps_.empty();
    if (ui::PrimaryButton("Apply to the graph", changed && node_editor_, "Change the model first")) {
        ReplaceCanvas(nodes_, links_);
        change_note_ = "Applied to the canvas; Ctrl+Z takes it back.";
        spdlog::info("Architecture search: the changed model applied to the graph");
    }
    ImGui::SameLine();
    if (ui::LinkButton("Back to the graph's version", changed)) {
        nodes_ = graph_nodes_;
        links_ = graph_links_;
        steps_.clear();
        change_note_.clear();
        Rescore();
    }
    Wrapped("Apply replaces the graph on the canvas; Ctrl+Z takes it back. " +
                np::HeadNote(graph_nodes_, graph_links_, shape_),
            t.text_dim);
}

void NASPanel::RenderSearch() {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("Search automatically");
    ImGui::Spacing();
    const bool searching = searching_;
    ImGui::BeginDisabled(searching || IsTraining());
    ArchitectureSearchFields(search_config_, nullptr);
    ImGui::EndDisabled();

    ImGui::Spacing();
    if (searching) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.running, "Searching");
        ImGui::SameLine();
        ImGui::Text("round %d of %d \xC2\xB7 best quick score %s", search_round_.load(), search_config_.generations,
                    np::ScoreText(search_best_.load()).c_str());
        ImGui::SameLine();
        if (ui::DangerButton("Stop")) stop_search_ = true;
    } else {
        if (ui::SecondaryButton("Start the search", !IsTraining(), "Versions are training", ui::ButtonSize::Regular))
            StartSearch();
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Rounds use the quick score; then train the best for a real one.");
    }

    bool has_result = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        has_result = has_result_;
    }
    if (has_result && !searching) {
        ImGui::Spacing();
        RenderVersions();
    }
}

void NASPanel::RenderVersions() {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<NASCandidate> rows;
    std::vector<Trained> trained;
    NASSearchResult summary;
    size_t kept = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!result_.success) {
            ui::NoticeLine("The search found nothing to score: " + result_.error_message, t.caution);
            return;
        }
        trained = trained_;
        if (IsTraining() || !to_train_.empty()) {
            rows = to_train_;
        } else {
            const size_t count = std::min(result_.ranked.size(), static_cast<size_t>(train_best_));
            rows.assign(result_.ranked.begin(), result_.ranked.begin() + static_cast<std::ptrdiff_t>(count));
        }
        summary.total_generations = result_.total_generations;
        summary.total_evaluations = result_.total_evaluations;
        summary.stopped = result_.stopped;
        kept = result_.ranked.size();
    }

    ui::BeginCard("##nas_versions");
    ImGui::TextColored(t.text_dim, "%d rounds%s, %d versions scored, %zu different kept", summary.total_generations,
                       summary.stopped ? " (stopped)" : "", summary.total_evaluations, kept);
    ImGui::Spacing();
    if (train_stage_ == TrainStage::Idle) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Train the best");
        ImGui::SameLine();
        if (ui::NumberField("##train_best", &train_best_, 1, core::kArchitectureSearchMaxTrain, 1, 50.0f)) {
            std::lock_guard<std::mutex> lock(mutex_);
            to_train_.clear();
            trained_.clear();
        }
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("for the DataLoader's epochs, the way Train does");
        ImGui::SameLine();
        if (ui::SecondaryButton("Train them", node_editor_ != nullptr, "No graph is open", ui::ButtonSize::Regular))
            StartTraining();
    } else {
        size_t index = 0;
        while (index < trained.size() && trained[index].done) ++index;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.running, train_stage_ == TrainStage::Preparing ? "Preparing" : "Training");
        ImGui::SameLine();
        if (train_stage_ == TrainStage::Preparing) {
            ImGui::TextUnformatted("the data, as Train does; the versions train when it is ready.");
        } else if (index < trained.size()) {
            ImGui::Text("version %zu of %zu \xC2\xB7 epoch %d of %d", index + 1, trained.size(), trained[index].epoch,
                        trained[index].epochs);
        }
        ImGui::SameLine();
        if (ui::DangerButton("Stop", train_stage_ == TrainStage::Training, "Starts when the data is ready"))
            stop_training_ = true;
    }
    if (!not_started_.empty()) ui::NoticeLine(not_started_, t.caution);

    ImGui::Spacing();
    if (ImGui::BeginTable("##nas_versions_table", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("00").x);
        ImGui::TableSetupColumn("layers", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("quick", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("quick 0").x);
        ImGui::TableSetupColumn("val loss", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("epoch 00 of 00").x);
        ImGui::TableSetupColumn(shape_.task == "regression" ? "MAE" : "accuracy", ImGuiTableColumnFlags_WidthFixed,
                                ImGui::CalcTextSize("accuracy 0").x);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Apply").x + 4.0f);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < rows.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%zu", i + 1);
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(np::LayersText(rows[i].nodes, rows[i].links).c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(np::ScoreText(rows[i].score.overall_score).c_str());
            const Trained* row = i < trained.size() ? &trained[i] : nullptr;
            ImGui::TableNextColumn();
            if (row && row->done && row->ok) {
                ImGui::TextUnformatted(np::LossText(row->score).c_str());
            } else if (row && row->done) {
                ImGui::TextColored(t.caution, row->error == "Stopped" ? "Stopped" : "Failed");
                if (ImGui::IsItemHovered() && !row->error.empty()) ImGui::SetTooltip("%s", row->error.c_str());
            } else if (row && train_stage_ == TrainStage::Training && row->epochs > 0) {
                ImGui::TextColored(t.running, "epoch %d of %d", row->epoch, row->epochs);
            } else {
                ImGui::TextColored(t.text_faint, row ? "..." : "-");
            }
            ImGui::TableNextColumn();
            if (row && row->done && row->ok) ImGui::TextUnformatted(np::AccuracyText(row->score).c_str());
            else ImGui::TextColored(t.text_faint, row ? "..." : "-");
            ImGui::TableNextColumn();
            if (ui::InlineLink("Apply") && node_editor_) {
                ReplaceCanvas(rows[i].nodes, rows[i].links);
                versions_note_ = "Version " + std::to_string(i + 1) + " applied to the canvas; Ctrl+Z takes it back.";
                spdlog::info("Architecture search: version {} applied to the graph", i + 1);
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    Wrapped(versions_note_.empty() ? "Trained versions are scored on the graph's validation part; the model in use is "
                                     "not touched. Apply replaces the graph on the canvas."
                                   : versions_note_,
            t.text_dim);
    ui::EndCard();
}

void NASPanel::RenderSuggestions() {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("Start from a suggestion");
    Wrapped(np::SuggestionsIntro(shape_, 5) +
                " A suggestion is added beside the graph; link it to your data, or delete the old model, before "
                "training.",
            t.text_dim);
    ImGui::Spacing();
    if (suggestions_.empty()) {
        if (ui::SecondaryButton("Show suggestions", true, nullptr, ui::ButtonSize::Regular)) {
            for (auto& [nodes, links] :
                 NASEvaluator::SuggestArchitectures(shape_.task, shape_.input_shape, shape_.outputs, 5)) {
                ArchitectureScore score = NASEvaluator::ScoreArchitecture(nodes, links, shape_.input_shape, search_config_);
                suggestions_.push_back({std::move(nodes), std::move(links), std::move(score)});
            }
        }
        return;
    }
    if (!ImGui::BeginTable("##nas_suggestions", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) return;
    const char* add = "Add beside the graph";
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("00").x);
    ImGui::TableSetupColumn("layers", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("quick", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("quick 0").x);
    ImGui::TableSetupColumn("parameters", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("parameters").x);
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize(add).x + 4.0f);
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < suggestions_.size(); ++i) {
        const NASCandidate& suggestion = suggestions_[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%zu", i + 1);
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(np::LayersText(suggestion.nodes, suggestion.links).c_str());
        ImGui::PopTextWrapPos();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(np::ScoreText(suggestion.score.overall_score).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(np::Grouped(suggestion.score.total_params).c_str());
        ImGui::TableNextColumn();
        if (ui::InlineLink(add) && node_editor_) {
            node_editor_->InsertPattern(suggestion.nodes, suggestion.links);
            spdlog::info("Architecture search: suggestion {} added beside the graph", i + 1);
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
}

} // namespace cyxwiz
