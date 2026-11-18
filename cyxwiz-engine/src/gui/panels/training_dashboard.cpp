#include "training_dashboard.h"

#include "../../core/engine_config.h"
#include "../../core/plot/plot_prepare.h"
#include "../../core/rl_playback_presentation.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace cyxwiz {

namespace {

std::string Format(double v, const char* fmt = "%.4g") {
    char buf[48];
    std::snprintf(buf, sizeof(buf), fmt, v);
    return buf;
}

// One KPI card: label, value, a muted detail (it wraps: a narrow card hides nothing).
void Card(const char* id, const char* label, const std::string& value, const std::string& detail, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
    ImGui::BeginChild(id, ImVec2(width, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::TextColored(t.text_dim, "%s", label);
    {
        ui::FontScope big(ui::Font::Heading);
        ImGui::TextUnformatted(value.c_str());
    }
    // Room for three lines of detail: a note that grows by a line as its
    // number gets longer ("best 810.3") does not move what is below the cards
    const float detail_top = ImGui::GetCursorPosY();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", detail.c_str());
    ImGui::PopTextWrapPos();
    const float three_lines = 3.0f * ImGui::GetTextLineHeightWithSpacing();
    const float used = ImGui::GetCursorPosY() - detail_top;
    const float missing = three_lines - used - ImGui::GetStyle().ItemSpacing.y;
    if (missing > 0.0f) ImGui::Dummy(ImVec2(0.0f, missing));
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// KPI cards in rows: as many a row as fit at a readable width, at most `most`,
// so a narrow dashboard wraps the cards rather than cutting their labels
class CardRows {
public:
    explicit CardRows(size_t most) {
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const float avail = ImGui::GetContentRegionAvail().x;
        const float readable = 10.0f * ImGui::GetFontSize();
        per_row_ = std::clamp(static_cast<size_t>((avail + gap) / (readable + gap)), size_t{1}, most);
        width_ = (avail - static_cast<float>(per_row_ - 1) * gap) / static_cast<float>(per_row_);
    }
    void Add(const char* id, const char* label, const std::string& value, const std::string& detail) {
        if (placed_ % per_row_ != 0) ImGui::SameLine();
        Card(id, label, value, detail, width_);
        ++placed_;
    }
    // The next card starts a row
    void NewRow() { placed_ = 0; }

private:
    size_t per_row_ = 1;
    float width_ = 0.0f;
    size_t placed_ = 0;
};

constexpr const char* kReferenceColumn = "SB3 reference";

// The preview card's label and value rows
void FactRow(const char* label, const std::string& value) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TableNextColumn();
    ImGui::TextColored(t.text_dim, "%s", label);
    ImGui::TableNextColumn();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(value.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

TrainingDashboardPanel::TrainingDashboardPanel() : Panel("RL Training Dashboard", true) {
    // Registered up front: values that arrive before the first frame are
    // kept, not dropped as unknown metrics (TOFIX134 P0 item 7). The order
    // gives each its theme series colour.
    RegisterCustomPlot("episode_reward", "Episode reward");
    RegisterCustomPlot("episode_length", "Episode length");
    RegisterCustomPlot("policy_loss", "Policy loss");
    RegisterCustomPlot("value_loss", "Value loss");
    RegisterCustomPlot("explained_variance", "Explained variance");
    RegisterCustomPlot("entropy", "Entropy");
    RegisterCustomPlot("approx_kl", "Approx KL");
    RegisterCustomPlot("clip_fraction", "Clip fraction");
    RegisterCustomPlot("action_std", "Action std");
    // A SAC run's (TOFIX216 S4)
    RegisterCustomPlot("critic_loss", "Critic loss");
    RegisterCustomPlot("actor_loss", "Actor loss");
    RegisterCustomPlot("ent_coef", "Entropy weight");
    RegisterCustomPlot("mean_q", "Mean Q");
}

void TrainingDashboardPanel::RegisterCustomPlot(const std::string& name, const std::string& display_name) {
    if (metrics_.count(name)) return;
    Metric m;
    m.display_name = display_name;
    m.view = std::make_unique<plot::PlotView>("rl_" + name);
    metrics_[name] = std::move(m);
    order_.push_back(name);
}

void TrainingDashboardPanel::Push(const std::string& name, double value, double steps) {
    auto it = metrics_.find(name);
    if (it == metrics_.end()) return;
    auto& m = it->second;
    // a script's value after an Engine run's starts the series by report again
    if (std::isnan(steps) && !m.steps.empty()) {
        m.history.clear();
        m.steps.clear();
    }
    // and an Engine run's value after a script's starts a series by steps: the two
    // lists stay the same length
    if (!std::isnan(steps) && m.steps.size() != m.history.size()) {
        m.history.clear();
        m.steps.clear();
    }
    m.history.push_back(value);
    if (!std::isnan(steps)) m.steps.push_back(steps);
    if (m.history.size() > kMaxHistory + kMaxHistory / 10) {
        const size_t drop = m.history.size() - kMaxHistory;
        m.history.erase(m.history.begin(), m.history.begin() + static_cast<long long>(drop));
        const size_t step_drop = (std::min)(drop, m.steps.size());
        m.steps.erase(m.steps.begin(), m.steps.begin() + static_cast<long long>(step_drop));
    }
    m.dirty = true;
}

void TrainingDashboardPanel::UpdateCustomMetric(const std::string& name, float value) {
    Push(name, static_cast<double>(value), std::nan(""));
}

void TrainingDashboardPanel::SetTrainingState(bool is_training) { training_ = is_training; }

void TrainingDashboardPanel::SetRLTrainingState(bool is_rl_training) { training_ = is_rl_training; }

void TrainingDashboardPanel::ResetRLMetrics() {
    for (auto& [name, m] : metrics_) {
        m.history.clear();
        m.steps.clear();
        m.dirty = true;
    }
    training_ = false;
    run_.reset();
    controls_ = {};
    device_chip_.clear();
    device_tooltip_.clear();
    device_fallback_ = false;
    device_note_.clear();
    device_compute_path_.clear();
    memory_place_.clear();
    last_update_.reset();
    last_sac_.reset();
    sac_report_updates_ = 0;
    outcome_.clear();
    failed_ = false;
}

void TrainingDashboardPanel::BeginRun(RunInfo info, RunControls controls) {
    ResetRLMetrics();
    run_ = std::move(info);
    controls_ = std::move(controls);
    training_ = true;
    visible_ = true;
    const auto& config = core::EngineConfig::Instance();
    watch_ = config.WatchTrainingMode();
    watch_every_ = config.WatchTrainingEvery();
}

void TrainingDashboardPanel::SetRunDevice(std::string chip, std::string tooltip, bool fallback, std::string note) {
    device_chip_ = std::move(chip);
    device_tooltip_ = std::move(tooltip);
    device_fallback_ = fallback;
    device_note_ = std::move(note);
}

void TrainingDashboardPanel::AddPpoUpdate(const rl::PpoUpdateMetrics& m) {
    const double steps = static_cast<double>(m.timesteps);
    if (!std::isnan(m.mean_episode_return)) {
        Push("episode_reward", m.mean_episode_return, steps);
        Push("episode_length", m.mean_episode_length, steps);
    }
    Push("policy_loss", m.policy_loss, steps);
    Push("value_loss", m.value_loss, steps);
    if (!std::isnan(m.explained_variance)) Push("explained_variance", m.explained_variance, steps);
    Push("entropy", -m.entropy_loss, steps);
    Push("approx_kl", m.approx_kl, steps);
    Push("clip_fraction", m.clip_fraction, steps);
    if (run_ && run_->continuous) Push("action_std", m.std, steps);
    last_update_ = m;
}

void TrainingDashboardPanel::AddSacReport(const rl::SacUpdateMetrics& m) {
    const double steps = static_cast<double>(m.timesteps);
    if (!std::isnan(m.mean_episode_return)) {
        Push("episode_reward", m.mean_episode_return, steps);
        Push("episode_length", m.mean_episode_length, steps);
    }
    // The losses average the gradient steps since the last report (none before learning starts)
    sac_report_updates_ = m.updates - (last_sac_ ? last_sac_->updates : 0);
    if (sac_report_updates_ > 0) {
        Push("critic_loss", m.critic_loss, steps);
        Push("actor_loss", m.actor_loss, steps);
        Push("mean_q", m.mean_q, steps);
    }
    Push("ent_coef", m.ent_coef, steps);
    last_sac_ = m;
}

long long TrainingDashboardPanel::RunSteps() const {
    if (run_ && run_->sac) return last_sac_ ? static_cast<long long>(last_sac_->timesteps) : 0;
    return last_update_ ? static_cast<long long>(last_update_->timesteps) : 0;
}

void TrainingDashboardPanel::EndRun(const std::string& outcome, bool failed) {
    training_ = false;
    outcome_ = outcome;
    failed_ = failed;
    // The run's own controls go with it; Play and Save of its policy stay
    controls_.pause = nullptr;
    controls_.resume = nullptr;
    controls_.stop = nullptr;
    controls_.paused = nullptr;
}

void TrainingDashboardPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(760, 820), ImGuiCond_FirstUseEver);
    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& rewards = metrics_["episode_reward"].history;
    const bool running = training_.load();

    if (run_) {
        RenderRunHeader();
    } else {
        // Status, episodes, reset.
        ui::StatusPill("##rl_status", running ? "Training" : "Stopped", running ? t.success : t.pending);
        ImGui::SameLine();
        if (rewards.empty()) ImGui::TextColored(t.text_dim, "No episodes reported yet");
        else ImGui::Text("%s episodes reported", plot::Thousands(static_cast<long long>(rewards.size())).c_str());
        ImGui::SameLine();
        const char* reset = "Reset metrics";
        const float w = ui::ButtonWidth(reset, ui::ButtonSize::Small);
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - w));
        if (ui::SecondaryButton(reset)) ResetRLMetrics();
    }

    ImGui::Spacing();
    RenderKpis();
    ImGui::Spacing();
    if (run_ && training_.load()) RenderWatch();

    if (ImGui::BeginTabBar("##rl_tabs")) {
        if (ImGui::BeginTabItem("Episodes")) {
            const float h = std::max(160.0f, (ImGui::GetContentRegionAvail().y - 70.0f) / 2.0f);
            RenderMetric("episode_reward", h);
            RenderMetric("episode_length", h);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Policy diagnostics")) {
            // Every diagnostic with values, in registration order; the learner's
            // main loss even before its first value
            const char* main_loss = run_ && run_->sac ? "critic_loss" : "policy_loss";
            std::vector<std::string> shown;
            for (const auto& name : order_)
                if (name != "episode_reward" && name != "episode_length" &&
                    (!metrics_[name].history.empty() || name == main_loss))
                    shown.push_back(name);
            ImGui::BeginChild("##rl_diag", ImVec2(0, 0));
            for (const auto& name : shown) RenderMetric(name, 150.0f);
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    if (rewards.empty() && !running && !run_)
        ImGui::TextColored(t.text_dim, "Train an RL graph (Environment, Policy Head, then Value Head and PPO, or "
                                       "Q Head and SAC) on the canvas, or report metrics from a script with "
                                       "pycyxwiz.rl_update_metric.");
    ImGui::End();

    // Charts opened in their own windows.
    for (const auto& name : order_) {
        auto& m = metrics_[name];
        plot::PlotView::Options o;
        o.export_name = name;
        m.view->DrawOwnWindow(o);
    }
}

void TrainingDashboardPanel::RenderRunHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool running = training_.load();
    const bool paused = running && controls_.paused && controls_.paused();
    const bool waiting = running && controls_.waiting && controls_.waiting();
    const char* state = running ? (waiting ? "Waiting" : paused ? "Paused" : "Training")
                                : (failed_ ? "Failed" : outcome_.c_str());
    const ImVec4 colour =
        running ? (waiting ? t.caution : paused ? t.pending : t.success) : (failed_ ? t.error : t.pending);
    ui::StatusPill("##rl_status", state, colour);
    if (failed_ && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", outcome_.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%s", run_->title.c_str());
    if (!device_chip_.empty()) {  // where it trains; a fallback in the caution tone
        ImGui::SameLine();
        ui::StatusPill("##rl_device", device_chip_.c_str(), device_fallback_ ? t.caution : t.text_dim);
        if (ImGui::IsItemHovered()) {
            if (device_compute_path_.empty()) ImGui::SetTooltip("%s", device_tooltip_.c_str());
            else ImGui::SetTooltip("%s\nRuns as %s", device_tooltip_.c_str(), device_compute_path_.c_str());
        }
    }

    // Pause / Resume and Stop while it runs; after it, Play policy and Save
    // policy... when it left one; Reset at the right
    const char* pause = paused ? "Resume" : "Pause";
    const char* stop = "Stop";
    const char* play = ICON_FA_PLAY " Play policy";
    const char* save = "Save policy...";
    const char* reset = "Reset metrics";
    const bool has_policy = !running && controls_.has_policy && controls_.has_policy();
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float w = (running ? ui::ButtonWidth(pause, ui::ButtonSize::Small) + ui::ButtonWidth(stop, ui::ButtonSize::Small) +
                                   2 * gap
                             : 0.0f) +
                    (has_policy ? ui::ButtonWidth(play, ui::ButtonSize::Small) +
                                      ui::ButtonWidth(save, ui::ButtonSize::Small) + 2 * gap
                                : 0.0f) +
                    ui::ButtonWidth(reset, ui::ButtonSize::Small);
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < w) ImGui::NewLine();  // narrow: the buttons on a line of their own
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - w));
    if (has_policy) {
        if (ui::PrimaryButton(play, true, nullptr, ui::ButtonSize::Small) && controls_.play) controls_.play();
        ImGui::SameLine();
        if (ui::SecondaryButton(save) && controls_.save) controls_.save();
        ImGui::SameLine();
    }
    if (running) {
        if (ui::SecondaryButton(pause)) {
            if (paused && controls_.resume) controls_.resume();
            else if (!paused && controls_.pause) controls_.pause();
        }
        ImGui::SameLine();
        if (ui::DangerButton(stop) && controls_.stop) controls_.stop();
        ImGui::SameLine();
    }
    if (ui::SecondaryButton(reset, !running, "Stop the run first")) ResetRLMetrics();
    if (!run_) return;

    // Progress: steps of the total, the update (SAC: its updates and random steps), the target
    const long long steps = RunSteps();
    std::string progress;
    if (run_->sac) {
        progress = rl_playback::SacProgress(steps, run_->total_timesteps, last_sac_ ? last_sac_->updates : 0,
                                            run_->learning_starts);
    } else {
        const int update = last_update_ ? last_update_->update : 0;
        progress = plot::Thousands(steps) + " of " + plot::Thousands(run_->total_timesteps) + " steps";
        if (update > 0) progress += " \xC2\xB7 update " + std::to_string(update);
    }
    ImGui::TextUnformatted(progress.c_str());
    if (!std::isnan(run_->target_return)) {
        const std::string target = "stops at mean return " + Format(run_->target_return, "%g") + " (last 100 episodes)";
        ImGui::SameLine();
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize(target.c_str()).x));
        ImGui::TextColored(t.text_dim, "%s", target.c_str());
    }
    const float fraction = run_->total_timesteps > 0
                               ? std::clamp(static_cast<float>(steps) / static_cast<float>(run_->total_timesteps), 0.0f, 1.0f)
                               : 0.0f;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, t.accent);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_input);
    ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 6.0f), "");
    ImGui::PopStyleColor(2);
    if (!device_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(device_fallback_ ? t.caution : t.text_dim, "%s", device_note_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void TrainingDashboardPanel::RenderWatch() {
    if (!controls_.watch || !controls_.watch_view) return;
    const ui::Tokens& t = ui::CurrentTokens();
    namespace pb = rl_playback;
    const auto view = controls_.watch_view();
    ImGui::TextColored(t.text_dim, "Watch progress");
    if (!view.unavailable.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", view.unavailable.c_str());
        ImGui::PopTextWrapPos();
        controls_.watch(0, watch_every_);
        ImGui::Spacing();
        return;
    }
    ui::SegmentedControl("##watch_progress", pb::kWatchChoices, pb::kWatchChoiceCount, &watch_);
    if (watch_ == 1) {
        // A SAC run copies its actor at a report (every report_every steps)
        const bool sac = run_ && run_->sac;
        const std::string every_tip =
            sac ? "Reports between two copies of the actor (a report every " +
                      plot::Thousands(static_cast<long long>(run_->report_every)) + " steps)"
                : std::string("Updates between two copies of the networks");
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Take the networks every");
        ImGui::SameLine();
        ui::NumberField("##watch_every", &watch_every_, 1, 1000, 1, 60.0f, every_tip.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", sac ? "reports" : "updates");
        ImGui::SameLine();
        if (ui::SecondaryButton("Now") && controls_.watch_now) controls_.watch_now();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", sac ? "Take the networks at the next report" : "Take the networks at the next update");
        // The preview card: which networks it plays, its episodes, how it plays
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
        ImGui::BeginChild("##watch_card", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        if (ImGui::BeginTable("##watch_facts", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("##l", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("##v", ImGuiTableColumnFlags_WidthStretch);
            FactRow("Preview of", view.preview_of);
            FactRow("Its episodes", view.episodes);
            FactRow("Plays", "best actions, real time, the run's own ground and settings");
            ImGui::EndTable();
        }
        if (!view.message.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.error, "%s", view.message.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }
    controls_.watch(watch_, watch_every_);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s Preferences > Simulation keeps the default.", pb::kWatchHint);
    ImGui::PopTextWrapPos();
    if (watch_ != 0 && controls_.show_view && ui::SecondaryButton(view.view_label.c_str())) controls_.show_view();
    ImGui::Spacing();
}

void TrainingDashboardPanel::RenderKpis() {
    const auto& reward = metrics_["episode_reward"].history;
    const auto& length = metrics_["episode_length"].history;
    const auto& policy = metrics_["policy_loss"].history;
    if (run_ && run_->sac) {
        RenderSacKpis();
        return;
    }
    const auto mean = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    CardRows cards(run_ ? 4 : 3);
    if (run_) {
        // a PPO run reports the mean of the last 100 episodes each update
        cards.Add("##kpi_reward", "Episode reward", reward.empty() ? "-" : Format(reward.back(), "%.1f"),
                  reward.empty() ? "no episode finished yet"
                                 : "mean of last 100 \xC2\xB7 best " +
                                       Format(*std::max_element(reward.begin(), reward.end()), "%.1f"));
        cards.Add("##kpi_length", "Episode length", length.empty() ? "-" : Format(length.back(), "%.1f"),
                  "mean of last 100");
    } else {
        cards.Add("##kpi_reward", "Episode reward", reward.empty() ? "-" : Format(reward.back(), "%.2f"),
                  reward.empty() ? "" : "best " + Format(*std::max_element(reward.begin(), reward.end()), "%.2f") +
                                            " \xC2\xB7 mean " + Format(mean(reward), "%.2f"));
        cards.Add("##kpi_length", "Episode length", length.empty() ? "-" : Format(length.back(), "%.0f"),
                  length.empty() ? "" : "mean " + Format(mean(length), "%.0f"));
    }
    cards.Add("##kpi_policy", "Policy loss", policy.empty() ? "-" : Format(policy.back(), "%.4f"),
              policy.empty() ? "" : plot::Thousands(static_cast<long long>(policy.size())) + " updates");
    if (run_) {
        cards.Add("##kpi_speed", "Speed", last_update_ ? Format(last_update_->steps_per_second, "%.0f") : "-",
                  "steps per second");
        // The environment's own numbers (a curriculum's difficulty, falls) on a row of their own
        if (last_update_ && !last_update_->environment_stats.empty()) {
            cards.NewRow();
            const auto& stats = last_update_->environment_stats;
            for (size_t i = 0; i < stats.size(); ++i) {
                const std::string id = "##kpi_env_" + std::to_string(i);
                cards.Add(id.c_str(), stats[i].label.c_str(), stats[i].value, stats[i].note);
            }
        }
    }
}

// A SAC run's cards (approved board 2): reward, critic and actor loss,
// entropy weight, replay buffer; then length, speed and the environment's own
void TrainingDashboardPanel::RenderSacKpis() {
    namespace pb = rl_playback;
    CardRows cards(5);
    const auto& reward = metrics_["episode_reward"].history;
    const auto& length = metrics_["episode_length"].history;
    const auto& critic = metrics_["critic_loss"].history;
    const auto& actor = metrics_["actor_loss"].history;
    const auto& mean_q = metrics_["mean_q"].history;
    const auto& alpha = metrics_["ent_coef"].history;
    cards.Add("##kpi_reward", "Episode reward", reward.empty() ? "-" : Format(reward.back(), "%.1f"),
              reward.empty() ? "no episode finished yet"
                             : "mean of last 100 \xC2\xB7 best " +
                                   Format(*std::max_element(reward.begin(), reward.end()), "%.1f"));
    cards.Add("##kpi_critic", "Critic loss", critic.empty() ? "-" : Format(critic.back(), "%.4g"),
              pb::SacCriticDetail(critic.empty() ? 0 : sac_report_updates_, run_->learning_starts));
    cards.Add("##kpi_actor", "Actor loss", actor.empty() ? "-" : Format(actor.back(), "%.4g"),
              mean_q.empty() ? "" : "mean Q " + Format(mean_q.back(), "%.4g"));
    cards.Add("##kpi_alpha", "Entropy weight", alpha.empty() ? "-" : Format(alpha.back(), "%.3g"),
              pb::SacEntropyDetail(run_->ent_coef_auto, run_->ent_coef_start, run_->target_entropy));
    const auto buffer = last_sac_ ? pb::SacReplayBufferCard(last_sac_->buffer_size, last_sac_->buffer_capacity,
                                                            last_sac_->buffer_bytes, memory_place_)
                                  : pb::CardText{"-", "fills from the first step"};
    cards.Add("##kpi_buffer", "Replay buffer", buffer.value, buffer.detail);
    // Length, speed and the environment's own numbers (falls, a curriculum's difficulty) start a row
    cards.NewRow();
    cards.Add("##kpi_length", "Episode length", length.empty() ? "-" : Format(length.back(), "%.1f"),
              "mean of last 100");
    cards.Add("##kpi_speed", "Speed", last_sac_ ? Format(last_sac_->steps_per_second, "%.0f") : "-",
              pb::SacSpeedDetail(run_->updates_per_step));
    if (last_sac_) {
        const auto& stats = last_sac_->environment_stats;
        for (size_t i = 0; i < stats.size(); ++i) {
            const std::string id = "##kpi_env_" + std::to_string(i);
            cards.Add(id.c_str(), stats[i].label.c_str(), stats[i].value, stats[i].note);
        }
    }
}

void TrainingDashboardPanel::RenderMetric(const std::string& name, float height) {
    auto& m = metrics_[name];
    const bool by_steps = !m.steps.empty() && m.steps.size() == m.history.size();
    const bool reference = by_steps && name == "episode_reward" && run_ && run_->reference;
    if (m.dirty) {
        // The value by report number, or by environment step for an Engine run
        // (with the SB3 reference of the same settings under the reward).
        plot::PlotSpec spec;
        spec.kind = plot::Kind::Line;
        spec.x_column = "report";
        spec.y_columns = {m.display_name};
        spec.title = m.display_name;
        spec.x_label = by_steps ? "environment steps"
                                : (name == "episode_reward" || name == "episode_length" ? "episode" : "update");
        spec.legend = reference;
        plot::Source src;
        plot::SourceColumn x, y;
        x.name = "report";
        if (by_steps) {
            x.numbers = m.steps;
        } else {
            x.numbers.resize(m.history.size());
            std::iota(x.numbers.begin(), x.numbers.end(), 1.0);
        }
        y.name = m.display_name;
        y.numbers = m.history;
        src.columns = {std::move(x), std::move(y)};
        if (reference) {
            plot::SourceColumn ref;
            ref.name = std::string(kReferenceColumn) + " (" + run_->reference->label + ")";
            for (const double s : m.steps) ref.numbers.push_back(run_->reference->At(s).value_or(std::nan("")));
            spec.y_columns.push_back(ref.name);
            src.columns.push_back(std::move(ref));
        }
        plot::Prepared p = plot::Prepare(spec, src);
        if (m.history.empty()) p.problem = "No values yet.";
        m.view->SetData(std::move(p));
        m.dirty = false;
    }
    {
        ui::FontScope medium(ui::Font::Medium);
        ImGui::TextUnformatted(m.display_name.c_str());
    }
    plot::PlotView::Options o;
    o.export_name = name;
    // The series colour by registration order (reward violet, length sky...).
    o.colour_offset = static_cast<size_t>(std::find(order_.begin(), order_.end(), name) - order_.begin());
    m.view->Draw(ImVec2(0, height), o);
}

}  // namespace cyxwiz
