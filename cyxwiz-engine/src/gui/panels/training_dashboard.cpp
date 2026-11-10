#include "training_dashboard.h"

#include "../../core/plot/plot_prepare.h"
#include "../icons.h"
#include "../ui_buttons.h"
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

// One KPI card: label, value, a muted detail line.
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
    ImGui::TextColored(t.text_dim, "%s", detail.c_str());
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

constexpr const char* kReferenceColumn = "SB3 reference";

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
    // a script's value after a PPO run's starts the series by report again
    if (std::isnan(steps) && !m.steps.empty()) {
        m.history.clear();
        m.steps.clear();
    }
    m.history.push_back(value);
    if (!std::isnan(steps)) m.steps.push_back(steps);
    if (m.history.size() > kMaxHistory + kMaxHistory / 10) {
        const auto drop = static_cast<long long>(m.history.size() - kMaxHistory);
        m.history.erase(m.history.begin(), m.history.begin() + drop);
        if (!m.steps.empty()) m.steps.erase(m.steps.begin(), m.steps.begin() + drop);
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
    last_update_.reset();
    outcome_.clear();
    failed_ = false;
}

void TrainingDashboardPanel::BeginPpoRun(PpoRunInfo info, PpoRunControls controls) {
    ResetRLMetrics();
    run_ = std::move(info);
    controls_ = std::move(controls);
    training_ = true;
    visible_ = true;
}

void TrainingDashboardPanel::SetPpoDevice(std::string chip, std::string tooltip, bool fallback, std::string note) {
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

void TrainingDashboardPanel::EndPpoRun(const std::string& outcome, bool failed) {
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

    if (ImGui::BeginTabBar("##rl_tabs")) {
        if (ImGui::BeginTabItem("Episodes")) {
            const float h = std::max(160.0f, (ImGui::GetContentRegionAvail().y - 70.0f) / 2.0f);
            RenderMetric("episode_reward", h);
            RenderMetric("episode_length", h);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Policy diagnostics")) {
            // Every diagnostic with values, in registration order
            std::vector<std::string> shown;
            for (const auto& name : order_)
                if (name != "episode_reward" && name != "episode_length" &&
                    (!metrics_[name].history.empty() || name == "policy_loss"))
                    shown.push_back(name);
            ImGui::BeginChild("##rl_diag", ImVec2(0, 0));
            for (const auto& name : shown) RenderMetric(name, 150.0f);
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    if (rewards.empty() && !running && !run_)
        ImGui::TextColored(t.text_dim, "Train an RL graph (Environment, Policy Head, Value Head, PPO) on the canvas, "
                                       "or report metrics from a script with pycyxwiz.rl_update_metric.");
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
    const char* state = running ? (paused ? "Paused" : "Training") : (failed_ ? "Failed" : outcome_.c_str());
    const ImVec4 colour = running ? (paused ? t.pending : t.success) : (failed_ ? t.error : t.pending);
    ui::StatusPill("##rl_status", state, colour);
    if (failed_ && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", outcome_.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%s", run_->title.c_str());
    if (!device_chip_.empty()) {  // where it trains; a fallback in the caution tone
        ImGui::SameLine();
        ui::StatusPill("##rl_device", device_chip_.c_str(), device_fallback_ ? t.caution : t.text_dim);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", device_tooltip_.c_str());
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

    // Progress: steps of the total, the update, the target
    const long long steps = last_update_ ? static_cast<long long>(last_update_->timesteps) : 0;
    const int update = last_update_ ? last_update_->update : 0;
    std::string progress = plot::Thousands(steps) + " of " + plot::Thousands(run_->total_timesteps) + " steps";
    if (update > 0) progress += " \xC2\xB7 update " + std::to_string(update);
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

void TrainingDashboardPanel::RenderKpis() {
    const auto& reward = metrics_["episode_reward"].history;
    const auto& length = metrics_["episode_length"].history;
    const auto& policy = metrics_["policy_loss"].history;
    const int cards = run_ ? 4 : 3;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float w = (ImGui::GetContentRegionAvail().x - static_cast<float>(cards - 1) * gap) / static_cast<float>(cards);
    const auto mean = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    if (run_) {
        // a PPO run reports the mean of the last 100 episodes each update
        Card("##kpi_reward", "Episode reward", reward.empty() ? "-" : Format(reward.back(), "%.1f"),
             reward.empty() ? "no episode finished yet"
                            : "mean of last 100 \xC2\xB7 best " +
                                  Format(*std::max_element(reward.begin(), reward.end()), "%.1f"),
             w);
        ImGui::SameLine();
        Card("##kpi_length", "Episode length", length.empty() ? "-" : Format(length.back(), "%.1f"), "mean of last 100",
             w);
    } else {
        Card("##kpi_reward", "Episode reward", reward.empty() ? "-" : Format(reward.back(), "%.2f"),
             reward.empty() ? "" : "best " + Format(*std::max_element(reward.begin(), reward.end()), "%.2f") +
                                       " \xC2\xB7 mean " + Format(mean(reward), "%.2f"),
             w);
        ImGui::SameLine();
        Card("##kpi_length", "Episode length", length.empty() ? "-" : Format(length.back(), "%.0f"),
             length.empty() ? "" : "mean " + Format(mean(length), "%.0f"), w);
    }
    ImGui::SameLine();
    Card("##kpi_policy", "Policy loss", policy.empty() ? "-" : Format(policy.back(), "%.4f"),
         policy.empty() ? "" : plot::Thousands(static_cast<long long>(policy.size())) + " updates", w);
    if (run_) {
        ImGui::SameLine();
        Card("##kpi_speed", "Speed", last_update_ ? Format(last_update_->steps_per_second, "%.0f") : "-",
             "steps per second", w);
    }
}

void TrainingDashboardPanel::RenderMetric(const std::string& name, float height) {
    auto& m = metrics_[name];
    const bool by_steps = !m.steps.empty() && m.steps.size() == m.history.size();
    const bool reference = by_steps && name == "episode_reward" && run_ && run_->reference;
    if (m.dirty) {
        // The value by report number, or by environment step for a PPO run
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
