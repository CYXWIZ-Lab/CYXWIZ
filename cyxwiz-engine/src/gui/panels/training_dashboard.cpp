#include "training_dashboard.h"

#include "../../core/engine_config.h"
#include "../../core/file_dialogs.h"
#include "../../core/plot/plot_prepare.h"
#include "../../core/project_manager.h"
#include "../../core/rl_playback_presentation.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace cyxwiz {

namespace {

namespace rd = rl_dashboard;

constexpr const char* kReferenceColumn = "SB3 reference";
constexpr const char* kClearTitle = "Clear the dashboard?";

ImVec4 ToneColour(rd::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
    case rd::Tone::Good: return t.success;
    case rd::Tone::Caution: return t.caution;
    case rd::Tone::Error: return t.error;
    case rd::Tone::Info: return t.info;
    case rd::Tone::Plain: return t.text;
    case rd::Tone::Faint: break;
    }
    return t.text_dim;
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// The Watch progress card's label and value rows
void FactRow(const char* label, const std::string& value) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TableNextColumn();
    ImGui::TextColored(t.text_dim, "%s", label);
    ImGui::TableNextColumn();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(value.c_str());
    ImGui::PopTextWrapPos();
}

// "target 475"
std::string TargetLabel(float target) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "target %g", static_cast<double>(target));
    return buf;
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
    // and a PPO run's value after a script's starts a series by steps: the two
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
    // A script's values: a PPO run's figures give way to them
    if (run_ && !Running()) ResetRLMetrics();
    Push(name, static_cast<double>(value), std::nan(""));
    last_script_value_ = std::chrono::steady_clock::now();
}

void TrainingDashboardPanel::ResetRLMetrics() {
    for (auto& [name, m] : metrics_) {
        m.history.clear();
        m.steps.clear();
        m.dirty = true;
    }
    run_.reset();
    controls_ = {};
    device_ = {};
    waiting_.clear();
    device_note_.clear();
    compute_path_.clear();
    updates_.clear();
    last_update_.reset();
    end_ = rd::End::None;
    end_message_.clear();
    export_note_.clear();
    last_script_value_ = {};
}

void TrainingDashboardPanel::BeginPpoRun(PpoRunInfo info, PpoRunControls controls) {
    ResetRLMetrics();
    run_ = std::move(info);
    controls_ = std::move(controls);
    visible_ = true;
    const auto& config = core::EngineConfig::Instance();
    watch_ = config.WatchTrainingMode();
    watch_every_ = config.WatchTrainingEvery();
}

void TrainingDashboardPanel::SetPpoDevice(const rl_device::RunDevice& device, std::string note) {
    device_ = device;
    device_note_ = std::move(note);
    waiting_.clear();
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
    const bool continuous = run_ && run_->continuous;
    if (continuous) Push("action_std", m.std, steps);

    rd::Update u;
    u.update = m.update;
    u.steps = steps;
    u.mean_return = m.mean_episode_return;
    u.mean_length = m.mean_episode_length;
    u.policy_loss = m.policy_loss;
    u.value_loss = m.value_loss;
    u.entropy = -m.entropy_loss;
    u.approx_kl = m.approx_kl;
    u.clip_fraction = m.clip_fraction;
    u.explained_variance = m.explained_variance;
    u.action_std = continuous ? static_cast<double>(m.std) : std::nan("");
    u.steps_per_second = m.steps_per_second;
    u.reached_target = m.reached_target;
    updates_.push_back(u);
    last_update_ = m;
}

void TrainingDashboardPanel::EndPpoRun(rd::End end, const std::string& message) {
    end_ = end;
    end_message_ = message;
    waiting_.clear();
    // The run's own controls go with it; Play and Save of its policy stay
    controls_.pause = nullptr;
    controls_.resume = nullptr;
    controls_.stop = nullptr;
    controls_.paused = nullptr;
    controls_.waiting = nullptr;
}

rd::RunFacts TrainingDashboardPanel::Facts() const {
    rd::RunFacts f;
    if (run_) {
        f.ppo = true;
        f.environment = run_->environment;
        f.copies = run_->copies;
        f.total_steps = run_->total_timesteps;
        f.target = static_cast<double>(run_->target_return);
        f.continuous = run_->continuous;
        f.running = Running();
        f.paused = f.running && controls_.paused && controls_.paused();
        if (f.running && controls_.waiting && controls_.waiting())
            f.waiting = waiting_.empty() ? std::string("Waiting for the GPU: another training run is using it") : waiting_;
        f.end = end_;
        f.end_message = end_message_;
        f.has_policy = !f.running && controls_.has_policy && controls_.has_policy();
        f.device = device_;
        f.compute_path = compute_path_;
        f.device_note = device_note_;
        if (run_->reference) {
            f.reference_label = run_->reference->label;
            if (!updates_.empty()) f.reference_now = run_->reference->At(updates_.back().steps);
        }
        f.updates = updates_;
        return f;
    }
    const auto history = [this](const char* name) {
        const auto it = metrics_.find(name);
        return it == metrics_.end() ? std::span<const double>{} : std::span<const double>(it->second.history);
    };
    f.script_rewards = history("episode_reward");
    f.script_lengths = history("episode_length");
    f.script_policy_losses = history("policy_loss");
    f.script_recent = last_script_value_ != std::chrono::steady_clock::time_point{} &&
                      std::chrono::steady_clock::now() - last_script_value_ < std::chrono::seconds(5);
    return f;
}

void TrainingDashboardPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(760, 820), ImGuiCond_FirstUseEver);
    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }
    const rd::RunFacts facts = Facts();
    const rd::Header header = rd::BuildHeader(facts);
    RenderHeader(facts, header);
    if (header.state != rd::State::NoRun) {
        RenderCards(rd::BuildCards(facts));
        if (Running()) RenderWatch();
        RenderCharts();
    }
    ImGui::End();

    // Charts opened in their own windows keep the wheel zoom.
    for (const auto& name : order_) metrics_[name].view->DrawOwnWindow(MetricOptions(name));
    RunPendingExport();
}

void TrainingDashboardPanel::RenderHeader(const rd::RunFacts& facts, const rd::Header& header) {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool running = facts.running;
    const bool ended = header.state == rd::State::ReachedTarget || header.state == rd::State::RanAllSteps ||
                       header.state == rd::State::Stopped || header.state == rd::State::Failed;
    ui::BeginCard("##rl_header");

    // Row 1: the state and the update, then the run's actions at the right;
    // each part moves to the next line when the window is narrow.
    ui::FlowRow row;
    row.Next(0.0f);
    ui::Chip(header.pill.c_str(), ToneColour(header.pill_tone));
    if (!header.beside.empty()) {
        row.Next(ImGui::CalcTextSize(header.beside.c_str()).x);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text, "%s", header.beside.c_str());
    }
    const char* pause_label = facts.paused ? ICON_FA_PLAY " Resume" : ICON_FA_PAUSE " Pause";
    const char* stop_label = ICON_FA_STOP " Stop";
    const char* play_label = ICON_FA_PLAY " Play policy";
    const char* save_label = "Save policy...";
    const char* more_label = "More " ICON_FA_CHEVRON_DOWN;
    const bool policy = ended && facts.has_policy;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    float actions_width = ui::ButtonWidth(more_label, ui::ButtonSize::Small);
    if (running)
        actions_width += ui::ButtonWidth(pause_label, ui::ButtonSize::Small) +
                         ui::ButtonWidth(stop_label, ui::ButtonSize::Small) + spacing * 2.0f;
    if (policy)
        actions_width += ui::ButtonWidth(play_label, ui::ButtonSize::Small) +
                         ui::ButtonWidth(save_label, ui::ButtonSize::Small) + spacing * 2.0f;
    ui::SameLineRight(actions_width);
    ui::FlowRow buttons;
    if (running) {
        buttons.Next(ui::ButtonWidth(pause_label, ui::ButtonSize::Small));
        if (ui::SecondaryButton(pause_label)) {
            if (facts.paused && controls_.resume) controls_.resume();
            else if (!facts.paused && controls_.pause) controls_.pause();
        }
        buttons.Next(ui::ButtonWidth(stop_label, ui::ButtonSize::Small));
        if (ui::DangerButton(stop_label) && controls_.stop) controls_.stop();
        ui::Tooltip("Stops after the update in progress; the policy so far stays to play or save.");
    }
    if (policy) {
        buttons.Next(ui::ButtonWidth(play_label, ui::ButtonSize::Small));
        if (ui::PrimaryButton(play_label, true, nullptr, ui::ButtonSize::Small) && controls_.play) controls_.play();
        buttons.Next(ui::ButtonWidth(save_label, ui::ButtonSize::Small));
        if (ui::SecondaryButton(save_label) && controls_.save) controls_.save();
    }
    buttons.Next(ui::ButtonWidth(more_label, ui::ButtonSize::Small));
    if (ui::SecondaryButton(more_label)) ImGui::OpenPopup("##rl_actions");
    if (ImGui::BeginPopup("##rl_actions")) {
        if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Export updates (CSV)...", nullptr, false, !updates_.empty()))
            pending_export_ = true;
        ui::Tooltip(updates_.empty() ? "No PPO update yet. A script's values are in each chart's Export."
                                     : "Every update's figures: steps, returns, losses, entropy, KL, clip fraction, "
                                       "speed. Asks where to save.");
        ImGui::Separator();
        const bool can_clear = !running && header.state != rd::State::NoRun;
        ImGui::PushStyleColor(ImGuiCol_Text, can_clear ? t.error : t.text_dim);
        if (ImGui::MenuItem(ICON_FA_TRASH " Clear all...", nullptr, false, can_clear)) open_clear_all_ = true;
        ImGui::PopStyleColor();
        ui::Tooltip(running ? "Stop the run first." : "Clears the figures and charts from the dashboard. Asks first.");
        ImGui::EndPopup();
    }
    if (open_clear_all_) {
        ImGui::OpenPopup(kClearTitle);
        open_clear_all_ = false;
    }
    std::vector<std::string> bullets;
    if (facts.has_policy)
        bullets.push_back("Play policy and Save policy... of this run go with it: save the policy first to keep it.");
    if (ui::ConfirmDialog(kClearTitle, "This clears the run's figures and charts from the dashboard.", "Clear all", true,
                          bullets) == ui::DialogResult::Primary) {
        ResetRLMetrics();
    }
    Wrapped(export_failed_ ? t.error : t.success, export_note_);

    // Progress, the run and where it runs
    if (header.show_progress) ui::ProgressLine(static_cast<float>(header.progress), ended ? ToneColour(header.progress_tone) : t.accent);
    Wrapped(t.text_dim, header.run_line);
    Wrapped(ToneColour(header.where_tone), header.where_line);
    Wrapped(ToneColour(header.note_tone), header.note);
    Wrapped(ToneColour(header.result_tone), header.result);
    Wrapped(t.text, header.policy_line);
    Wrapped(t.text_dim, header.hint);

    // Run details: the route, how it computes, the reference
    if (!header.details.empty()) {
        ImGui::Spacing();
        ImGui::SetNextItemOpen(run_details_open_);
        run_details_open_ = ImGui::TreeNodeEx("Run details##rl_run_details",
                                              ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanLabelWidth);
        ImGui::SameLine();
        Wrapped(t.text_faint, "device route, how it computes, the reference");
        if (run_details_open_) {
            std::vector<ui::KeyValue> rows;
            for (const auto& r : header.details) rows.push_back({r.key, r.value});
            ui::FactsTable("##rl_details", rows);
        }
    }
    ui::EndCard();
}

void TrainingDashboardPanel::RenderCards(const std::vector<rd::Card>& cards) {
    if (cards.empty()) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const auto dot = [&](int series) { return series < 0 ? t.info : ui::SeriesColour(t, static_cast<size_t>(series)); };
    struct Shown {
        std::string id, label, value, tooltip, change, line2, line3;
        ImVec4 dot, change_colour, line2_colour, line3_colour;
    };
    std::vector<Shown> shown;
    for (const auto& c : cards) {
        std::string change = c.change;
        if (c.direction != 0 && !change.empty())
            change = std::string(c.direction < 0 ? ICON_FA_CARET_DOWN : ICON_FA_CARET_UP) + " " + change;
        shown.push_back({c.id, c.label, c.value, c.tooltip, change, c.line2, c.line3, dot(c.series),
                         ToneColour(c.change_tone), t.text_dim, t.text});
    }
    // The environment's own figures (a curriculum's difficulty, falls) on a row of their own
    std::vector<Shown> environment;
    if (Running() || end_ != rd::End::None) {
        if (last_update_) {
            for (size_t i = 0; i < last_update_->environment_stats.size(); ++i) {
                const auto& s = last_update_->environment_stats[i];
                environment.push_back({"##rl_env_" + std::to_string(i), s.label, s.value, {}, {}, s.note, {}, t.info,
                                       t.text_dim, t.text_dim, t.text});
            }
        }
    }

    // As many columns as cards of at least 150 px fit
    const auto draw = [&](const char* id, const std::vector<Shown>& list) {
        if (list.empty()) return;
        const float available = ImGui::GetContentRegionAvail().x;
        const int columns = std::clamp(static_cast<int>(available / 150.0f), 1, static_cast<int>(list.size()));
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
        if (ImGui::BeginTable(id, columns, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)) {
            for (const auto& c : list) {
                ImGui::TableNextColumn();
                ui::KpiCard(c.id.c_str(), c.label.c_str(), c.dot, c.value.c_str(), c.tooltip.c_str(), c.change.c_str(),
                            c.change_colour, c.line2.c_str(), c.line2_colour, c.line3.empty() ? nullptr : c.line3.c_str(),
                            c.line3_colour);
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    };
    draw("##rl_kpis", shown);
    draw("##rl_env_kpis", environment);
}

void TrainingDashboardPanel::RenderWatch() {
    if (!controls_.watch || !controls_.watch_view) return;
    const ui::Tokens& t = ui::CurrentTokens();
    namespace pb = rl_playback;
    const auto view = controls_.watch_view();
    ui::BeginCard("##rl_watch");
    ui::CardTitle("Watch progress");
    if (!view.unavailable.empty()) {
        Wrapped(t.text_dim, view.unavailable);
        controls_.watch(0, watch_every_);
        ui::EndCard();
        return;
    }
    ui::SegmentedControl("##watch_progress", pb::kWatchChoices, pb::kWatchChoiceCount, &watch_);
    if (watch_ == 1) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Take the networks every");
        ImGui::SameLine();
        ui::NumberField("##watch_every", &watch_every_, 1, 1000, 1, 60.0f, "Updates between two copies of the networks");
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "updates");
        ImGui::SameLine();
        if (ui::SecondaryButton("Now") && controls_.watch_now) controls_.watch_now();
        ui::Tooltip("Take the networks at the next update");
        // What the preview plays: which networks, its episodes, how it plays
        if (ImGui::BeginTable("##watch_facts", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("##l", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("##v", ImGuiTableColumnFlags_WidthStretch);
            FactRow("Preview of", view.preview_of);
            FactRow("Its episodes", view.episodes);
            FactRow("Plays", "best actions, real time, the run's own ground and settings");
            ImGui::EndTable();
        }
        Wrapped(t.error, view.message);
    }
    controls_.watch(watch_, watch_every_);
    Wrapped(t.text_dim, std::string(pb::kWatchHint) + " Preferences > Simulation keeps the default.");
    if (watch_ != 0 && controls_.show_view && ui::SecondaryButton(view.view_label.c_str())) controls_.show_view();
    ui::EndCard();
}

void TrainingDashboardPanel::RenderCharts() {
    ui::BeginCard("##rl_charts");
    ui::CardTitle("Charts");
    ImGui::SameLine();
    static const char* const kTabs[] = {"Episodes", "Policy diagnostics"};
    ui::SegmentedControl("##rl_chart_tab", kTabs, 2, &chart_tab_);

    std::vector<std::string> names;
    if (chart_tab_ == 0) {
        names = {"episode_reward", "episode_length"};
    } else {
        // Every diagnostic with values, in registration order
        for (const auto& name : order_)
            if (name != "episode_reward" && name != "episode_length" &&
                (!metrics_[name].history.empty() || name == "policy_loss"))
                names.push_back(name);
    }
    const bool wide = ImGui::GetContentRegionAvail().x >= 900.0f;
    const float height = chart_tab_ == 0 ? (wide ? 300.0f : 240.0f) : (wide ? 190.0f : 170.0f);
    if (wide && ImGui::BeginTable("##rl_chart_grid", 2, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)) {
        for (const auto& name : names) {
            ImGui::TableNextColumn();
            RenderMetric(name, height);
        }
        ImGui::EndTable();
    } else {
        for (const auto& name : names) RenderMetric(name, height);
    }
    ui::EndCard();
}

void TrainingDashboardPanel::RenderMetric(const std::string& name, float height) {
    auto& m = metrics_[name];
    const bool by_steps = !m.steps.empty() && m.steps.size() == m.history.size();
    const bool reference = by_steps && name == "episode_reward" && run_ && run_->reference;
    // A PPO run reports the mean return of the last 100 episodes; a script, each episode's reward
    const std::string label = name == "episode_reward" && run_ ? std::string("Mean return") : m.display_name;
    if (m.dirty) {
        // The value by report number, or by environment step for a PPO run
        // (with the SB3 reference of the same settings under the return).
        plot::PlotSpec spec;
        spec.kind = plot::Kind::Line;
        spec.x_column = "report";
        spec.y_columns = {label};
        spec.title = label;
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
        y.name = label;
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
        // The count is the run's own values, not the reference's beside them
        if (reference && p.label.state == plot::DataLabel::State::Exact) p.label.shown = p.label.total = m.history.size();
        m.view->SetData(std::move(p));
        m.dirty = false;
    }
    {
        ui::FontScope medium(ui::Font::Medium);
        ImGui::TextUnformatted(label.c_str());
    }
    plot::PlotView::Options o = MetricOptions(name);
    // The wheel scrolls the dashboard; Ctrl + wheel zooms the chart (B4-10)
    o.zoom_with_ctrl = true;
    m.view->Draw(ImVec2(0, height), o);
}

plot::PlotView::Options TrainingDashboardPanel::MetricOptions(const std::string& name) const {
    plot::PlotView::Options o;
    o.export_name = name;
    // The series colour by registration order (return violet, length sky...).
    o.colour_offset = static_cast<size_t>(std::find(order_.begin(), order_.end(), name) - order_.begin());
    const auto it = metrics_.find(name);
    const bool by_steps = it != metrics_.end() && !it->second.steps.empty() &&
                          it->second.steps.size() == it->second.history.size();
    if (name == "episode_reward" && run_ && by_steps && std::isfinite(run_->target_return))
        o.y_lines.push_back({static_cast<double>(run_->target_return), TargetLabel(run_->target_return)});
    return o;
}

void TrainingDashboardPanel::RunPendingExport() {
    if (!pending_export_) return;
    pending_export_ = false;
    if (updates_.empty()) return;
    const std::string csv = rd::UpdatesCsv(updates_);
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    const std::string folder = root.empty() ? std::string() : (std::filesystem::path(root) / "exports").string();
    const auto path = FileDialogs::SaveFile("Export updates", {{"CSV Files", "csv"}, {"All Files", "*"}},
                                            folder.empty() ? nullptr : folder.c_str(), "rl_updates.csv");
    if (!path) return;  // cancelled
    std::ofstream file(*path, std::ios::binary | std::ios::trunc);
    const bool ok = file && (file << csv) && file.flush();
    export_failed_ = !ok;
    export_note_ = ok ? "Saved the updates to " + *path : "Could not save the updates: could not write " + *path;
}

}  // namespace cyxwiz
