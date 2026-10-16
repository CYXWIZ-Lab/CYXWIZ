#include "training_dashboard.h"

#include "../../core/plot/plot_prepare.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
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
}

void TrainingDashboardPanel::RegisterCustomPlot(const std::string& name, const std::string& display_name) {
    if (metrics_.count(name)) return;
    Metric m;
    m.display_name = display_name;
    m.view = std::make_unique<plot::PlotView>("rl_" + name);
    metrics_[name] = std::move(m);
    order_.push_back(name);
}

void TrainingDashboardPanel::UpdateCustomMetric(const std::string& name, float value) {
    auto it = metrics_.find(name);
    if (it == metrics_.end()) return;
    auto& h = it->second.history;
    h.push_back(static_cast<double>(value));
    if (h.size() > kMaxHistory + kMaxHistory / 10) h.erase(h.begin(), h.begin() + static_cast<long long>(h.size() - kMaxHistory));
    it->second.dirty = true;
}

void TrainingDashboardPanel::SetTrainingState(bool is_training) { training_ = is_training; }

void TrainingDashboardPanel::SetRLTrainingState(bool is_rl_training) { training_ = is_rl_training; }

void TrainingDashboardPanel::ResetRLMetrics() {
    for (auto& [name, m] : metrics_) {
        m.history.clear();
        m.dirty = true;
    }
    training_ = false;
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

    // Status, episodes, reset.
    const bool running = training_.load();
    ui::StatusPill("##rl_status", running ? "Training" : "Stopped", running ? t.success : t.pending);
    ImGui::SameLine();
    if (rewards.empty()) ImGui::TextColored(t.text_dim, "No episodes reported yet");
    else ImGui::Text("%s episodes reported", plot::Thousands(static_cast<long long>(rewards.size())).c_str());
    ImGui::SameLine();
    const char* reset = "Reset metrics";
    const float w = ui::ButtonWidth(reset, ui::ButtonSize::Small);
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - w));
    if (ui::SecondaryButton(reset)) ResetRLMetrics();

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
            const float h = std::max(140.0f, (ImGui::GetContentRegionAvail().y - 100.0f) / 3.0f);
            RenderMetric("policy_loss", h);
            RenderMetric("value_loss", h);
            RenderMetric("explained_variance", h);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    if (rewards.empty() && !running)
        ImGui::TextColored(t.text_dim, "Start RL training with Train RL on the canvas, or report metrics from a script "
                                       "with pycyxwiz.rl_update_metric.");
    ImGui::End();

    // Charts opened in their own windows.
    for (const auto& name : order_) {
        auto& m = metrics_[name];
        plot::PlotView::Options o;
        o.export_name = name;
        m.view->DrawOwnWindow(o);
    }
}

void TrainingDashboardPanel::RenderKpis() {
    const auto& reward = metrics_["episode_reward"].history;
    const auto& length = metrics_["episode_length"].history;
    const auto& policy = metrics_["policy_loss"].history;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float w = (ImGui::GetContentRegionAvail().x - 2 * gap) / 3.0f;
    const auto mean = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    Card("##kpi_reward", "Episode reward", reward.empty() ? "-" : Format(reward.back(), "%.2f"),
         reward.empty() ? "" : "best " + Format(*std::max_element(reward.begin(), reward.end()), "%.2f") + " \xC2\xB7 mean " +
                                    Format(mean(reward), "%.2f"),
         w);
    ImGui::SameLine();
    Card("##kpi_length", "Episode length", length.empty() ? "-" : Format(length.back(), "%.0f"),
         length.empty() ? "" : "mean " + Format(mean(length), "%.0f"), w);
    ImGui::SameLine();
    Card("##kpi_policy", "Policy loss", policy.empty() ? "-" : Format(policy.back(), "%.4f"),
         policy.empty() ? "" : plot::Thousands(static_cast<long long>(policy.size())) + " updates", w);
}

void TrainingDashboardPanel::RenderMetric(const std::string& name, float height) {
    auto& m = metrics_[name];
    if (m.dirty) {
        // One series: the value by report number, reduced when long.
        plot::PlotSpec spec;
        spec.kind = plot::Kind::Line;
        spec.x_column = "report";
        spec.y_columns = {m.display_name};
        spec.title = m.display_name;
        spec.x_label = name == "episode_reward" || name == "episode_length" ? "episode" : "update";
        spec.legend = false;
        plot::Source src;
        plot::SourceColumn x, y;
        x.name = "report";
        x.numbers.resize(m.history.size());
        std::iota(x.numbers.begin(), x.numbers.end(), 1.0);
        y.name = m.display_name;
        y.numbers = m.history;
        src.columns = {std::move(x), std::move(y)};
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
