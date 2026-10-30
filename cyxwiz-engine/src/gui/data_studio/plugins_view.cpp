#include "plugins_view.h"

#include "../../core/dataset_column_stream.h"
#include "../../core/file_dialogs.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/registries/plugin_analytics_registry.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>

namespace cyxwiz {

namespace {

namespace ep = explore_plugins;

void Faint(const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine(ImGui::CalcTextSize("Analytic").x + ImGui::GetStyle().ItemSpacing.x * 2.0f);
}

ep::ColumnKind KindOf(plugin::AnalyticsColumns c) {
    switch (c) {
        case plugin::AnalyticsColumns::Numeric: return ep::ColumnKind::Numeric;
        case plugin::AnalyticsColumns::Text: return ep::ColumnKind::Text;
        case plugin::AnalyticsColumns::Any: break;
    }
    return ep::ColumnKind::Any;
}

std::vector<ep::Column> ColumnsOf(const DatasetProfile& p) {
    std::vector<ep::Column> columns;
    for (const auto& c : p.columns)
        columns.push_back({c.facts.name, c.numeric, c.facts.type == ColumnFacts::Type::Text});
    return columns;
}

std::string ClockNow() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &local);
    return buf;
}

}  // namespace

void PluginsView::RefreshAnalytics() {
    const uint64_t generation = plugin::PluginAnalyticsRegistry::Instance().Generation();
    if (generation == generation_) return;
    generation_ = generation;
    analytics_.clear();
    auto& manager = plugin::PluginManager::Instance();
    for (const auto& a : plugin::PluginAnalyticsRegistry::Instance().Analytics()) {
        ep::Analytic e;
        e.id = a.info.analytics_id;
        e.name = a.info.name;
        e.description = a.info.description;
        e.category = a.info.category;
        const auto* loaded = manager.GetLoadedPlugin(a.plugin_id);
        e.plugin_name = loaded && !loaded->manifest.name.empty() ? loaded->manifest.name : a.plugin_id;
        e.plugin_version = a.plugin_version;
        e.kind = KindOf(a.info.columns);
        e.min_columns = a.info.min_columns;
        e.max_columns = a.info.max_columns;
        analytics_.push_back(std::move(e));
    }
    // A plugin loaded again brings its analytic back under the same id.
    for (const auto& a : analytics_)
        if (a.id == chosen_.id) chosen_ = a;
    if (chosen_.id.empty() && !analytics_.empty()) chosen_ = analytics_.front();
}

bool PluginsView::Loaded(const std::string& analytic_id) const {
    return std::any_of(analytics_.begin(), analytics_.end(), [&](const ep::Analytic& a) { return a.id == analytic_id; });
}

void PluginsView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    RefreshAnalytics();
    if (analytics_.empty() && chosen_.id.empty()) {
        ui::EmptyState(ICON_FA_PLUG, "No plugin adds analytics yet",
                       "Plugins can add their own questions about a dataset here. Install or enable one in the "
                       "Plugin Manager; the SDK's example Example Data Formats adds Coefficient of variation.");
        const float w = ui::ButtonWidth("Open Plugin Manager", ui::ButtonSize::Regular);
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), (ImGui::GetContentRegionAvail().x - w) * 0.5f));
        if (ui::SecondaryButton("Open Plugin Manager") && open_plugin_manager) open_plugin_manager();
        return;
    }
    if (!ExploreProfileReady(context, "Choose a table in the list to run a plugin's analytic on it.")) return;
    const auto columns = ColumnsOf(*context.profile);
    if (context.dataset != dataset_) {
        dataset_ = context.dataset;
        picked_by_user_ = false;
        message_.clear();
    }
    if (picked_by_user_) ep::KeepReadable(chosen_, columns, picked_);
    else picked_ = ep::DefaultColumns(chosen_, columns);

    RenderPicker(columns);
    RenderRun(context);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();

    if (job_.Running()) {
        const int64_t read = rows_read_->load();
        const std::string line = std::string(ICON_FA_SPINNER " ") + chosen_.name + ": reading rows, " +
                                 std::to_string(read) + " of " + std::to_string(rows_total_);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.info, "%s", line.c_str());
        ImGui::SameLine();
        if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Small)) {
            job_.Cancel();
            asked_key_.clear();
        }
        if (rows_total_ > 0)
            ImGui::ProgressBar(static_cast<float>(read) / static_cast<float>(rows_total_), ImVec2(420.0f, 6.0f), "");
        Faint("Task View shows the run. Cancel stops the rows, and the plugin's partial answer is dropped.");
        return;
    }
    if (asked_key_.empty()) return;
    if (const Answer* a = job_.Answer(asked_key_)) {
        RenderAnswer(context, *a, asked_key_ != ep::Key(context.dataset, chosen_.id, picked_));
    } else if (!job_.Error().empty()) {
        Wrapped(t.error, job_.Error());
    }
}

void PluginsView::RenderPicker(const std::vector<ep::Column>& columns) {
    const ui::Tokens& t = ui::CurrentTokens();
    Label("Analytic");
    ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.5f, 220.0f, 420.0f));
    const std::string preview = ep::Label(chosen_);
    if (ImGui::BeginCombo("##plugin_analytic", preview.c_str())) {
        std::string category = "\x01";
        for (const auto& a : analytics_) {
            if (a.category != category) {
                category = a.category;
                ImGui::TextColored(t.text_dim, "%s", category.empty() ? "Other" : category.c_str());
            }
            ImGui::PushID(a.id.c_str());
            const std::string row = "   " + a.name;
            if (ImGui::Selectable(row.c_str(), a.id == chosen_.id) && a.id != chosen_.id) {
                chosen_ = a;
                picked_by_user_ = false;
                message_.clear();
            }
            ImGui::SameLine();
            ImGui::TextColored(t.text_faint, "· %s", a.plugin_name.c_str());
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "%s", ep::FromLine(chosen_).c_str());
    if (!chosen_.description.empty()) Wrapped(t.text_dim, chosen_.description);

    std::vector<plot::ColumnSummary> summaries;
    for (const auto& c : columns) {
        plot::ColumnSummary s;
        s.name = c.name;
        s.numeric = c.numeric;
        summaries.push_back(std::move(s));
    }
    Label("Columns");
    if (picker_.PickMany("##plugin_columns", picked_, summaries, ep::RefusedReasons(chosen_, columns),
                         std::clamp(ImGui::GetContentRegionAvail().x, 160.0f, 420.0f)))
        picked_by_user_ = true;
    const std::string hint = ep::ColumnsHint(chosen_, columns);
    if (!hint.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "%s", hint.c_str());
    }
}

void PluginsView::RenderRun(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool loaded = Loaded(chosen_.id);
    const ep::RunState run = ep::CanRun(chosen_, picked_, static_cast<int64_t>(context.profile->rows), loaded);
    const bool can = run.enabled && !job_.Running();
    if (ui::PrimaryButton("Run", can, job_.Running() ? "Running" : run.reason.c_str())) Run(context);
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", (run.enabled ? run.summary : run.reason).c_str());
    if (!loaded) {
        ImGui::SameLine();
        if (ui::LinkButton("Open Plugin Manager") && open_plugin_manager) open_plugin_manager();
    }
}

void PluginsView::Run(const ExploreContext& context) {
    message_.clear();
    asked_key_ = ep::Key(context.dataset, chosen_.id, picked_);
    rows_read_->store(0);
    rows_total_ = static_cast<int64_t>(context.profile->rows);
    const ep::Analytic analytic = chosen_;
    const std::vector<std::string> columns = picked_;
    const std::string dataset = context.dataset;
    const auto rows_read = rows_read_;
    job_.Reset();
    job_.Ask("Plugin analytic: " + analytic.name, asked_key_, [=](LambdaTask& task) {
        Answer a;
        ColumnStreamOptions options;
        options.should_cancel = [&task] { return task.IsCancelRequested(); };
        auto total = std::make_shared<int64_t>(0);
        options.on_rows = [&task, rows_read, total](int64_t n) {
            rows_read->store(n);
            if (*total > 0) task.ReportProgress(static_cast<float>(n) / static_cast<float>(*total), "Reading rows");
        };
        std::string error;
        auto reader = OpenDatasetColumns(dataset, columns, options, error, total.get());
        if (!reader) {
            a.error = error;
            return a;
        }
        const auto start = std::chrono::steady_clock::now();
        const auto r = plugin::PluginAnalyticsRegistry::Instance().Compute(analytic.id, std::move(reader));
        if (task.IsCancelRequested()) {
            a.error = "Cancelled.";
            return a;
        }
        if (!r.success) {
            a.error = analytic.name + " failed: " + r.error_message;
            return a;
        }
        ep::Result& out = a.result;
        out.ok = true;
        out.name = r.name;
        out.category = r.category;
        out.has_scalar = r.has_scalar;
        out.scalar = r.scalar_value;
        out.labels = r.vector_labels;
        out.values.assign(r.vector_value.begin(), r.vector_value.end());
        out.text = r.text_value;
        out.plugin_name = analytic.plugin_name;
        out.plugin_version = analytic.plugin_version;
        out.rows = *total;
        out.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        out.finished_at = ClockNow();
        return a;
    });
}

void PluginsView::RenderAnswer(const ExploreContext& context, const Answer& a, bool changed) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ep::AnswerView v = ep::Present(a.result);
    const bool stale = changed || !Loaded(chosen_.id);
    if (changed) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.warning, "Choices changed since this answer.");
        ImGui::SameLine();
        const ep::RunState run = ep::CanRun(chosen_, picked_, static_cast<int64_t>(context.profile->rows), Loaded(chosen_.id));
        if (ui::LinkButton("Run again", run.enabled)) {
            Run(context);
            return;  // the old answer is gone with the new question
        }
    }
    if (stale) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.55f);
    ImGui::TextColored(t.text_dim, "ANSWER");
    if (!v.headline.empty()) {
        ImGui::SetWindowFontScale(1.4f);
        ImGui::TextUnformatted(v.headline.c_str());
        ImGui::SetWindowFontScale(1.0f);
    }
    if (!v.bars.empty() &&
        ImGui::BeginTable("##plugin_bars", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("bar", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthFixed);
        for (const auto& bar : v.bars) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(bar.label.c_str());
            ImGui::TableNextColumn();
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = std::max(40.0f, ImGui::GetContentRegionAvail().x);
            const float h = ImGui::GetTextLineHeight() * 0.7f;
            const float y = p.y + (ImGui::GetTextLineHeight() - h) * 0.5f;
            ImDrawList* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + w, y + h), ImGui::GetColorU32(t.plot_grid), 3.0f);
            draw->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + w * bar.fraction, y + h), ImGui::GetColorU32(t.accent_text), 3.0f);
            ImGui::Dummy(ImVec2(w, ImGui::GetTextLineHeight()));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(bar.value.c_str());
        }
        ImGui::EndTable();
    }
    Wrapped(t.text_dim, v.note);
    Faint(v.by_line);

    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "VALUES");
    if (!v.bars.empty() && ImGui::BeginTable("##plugin_values", 2,
                                             ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg |
                                                 ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("column");
        ImGui::TableSetupColumn(v.value_header.c_str());
        ImGui::TableHeadersRow();
        for (const auto& bar : v.bars) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(bar.label.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(bar.value.c_str());
        }
        ImGui::EndTable();
    }
    if (stale) ImGui::PopStyleVar();
    if (ui::SecondaryButton("Copy", true, nullptr, ui::ButtonSize::Small)) {
        ImGui::SetClipboardText(ep::CopyText(a.result).c_str());
        message_ = "Copied the answer.";
        message_error_ = false;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Export CSV...", true, nullptr, ui::ButtonSize::Small)) Export(a);
}

void PluginsView::Export(const Answer& a) {
    std::string name = a.result.name.empty() ? std::string("analytic") : a.result.name;
    std::replace(name.begin(), name.end(), ' ', '_');
    if (const auto path = FileDialogs::SaveFile("Export answer", {{"CSV", "csv"}}, nullptr, (name + ".csv").c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << ep::Csv(a.result);
        message_ = out ? "Saved " + *path : "Could not write " + *path;
        message_error_ = !out;
    }
}

}  // namespace cyxwiz
