#include "dashboard_links.h"

#include "../ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <cctype>

namespace cyxwiz::dashboard {

DashboardLinkHooks& DashboardLinks() {
    static DashboardLinkHooks hooks;
    return hooks;
}

bool DrawAddToDashboardItems(const std::string& dataset, const WidgetSpec& widget) {
    const ui::Tokens& t = ui::CurrentTokens();
    const DashboardLinkHooks& h = DashboardLinks();
    if (!h.list || !h.add) {
        ImGui::TextColored(t.text_dim, "Open a graph in CyxWiz Studio first.");
        return false;
    }
    bool added = false;
    int shown = 0;
    for (const auto& d : h.list(dataset)) {
        ImGui::PushID(d.node_id);
        std::string label = d.name;
        if (!d.feeder.empty()) label += " on " + d.feeder;
        label += "  \xC2\xB7  " + std::to_string(d.widgets) + (d.widgets == 1 ? " widget" : " widgets");
        if (ImGui::MenuItem(label.c_str(), nullptr, false, d.same_data)) {
            h.add(d.node_id, dataset, widget);
            added = true;
        }
        if (!d.same_data && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", d.reason.c_str());
        ImGui::PopID();
        ++shown;
    }
    const std::string source = h.new_source ? h.new_source(dataset) : std::string();
    if (!source.empty()) {
        if (shown > 0) ImGui::Separator();
        const std::string label = "New Dashboard node on " + source;
        if (ImGui::MenuItem(label.c_str())) {
            h.add(0, dataset, widget);
            added = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds a Dashboard node connected to the Data Input, with the automatic layout and this widget.");
    } else if (shown == 0) {
        ImGui::TextColored(t.text_dim, "No Dashboard node shows this data, and it does not come from a Data Input in the graph.");
    }
    ImGui::Spacing();
    ImGui::TextColored(t.text_faint, "Only dashboards on this data can take it; its filters then apply.");
    return added;
}

WidgetSpec QueryResultWidget(const std::string& sql, const std::string& table_name, const std::shared_ptr<arrow::Schema>& schema,
                             std::string* reason) {
    WidgetSpec w;
    w.type = WidgetType::Plot;
    w.query = sql;
    w.query_table = table_name;
    w.plot.legend = false;  // one series
    const auto lower = [](std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    if (table_name.empty() || lower(sql).find(lower(table_name)) == std::string::npos) {
        if (reason) *reason = "The query does not read " + (table_name.empty() ? std::string("the picked dataset") : "\"" + table_name + "\"") + ".";
        return w;
    }
    std::vector<std::string> texts, numbers;
    for (const auto& f : schema ? schema->fields() : arrow::FieldVector{}) {
        const auto id = f->type()->id();
        if (arrow::is_integer(id) || arrow::is_floating(id) || id == arrow::Type::DECIMAL128) numbers.push_back(f->name());
        else if (id == arrow::Type::STRING || id == arrow::Type::LARGE_STRING || id == arrow::Type::BOOL) texts.push_back(f->name());
    }
    if (!texts.empty() && !numbers.empty()) {
        w.plot.kind = plot::Kind::Bar;
        w.plot.x_column = texts.front();
        w.plot.y_columns = {numbers.front()};
        w.title = numbers.front() + " by " + texts.front();
    } else if (numbers.size() >= 2) {
        w.plot.kind = plot::Kind::Scatter;
        w.plot.x_column = numbers[0];
        w.plot.y_columns = {numbers[1]};
        w.title = numbers[1] + " by " + numbers[0];
    } else if (numbers.size() == 1) {
        w.plot.kind = plot::Kind::Histogram;
        w.plot.x_column = numbers.front();
        w.title = numbers.front();
    } else {
        if (reason) *reason = "The result has no number column to plot.";
        return w;
    }
    if (reason) reason->clear();
    return w;
}

}  // namespace cyxwiz::dashboard
