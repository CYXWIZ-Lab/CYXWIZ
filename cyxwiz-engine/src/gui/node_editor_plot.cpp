// Plot nodes on the canvas (TOFIX134 P2, approved boards 4-5): the status
// under each Plot node, its Plot window, and the window's header with the
// node's data state and actions.
#include "node_editor.h"

#include "icons.h"
#include "plot/plot_node_lane.h"
#include "plot/plot_window.h"
#include "dashboard/dashboard_links.h"
#include "dashboard/dashboard_window.h"
#include "../core/dashboard/dashboard_model.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

namespace gui {

namespace {

using Lane = cyxwiz::plot::PlotNodeLane;
using State = Lane::Status::State;

ImVec4 StateColour(State s) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    switch (s) {
        case State::Ready: return t.success;
        case State::Running: return t.running;
        case State::OutOfDate:
        case State::Unavailable: return t.warning;
        case State::Failed: return t.error;
        default: return t.pending;
    }
}

// The two lines under a Plot node.
std::pair<std::string, std::string> StatusLines(const Lane::Status& st, const cyxwiz::plot::PlotSpec& spec) {
    switch (st.state) {
        case State::NotConnected: return {"Not connected", "Connect a table to plot it."};
        case State::Unavailable:
            return {"Not available yet", st.reason + (st.alternative_name.empty() ? std::string()
                                                                              : " Plot its input (" + st.alternative_name + ") instead.")};
        case State::Idle:
            return {"Not read yet", "Open the plot to read " + (st.feeder_name.empty() ? std::string("its input") : st.feeder_name) + "."};
        case State::Running: {
            char pct[16];
            std::snprintf(pct, sizeof(pct), "%d%%", static_cast<int>(st.progress * 100.0f));
            return {std::string("Reading ") + pct, st.progress_text};
        }
        case State::Failed: return {"Could not read the data", st.error};
        case State::OutOfDate:
        case State::Ready: {
            // No columns chosen yet (the window picks a first plot when opened).
            const std::string column = !spec.x_column.empty() ? spec.x_column
                                       : !spec.y_columns.empty() ? spec.y_columns.front() : std::string();
            const std::string what = column.empty() ? std::string("Data ready")
                                                    : std::string(cyxwiz::plot::Info(spec.kind).label) + " \xC2\xB7 " + column;
            const long long rows = st.table ? st.table->num_rows() : 0;
            if (st.state == State::OutOfDate)
                return {"Out of date", st.feeder_name + " changed; refresh in the plot (data of " + st.read_at + ")."};
            return {what, cyxwiz::plot::Thousands(rows) + " rows \xC2\xB7 read " + st.read_at};
        }
    }
    return {"", ""};
}

cyxwiz::plot::PlotSpec SpecOf(const MLNode& node) {
    cyxwiz::plot::PlotSpec spec;
    auto it = node.parameters.find("plot_spec");
    if (it != node.parameters.end() && !it->second.empty()) cyxwiz::plot::SpecFromJson(it->second, spec);
    return spec;
}

}  // namespace

void NodeEditor::DrawPlotNodeStatus(const MLNode& node) {
    if (!plot_lane_) return;
    const Lane::Status& st = plot_lane_->StatusOf(node.id);
    auto [line1, line2] = StatusLines(st, SpecOf(node));
    if (node.type == NodeType::Dashboard && (st.state == State::Ready || st.state == State::Idle)) {
        cyxwiz::dashboard::DashboardSpec spec;
        auto it = node.parameters.find("dashboard_spec");
        if (it != node.parameters.end() && !it->second.empty()) cyxwiz::dashboard::DashboardFromJson(it->second, spec);
        if (st.state == State::Ready)
            line1 = spec.widgets.empty() ? std::string("Dashboard \xC2\xB7 automatic layout on open")
                                         : "Dashboard \xC2\xB7 " + std::to_string(spec.widgets.size()) + " widgets";
        else
            line2 = "Open the dashboard to read " + (st.feeder_name.empty() ? std::string("its input") : st.feeder_name) + ".";
    }
    const auto& t = cyxwiz::ui::CurrentTokens();
    const ImVec2 pos = ImNodes::GetNodeScreenSpacePos(node.id);
    const ImVec2 dims = ImNodes::GetNodeDimensions(node.id);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImFont* font = ImGui::GetFont();
    const float size = 13.0f * zoom_;
    const float pad = 6.0f * zoom_;
    const float width = std::max(dims.x * 1.6f, 230.0f * zoom_);
    const float x = pos.x + dims.x * 0.5f - width * 0.5f;
    const float y = pos.y + dims.y + 8.0f * zoom_;
    const ImVec2 l2 = font->CalcTextSizeA(size, FLT_MAX, width - pad * 2, line2.c_str());
    const float height = pad * 2 + size * 1.25f + l2.y + 2.0f * zoom_;
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + width, y + height), cyxwiz::ui::ToU32(cyxwiz::ui::WithAlpha(t.bg_panel, 0.94f)),
                      5.0f * zoom_);
    dl->AddCircleFilled(ImVec2(x + pad + size * 0.3f, y + pad + size * 0.55f), size * 0.28f, cyxwiz::ui::ToU32(StateColour(st.state)));
    dl->AddText(font, size, ImVec2(x + pad + size * 0.8f, y + pad), cyxwiz::ui::ToU32(t.text), line1.c_str());
    dl->AddText(font, size, ImVec2(x + pad, y + pad + size * 1.25f + 2.0f * zoom_), cyxwiz::ui::ToU32(t.text_dim), line2.c_str(),
                nullptr, width - pad * 2);
}

void NodeEditor::OpenPlotNode(int node_id) {
    MLNode* node = FindNodeById(node_id);
    if (!node) return;
    if (!plot_lane_) plot_lane_ = std::make_shared<cyxwiz::plot::PlotNodeLane>();
    auto& window = plot_windows_[node_id];
    if (!window) {
        window = std::make_shared<cyxwiz::plot::PlotWindow>("plot_node_" + std::to_string(node_id));
        window->SetSpec(SpecOf(*node));
        window->draw_header = [this, node_id]() { DrawPlotNodeHeader(node_id); };
        // The plot type, columns and labels are saved in the node.
        window->on_spec_changed = [this, node_id](const cyxwiz::plot::PlotSpec& spec) {
            if (MLNode* n = FindNodeById(node_id)) {
                const std::string json = cyxwiz::plot::SpecToJson(spec);
                if (n->parameters["plot_spec"] != json) n->parameters["plot_spec"] = json;
            }
        };
        plot_window_data_versions_[node_id] = ~0ull;
    }
    window->visible = true;
    // Opening is the user asking for the data: read or run once if there is
    // none yet (later edits above only mark it out of date).
    const Lane::Status& st = plot_lane_->StatusOf(node_id);
    if (!st.table && st.state != State::Running) plot_lane_->Refresh(node_id, nodes_, links_, task_owner_token_);
}

void NodeEditor::PlotNodeInput(int plot_id, int source_id) {
    MLNode* plot = FindNodeById(plot_id);
    MLNode* source = FindNodeById(source_id);
    if (!plot || !source || plot->inputs.empty() || source->outputs.empty()) return;
    std::erase_if(links_, [plot_id](const NodeLink& l) { return l.to_node == plot_id; });
    CreateLink(source->outputs.front().id, plot->inputs.front().id, source_id, plot_id);
    if (plot_lane_) plot_lane_->Refresh(plot_id, nodes_, links_, task_owner_token_);
}

void NodeEditor::DrawPlotNodeHeader(int node_id) {
    if (!plot_lane_) return;
    const Lane::Status& st = plot_lane_->StatusOf(node_id);
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Text, StateColour(st.state));
    ImGui::Bullet();
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (st.feeder_name.empty()) {
        ImGui::TextUnformatted("Nothing is connected to this Plot node.");
    } else {
        ImGui::Text("Data at %s", st.feeder_name.c_str());
        if (st.table) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s rows \xC3\x97 %d columns \xC2\xB7 read %s",
                               cyxwiz::plot::Thousands(st.table->num_rows()).c_str(), st.table->num_columns(), st.read_at.c_str());
        }
    }
    // Actions at the right: Refresh, or Cancel while running.
    const bool running = st.state == State::Running;
    const char* action = running ? "Cancel" : "Refresh";
    const float w = cyxwiz::ui::ButtonWidth(action, cyxwiz::ui::ButtonSize::Small);
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - w));
    const bool can_refresh = st.state != State::NotConnected && st.state != State::Unavailable;
    if (running) {
        if (cyxwiz::ui::SecondaryButton("Cancel##plot_cancel")) plot_lane_->Cancel(node_id);
    } else if (cyxwiz::ui::SecondaryButton("Refresh##plot_refresh", can_refresh, "Nothing can be read here")) {
        plot_lane_->Refresh(node_id, nodes_, links_, task_owner_token_);
    }
    if (ImGui::IsItemHovered() && !running && can_refresh && st.needs_run)
        ImGui::SetTooltip("Runs the %d node(s) above the plot (progress in Task View).", st.run_node_count);

    // Banners: running, out of date, unavailable, failed.
    const auto banner = [&](const ImVec4& tint, const std::function<void()>& body) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, cyxwiz::ui::WithAlpha(tint, 0.12f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
        ImGui::BeginChild("##plot_banner", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        body();
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    };
    if (running) {
        ImGui::ProgressBar(st.progress, ImVec2(-1, 0), st.progress_text.c_str());
    } else if (st.state == State::OutOfDate) {
        banner(t.warning, [&]() {
            ImGui::Text("Out of date: %s changed after %s.", st.feeder_name.c_str(), st.read_at.c_str());
            ImGui::TextColored(t.text_dim, "The plot shows the data of %s until you refresh.", st.read_at.c_str());
        });
    } else if (st.state == State::Unavailable) {
        banner(t.warning, [&]() {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(st.reason.c_str());
            ImGui::PopTextWrapPos();
            if (st.alternative_id >= 0) {
                const std::string label = "Plot its input (" + st.alternative_name + ")";
                if (cyxwiz::ui::SecondaryButton(label.c_str())) PlotNodeInput(node_id, st.alternative_id);
            }
        });
    } else if (st.state == State::Failed) {
        banner(t.error, [&]() {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(st.error.c_str());
            ImGui::PopTextWrapPos();
        });
    }
}

void NodeEditor::OpenDashboardNode(int node_id) {
    MLNode* node = FindNodeById(node_id);
    if (!node) return;
    if (!plot_lane_) plot_lane_ = std::make_shared<cyxwiz::plot::PlotNodeLane>();
    auto& window = dashboard_windows_[node_id];
    if (!window) {
        window = std::make_shared<cyxwiz::dashboard::DashboardWindow>("dashboard_node_" + std::to_string(node_id));
        auto it = node->parameters.find("dashboard_spec");
        if (it != node->parameters.end()) window->SetSpecJson(it->second);
        window->draw_header = [this, node_id]() { DrawPlotNodeHeader(node_id); };
        // The layout and filters are saved in the node.
        window->on_spec_changed = [this, node_id](const std::string& json) {
            if (MLNode* n = FindNodeById(node_id))
                if (n->parameters["dashboard_spec"] != json) n->parameters["dashboard_spec"] = json;
        };
        window->on_edit_roles = [this](const std::string& dataset) {
            if (open_data_studio_profile_) open_data_studio_profile_(dataset);
        };
        window->on_open_query = [this](const std::string& dataset, const std::string& sql) {
            if (open_data_studio_query_) open_data_studio_query_(dataset, sql);
        };
        dashboard_data_versions_[node_id] = ~0ull;
    }
    window->visible = true;
    const Lane::Status& st = plot_lane_->StatusOf(node_id);
    if (!st.table && st.state != State::Running) plot_lane_->Refresh(node_id, nodes_, links_, task_owner_token_);
}

int NodeEditor::DataInputOf(const std::string& dataset) const {
    if (dataset.empty()) return -1;
    for (const auto& n : nodes_) {
        if (n.type != NodeType::DataInput) continue;
        if (dataset == "ds_datainput_" + std::to_string(n.id)) return n.id;
        auto it = n.parameters.find("dataset_name");
        if (it != n.parameters.end() && it->second == dataset) return n.id;
    }
    return -1;
}

std::vector<cyxwiz::dashboard::DashboardTarget> NodeEditor::DashboardTargets(const std::string& dataset) {
    // Data is "the same" when both name one Data Input (or the same dataset).
    const auto key_of = [this](const std::string& name) {
        const int id = DataInputOf(name);
        return id >= 0 ? "node:" + std::to_string(id) : name;
    };
    const std::string want = key_of(dataset);
    std::vector<cyxwiz::dashboard::DashboardTarget> out;
    for (const auto& n : nodes_) {
        if (n.type != NodeType::Dashboard) continue;
        cyxwiz::dashboard::DashboardTarget d;
        d.node_id = n.id;
        d.name = n.name;
        auto spec_it = n.parameters.find("dashboard_spec");
        cyxwiz::dashboard::DashboardSpec spec;
        if (spec_it != n.parameters.end() && cyxwiz::dashboard::DashboardFromJson(spec_it->second, spec)) d.widgets = static_cast<int>(spec.widgets.size());
        // What it shows: the table its lane read, else the node connected to it.
        std::string key;
        if (plot_lane_) {
            const Lane::Status& st = plot_lane_->StatusOf(n.id);
            d.feeder = st.feeder_name;
            if (!st.dataset_name.empty()) key = key_of(st.dataset_name);
        }
        if (key.empty())
            for (const auto& l : links_)
                if (l.to_node == n.id)
                    if (const MLNode* from = FindNodeById(l.from_node)) {
                        if (d.feeder.empty()) d.feeder = from->name;
                        if (from->type == NodeType::DataInput) key = "node:" + std::to_string(from->id);
                    }
        d.same_data = !key.empty() && key == want;
        if (!d.same_data)
            d.reason = d.feeder.empty() ? "Nothing is connected to this dashboard."
                       : key.empty()    ? "It shows " + d.feeder + ", not this dataset (open it once to read its data)."
                                        : "It shows " + d.feeder + ", not this dataset.";
        out.push_back(std::move(d));
    }
    return out;
}

std::string NodeEditor::DashboardSourceFor(const std::string& dataset) const {
    const int id = DataInputOf(dataset);
    for (const auto& n : nodes_)
        if (n.id == id) return n.name;
    return {};
}

void NodeEditor::QueueDashboardWidget(int node_id, const std::string& dataset, const cyxwiz::dashboard::WidgetSpec& widget) {
    pending_dashboard_widgets_.push_back({node_id, dataset, std::make_shared<cyxwiz::dashboard::WidgetSpec>(widget)});
}

void NodeEditor::ApplyPendingDashboardWidgets() {
    auto pending = std::move(pending_dashboard_widgets_);
    pending_dashboard_widgets_.clear();
    for (const auto& add : pending) {
        int id = add.node_id;
        if (id == 0) {
            // A new Dashboard node to the right of the Data Input, connected to it.
            const int source_id = DataInputOf(add.dataset);
            const MLNode* source = FindNodeById(source_id);
            if (!source || source->outputs.empty()) {
                spdlog::warn("Add to Dashboard: no Data Input in the graph reads '{}'", add.dataset);
                continue;
            }
            const int source_pin = source->outputs.front().id;
            const ImVec2 at = ImNodes::GetNodeGridSpacePos(source_id);
            SaveUndoState();
            MLNode node = CreateNode(NodeType::Dashboard, "Dashboard");
            id = node.id;
            const int input_pin = node.inputs.empty() ? -1 : node.inputs.front().id;
            nodes_.push_back(std::move(node));
            pending_positions_[id] = ImVec2(at.x + 320.0f, at.y + 140.0f);
            pending_positions_frames_ = 3;
            if (input_pin >= 0) CreateLink(source_pin, input_pin, source_id, id);
            RebuildPinLookup();
        }
        MLNode* node = FindNodeById(id);
        if (!node || node->type != NodeType::Dashboard) continue;
        cyxwiz::dashboard::DashboardSpec spec;
        auto it = node->parameters.find("dashboard_spec");
        if (it != node->parameters.end() && !it->second.empty() && !cyxwiz::dashboard::DashboardFromJson(it->second, spec)) {
            spdlog::warn("Add to Dashboard: the layout of '{}' could not be read; not changed", node->name);
            continue;
        }
        cyxwiz::dashboard::WidgetSpec w = *add.widget;
        w.id = spec.NewId();
        w.automatic = false;
        w.at = {0, 1000, 6, 3};  // after the others, half the width
        spec.widgets.push_back(w);
        const std::string json = cyxwiz::dashboard::DashboardToJson(spec);
        node->parameters["dashboard_spec"] = json;
        auto win = dashboard_windows_.find(id);
        if (win != dashboard_windows_.end()) win->second->SetSpecJson(json);
        OpenDashboardNode(id);
        dashboard_windows_[id]->Select(w.id);
        spdlog::info("Add to Dashboard: {} of '{}'{} into {} (node {})", cyxwiz::plot::Info(w.plot.kind).label, w.plot.x_column,
                     w.IsQuery() ? " (query widget)" : "", node->name, id);
    }
}

void NodeEditor::RenderDashboardWindows() {
    for (auto it = dashboard_windows_.begin(); it != dashboard_windows_.end();) {
        const int id = it->first;
        MLNode* node = FindNodeById(id);
        if (!node || node->type != NodeType::Dashboard) {  // the node was deleted
            plot_lane_->Forget(id);
            dashboard_data_versions_.erase(id);
            it = dashboard_windows_.erase(it);
            continue;
        }
        auto& window = it->second;
        // The node's layout changed elsewhere (undo, redo, a reload): the window follows.
        {
            auto spec_it = node->parameters.find("dashboard_spec");
            if (spec_it != node->parameters.end() && !spec_it->second.empty() && spec_it->second != window->SavedJson())
                window->SetSpecJson(spec_it->second);
        }
        const Lane::Status& st = plot_lane_->StatusOf(id);
        const bool has_data = st.table && !st.dataset_name.empty() &&
                              (st.state == State::Ready || st.state == State::OutOfDate || st.state == State::Running);
        if (has_data) {
            window->SetData(st.dataset_name, node->name + " \xC2\xB7 " + st.feeder_name);
        } else {
            // The header banner already gives the reason when not available or
            // failed (it was shown twice); the body then only names the state.
            const auto [line1, line2] = StatusLines(st, cyxwiz::plot::PlotSpec{});
            const bool banner = st.state == State::Unavailable || st.state == State::Failed;
            window->ClearData(banner ? line1 + "." : line1 + ". " + line2);
        }
        window->Render();
        ++it;
    }
}

void NodeEditor::RenderPlotNodes() {
    bool any_plot = !pending_dashboard_widgets_.empty();
    for (const auto& n : nodes_) any_plot = any_plot || n.type == NodeType::Plot || n.type == NodeType::Dashboard;
    if (!any_plot && plot_windows_.empty() && dashboard_windows_.empty()) return;
    if (!plot_lane_) plot_lane_ = std::make_shared<cyxwiz::plot::PlotNodeLane>();
    if (!pending_dashboard_widgets_.empty()) ApplyPendingDashboardWidgets();
    plot_lane_->Poll(nodes_, links_, task_owner_token_);

    for (auto it = plot_windows_.begin(); it != plot_windows_.end();) {
        const int id = it->first;
        MLNode* node = FindNodeById(id);
        if (!node || node->type != NodeType::Plot) {  // the node was deleted
            plot_lane_->Forget(id);
            plot_window_data_versions_.erase(id);
            it = plot_windows_.erase(it);
            continue;
        }
        auto& window = it->second;
        // The node's settings changed elsewhere (undo, redo, a reload): the window follows.
        {
            auto spec_it = node->parameters.find("plot_spec");
            if (spec_it != node->parameters.end() && !spec_it->second.empty() &&
                spec_it->second != cyxwiz::plot::SpecToJson(window->Spec()))
                window->SetSpec(SpecOf(*node));
        }
        const Lane::Status& st = plot_lane_->StatusOf(id);
        uint64_t& shown = plot_window_data_versions_[id];
        const bool has_data = st.table && (st.state == State::Ready || st.state == State::OutOfDate || st.state == State::Running);
        if (has_data && shown != st.data_version) {
            window->SetArrowTable(node->name + " \xC2\xB7 " + st.feeder_name, st.table);
            shown = st.data_version;
        } else if (!has_data) {
            // The header banner already gives the reason when not available
            // or failed; the plot area then only names the state.
            const auto [line1, line2] = StatusLines(st, window->Spec());
            const bool banner = st.state == State::Unavailable || st.state == State::Failed;
            window->ClearData(node->name, banner ? line1 + "." : line1 + ". " + line2);
            shown = 0;
        }
        window->Render();
        ++it;
    }
    RenderDashboardWindows();
}

}  // namespace gui
