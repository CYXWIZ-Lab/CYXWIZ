// Plot nodes on the canvas (TOFIX134 P2, approved boards 4-5): the status
// under each Plot node, its Plot window, and the window's header with the
// node's data state and actions.
#include "node_editor.h"

#include "icons.h"
#include "plot/plot_node_lane.h"
#include "plot/plot_window.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>
#include <imnodes.h>

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
    const auto [line1, line2] = StatusLines(st, SpecOf(node));
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

void NodeEditor::RenderPlotNodes() {
    bool any_plot = false;
    for (const auto& n : nodes_) any_plot = any_plot || n.type == NodeType::Plot;
    if (!any_plot && plot_windows_.empty()) return;
    if (!plot_lane_) plot_lane_ = std::make_shared<cyxwiz::plot::PlotNodeLane>();
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
}

}  // namespace gui
