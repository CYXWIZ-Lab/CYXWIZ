#include "debug_session_manager.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

namespace {

std::string NowLocalTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now_time);
#else
    localtime_r(&now_time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

DebugGraphNodeSnapshot SnapshotNode(const gui::MLNode& node) {
    DebugGraphNodeSnapshot out;
    out.id = node.id;
    out.type = static_cast<int>(node.type);
    out.name = node.name;
    out.input_count = node.inputs.size();
    out.output_count = node.outputs.size();
    out.parameters.reserve(node.parameters.size());
    for (const auto& [key, value] : node.parameters) {
        out.parameters.emplace_back(key, value);
    }
    return out;
}

DebugGraphLinkSnapshot SnapshotLink(const gui::NodeLink& link) {
    DebugGraphLinkSnapshot out;
    out.id = link.id;
    out.from_node = link.from_node;
    out.from_pin = link.from_pin;
    out.to_node = link.to_node;
    out.to_pin = link.to_pin;
    out.type = static_cast<int>(link.type);
    return out;
}

} // namespace

DebugSession DebugSessionManager::StartSession(
    const std::string& run_id,
    const std::string& mode_name,
    uint64_t graph_hash,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    size_t selected_sample_index) {
    DebugSession session;
    session.run_id = run_id;
    session.mode_name = mode_name;
    session.graph_hash = graph_hash;
    session.node_count = nodes.size();
    session.link_count = links.size();
    session.selected_sample_index = selected_sample_index;
    session.started_at = std::chrono::steady_clock::now();

    session.graph_nodes.reserve(nodes.size());
    for (const auto& node : nodes) {
        session.graph_nodes.push_back(SnapshotNode(node));
    }

    session.graph_links.reserve(links.size());
    for (const auto& link : links) {
        session.graph_links.push_back(SnapshotLink(link));
    }

    session.studio_events.push_back({
        run_id,
        NowLocalTimestamp(),
        graph_hash,
        -1,
        "DebugSession.Start",
        "started",
        "Frozen Studio graph snapshot for debugger run."
    });

    session.traces.push_back(BuildGraphSnapshotTrace(session));
    return session;
}

DebugTraceRecord DebugSessionManager::BuildGraphSnapshotTrace(const DebugSession& session) {
    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        session.run_id,
        -1,
        "GraphSnapshot",
        "DebugSession",
        "GraphSnapshot",
        DebugTraceRole::CompileArtifact,
        {},
        {},
        "graph_snapshot",
        "Studio",
        "captured");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "graph_snapshot",
        "DebugSessionManager",
        "cyxwiz-engine/src/core/debug_session_manager.cpp",
        "cyxwiz::DebugSessionManager::BuildGraphSnapshotTrace");
    trace.payload["success"] = true;
    trace.payload["mode"] = session.mode_name;
    trace.payload["graph_hash"] = session.graph_hash;
    trace.payload["node_count"] = session.node_count;
    trace.payload["link_count"] = session.link_count;
    trace.payload["selected_sample_index"] = session.selected_sample_index;

    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : session.graph_nodes) {
        nlohmann::json params = nlohmann::json::object();
        for (const auto& [key, value] : node.parameters) {
            params[key] = value;
        }
        nodes.push_back({
            {"id", node.id},
            {"type", node.type},
            {"name", node.name},
            {"input_count", node.input_count},
            {"output_count", node.output_count},
            {"parameters", params}
        });
    }
    trace.payload["nodes"] = nodes;

    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : session.graph_links) {
        links.push_back({
            {"id", link.id},
            {"from_node", link.from_node},
            {"from_pin", link.from_pin},
            {"to_node", link.to_node},
            {"to_pin", link.to_pin},
            {"type", link.type}
        });
    }
    trace.payload["links"] = links;

    return trace;
}

bool DebugSessionManager::FullWorkflowSucceeded(
    bool compile_success,
    bool preflight_ready,
    bool smoke_supported,
    bool smoke_success,
    bool has_debug_result,
    bool debug_success) {
    return compile_success &&
        preflight_ready &&
        smoke_supported &&
        smoke_success &&
        has_debug_result &&
        debug_success;
}

} // namespace cyxwiz
