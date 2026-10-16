#include "data_studio_publish.h"

#include <cctype>
#include <filesystem>
#include <set>

namespace cyxwiz::studio {

using gui::NodeType;
namespace fs = std::filesystem;

namespace {

std::string Get(const std::map<std::string, std::string>& p, const char* key) {
    const auto it = p.find(key);
    return it == p.end() ? std::string() : it->second;
}

const char* KindName(NodeType type) {
    switch (type) {
        case NodeType::ExportCSV: return "Export CSV";
        case NodeType::ExportJSON: return "Export JSON";
        case NodeType::ExportParquet: return "Export Parquet";
        case NodeType::DataOutput: return "Data Output";
        default: return "output";
    }
}

std::string Extension(NodeType type, const std::map<std::string, std::string>& p) {
    switch (type) {
        case NodeType::ExportCSV: return ".csv";
        case NodeType::ExportJSON: return ".json";
        case NodeType::ExportParquet: return ".parquet";
        case NodeType::DataOutput: return Get(p, "format") == "parquet" ? ".parquet" : ".csv";
        default: return "";
    }
}

}  // namespace

bool IsOutputNode(NodeType type) {
    return type == NodeType::ExportCSV || type == NodeType::ExportJSON || type == NodeType::ExportParquet ||
           type == NodeType::DataOutput;
}

std::string OutputPath(NodeType type, const std::map<std::string, std::string>& p, const std::string& project_root,
                       const std::string& exports_root) {
    // The executor reads "file_path", then "path" (NormalizeDataOutputPath).
    std::string raw = Get(p, "file_path");
    if (raw.empty()) raw = Get(p, "path");
    if (raw.empty()) return "";
    fs::path path(raw);
    if (path.extension().empty()) path += Extension(type, p);
    if (path.is_relative()) {
        const bool project = Get(p, "path_base") == "project" && !project_root.empty();
        path = fs::path(project ? project_root : exports_root) / path;
    }
    return path.lexically_normal().generic_string();
}

std::vector<OutputRow> OutputRows(const std::vector<GraphNode>& nodes, const std::string& project_root,
                                  const std::string& exports_root) {
    const auto by_id = [&](int id) -> const GraphNode* {
        for (const auto& n : nodes)
            if (n.id == id) return &n;
        return nullptr;
    };
    std::vector<OutputRow> rows;
    for (const auto& n : nodes) {
        if (!IsOutputNode(n.type)) continue;
        OutputRow r;
        r.id = n.id;
        r.name = n.name;
        r.kind = KindName(n.type);
        const GraphNode* from = n.input_from.empty() ? nullptr : by_id(n.input_from.front());
        std::set<int> seen{n.id};
        while (from && from->type == NodeType::RowCountCheck && seen.insert(from->id).second)
            from = from->input_from.empty() ? nullptr : by_id(from->input_from.front());
        if (from) {
            r.reads = from->name;
            r.reads_id = from->id;
        }
        if (!n.error.empty()) {
            r.writes = OutputPath(n.type, n.parameters, project_root, exports_root);
            r.last = "failed";
            r.detail = n.error;
            r.tone = Tone::Error;
        } else if (n.ran && !n.written_path.empty()) {
            r.writes = fs::path(n.written_path).generic_string();
            r.written = true;
            r.last = (n.rows >= 0 ? Thousands(n.rows) + " rows \xC2\xB7 " : std::string()) + n.finished_at;
            r.tone = Tone::Ok;
        } else {
            r.writes = OutputPath(n.type, n.parameters, project_root, exports_root);
            r.last = "not written yet";
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

std::string SuggestedExportPath(const std::string& step_name) {
    std::string stem;
    for (char ch : step_name) {
        const unsigned char c = static_cast<unsigned char>(ch);
        stem += std::isalnum(c) || ch == '-' || ch == '_' ? ch : '_';
    }
    while (!stem.empty() && stem.back() == '_') stem.pop_back();
    if (stem.empty()) stem = "dataset";
    return "datasets/published/" + stem + ".parquet";
}

}  // namespace cyxwiz::studio
