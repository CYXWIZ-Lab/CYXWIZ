#include "dataset_lineage.h"

#include "sha256_digest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

std::string Thousands(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + digits : digits;
}

std::string Bytes(uint64_t b) {
    char buf[32];
    if (b < 1024) std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(b));
    else if (b < 1024ull * 1024) std::snprintf(buf, sizeof(buf), "%.1f KB", static_cast<double>(b) / 1024.0);
    else if (b < 1024ull * 1024 * 1024) std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(b) / (1024.0 * 1024.0));
    else std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

nlohmann::json FileJson(const LineageFile& f) {
    nlohmann::json j = {{"path", f.path}, {"bytes", f.bytes}, {"sha256", f.sha256}};
    if (!f.modified.empty()) j["modified"] = f.modified;
    if (!f.note.empty()) j["note"] = f.note;
    return j;
}

LineageFile FileFrom(const nlohmann::json& j) {
    LineageFile f;
    f.path = j.value("path", std::string());
    f.bytes = j.value("bytes", uint64_t{0});
    f.sha256 = j.value("sha256", std::string());
    f.modified = j.value("modified", std::string());
    f.note = j.value("note", std::string());
    return f;
}

}  // namespace

std::string LineagePathFor(const std::string& output_path) { return output_path + ".lineage.json"; }

std::string RecipeRevision(const std::vector<LineageStep>& steps, const std::string& recipe) {
    nlohmann::json canonical = nlohmann::json::array();
    for (const auto& s : steps) {
        if (recipe.empty() || s.recipe != recipe) continue;
        nlohmann::json params = nlohmann::json::object();
        for (const auto& [k, v] : s.parameters) params[k] = v;  // std::map: sorted keys
        canonical.push_back({{"type", s.type}, {"parameters", params}});
    }
    if (canonical.empty()) return "";
    Sha256Hasher hasher;
    std::string digest, error;
    const std::string text = canonical.dump();
    if (!hasher.Update(text, error) || !hasher.Finish(digest, error)) return "";
    return digest.substr(0, 12);
}

std::vector<LineageCheck> LineageChecks(const std::vector<LineageStep>& steps) {
    std::vector<LineageCheck> checks;
    for (const auto& s : steps) {
        if (s.type != "RowCountCheck") continue;
        checks.push_back({s.name, s.outcome, s.outcome.rfind("passed", 0) == 0});
    }
    return checks;
}

nlohmann::json LineageToJson(const DatasetLineage& l) {
    nlohmann::json j;
    j["lineage_version"] = l.version;
    j["written_at"] = l.written_at;
    j["engine"] = l.engine;
    j["graph"] = l.graph;
    j["output"] = FileJson(l.output);
    j["format"] = l.format;
    j["rows"] = l.rows;
    j["columns"] = l.columns;
    j["schema"] = nlohmann::json::array();
    for (const auto& [name, type] : l.schema) j["schema"].push_back({{"name", name}, {"type", type}});
    j["inputs"] = nlohmann::json::array();
    for (const auto& f : l.inputs) j["inputs"].push_back(FileJson(f));
    j["recipe"] = l.recipe;
    j["recipe_revision"] = l.recipe_revision;
    j["steps"] = nlohmann::json::array();
    for (const auto& s : l.steps)
        j["steps"].push_back({{"id", s.id}, {"type", s.type}, {"name", s.name}, {"recipe", s.recipe},
                              {"parameters", s.parameters}, {"inputs", s.inputs}, {"outcome", s.outcome}});
    j["checks"] = nlohmann::json::array();
    for (const auto& c : l.checks) j["checks"].push_back({{"name", c.name}, {"outcome", c.outcome}, {"passed", c.passed}});
    return j;
}

bool LineageFromJson(const nlohmann::json& j, DatasetLineage& l, std::string& error) {
    try {
        l = {};
        l.version = j.value("lineage_version", 0);
        if (l.version < 1) {
            error = "not a lineage file (no lineage_version)";
            return false;
        }
        l.written_at = j.value("written_at", std::string());
        l.engine = j.value("engine", std::string());
        l.graph = j.value("graph", std::string());
        if (j.contains("output")) l.output = FileFrom(j.at("output"));
        l.format = j.value("format", std::string());
        l.rows = j.value("rows", int64_t{-1});
        l.columns = j.value("columns", int64_t{-1});
        for (const auto& c : j.value("schema", nlohmann::json::array()))
            l.schema.emplace_back(c.value("name", std::string()), c.value("type", std::string()));
        for (const auto& f : j.value("inputs", nlohmann::json::array())) l.inputs.push_back(FileFrom(f));
        l.recipe = j.value("recipe", std::string());
        l.recipe_revision = j.value("recipe_revision", std::string());
        for (const auto& s : j.value("steps", nlohmann::json::array())) {
            LineageStep step;
            step.id = s.value("id", -1);
            step.type = s.value("type", std::string());
            step.name = s.value("name", std::string());
            step.recipe = s.value("recipe", std::string());
            step.parameters = s.value("parameters", std::map<std::string, std::string>{});
            step.inputs = s.value("inputs", std::vector<int>{});
            step.outcome = s.value("outcome", std::string());
            l.steps.push_back(std::move(step));
        }
        for (const auto& c : j.value("checks", nlohmann::json::array()))
            l.checks.push_back({c.value("name", std::string()), c.value("outcome", std::string()), c.value("passed", false)});
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool WriteLineage(const DatasetLineage& lineage, std::string& error) {
    if (lineage.output.path.empty()) {
        error = "no output path";
        return false;
    }
    const fs::path target = LineagePathFor(lineage.output.path);
    const fs::path temp = target.string() + ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot write " + temp.string();
            return false;
        }
        out << LineageToJson(lineage).dump(2) << "\n";
        if (!out) {
            error = "cannot write " + temp.string();
            return false;
        }
    }
    std::error_code ec;
    fs::rename(temp, target, ec);
    if (ec) {
        fs::remove(temp, ec);
        error = "cannot replace " + target.string();
        return false;
    }
    return true;
}

bool ReadLineage(const std::string& output_path, DatasetLineage& lineage, std::string& error) {
    const fs::path path = LineagePathFor(output_path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "no lineage file";
        return false;
    }
    try {
        return LineageFromJson(nlohmann::json::parse(in), lineage, error);
    } catch (const std::exception& e) {
        error = std::string("the lineage file cannot be read: ") + e.what();
        return false;
    }
}

std::string ShortDigest(const std::string& sha256) {
    if (sha256.size() < 8) return sha256.empty() ? "not read" : sha256;
    return sha256.substr(0, 4) + "\xE2\x80\xA6" + sha256.substr(sha256.size() - 2);
}

LineageSummary SummarizeLineage(const DatasetLineage& l) {
    LineageSummary s;
    const size_t steps = l.steps.size();
    const std::string count = std::to_string(steps) + (steps == 1 ? " step" : " steps");
    if (!l.recipe.empty())
        s.made_by = "recipe " + l.recipe + (l.recipe_revision.empty() ? "" : ", revision " + l.recipe_revision) + " (" + count + ")";
    else s.made_by = count + " of the graph";
    for (size_t i = 0; i < l.inputs.size(); ++i) {
        const auto& f = l.inputs[i];
        s.inputs += (i ? "; " : "") + fs::path(f.path).filename().string() + " \xC2\xB7 " + Bytes(f.bytes) + " \xC2\xB7 " +
                    (f.sha256.empty() && !f.note.empty() ? "not hashed (large file; size and time identify it)"
                                                         : "sha256 " + ShortDigest(f.sha256));
    }
    if (s.inputs.empty()) s.inputs = "no file input";
    s.result = (l.rows >= 0 ? Thousands(l.rows) + " rows" : std::string("rows unknown")) +
               (l.columns >= 0 ? " \xC2\xB7 " + std::to_string(l.columns) + " columns" : std::string()) + " \xC2\xB7 " +
               Bytes(l.output.bytes) + " \xC2\xB7 sha256 " + ShortDigest(l.output.sha256);
    size_t passed = 0;
    std::string failed;
    for (const auto& c : l.checks) {
        if (c.passed) ++passed;
        else if (failed.empty()) failed = c.name;
    }
    if (l.checks.empty()) {
        s.checks = "no checks";
    } else {
        s.checks = std::to_string(passed) + " of " + std::to_string(l.checks.size()) + " pass";
        if (!failed.empty()) s.checks += ": '" + failed + "' did not";
    }
    s.checks_ok = passed == l.checks.size();
    s.written = l.written_at + (l.graph.empty() ? std::string(" from an unsaved graph") : " from " + fs::path(l.graph).filename().string());
    return s;
}

}  // namespace cyxwiz
