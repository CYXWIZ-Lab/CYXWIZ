#include "column_role_store.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace fs = std::filesystem;
using nlohmann::json;

ColumnRoleStore::ColumnRoleStore(std::string file) : file_(std::move(file)) {}

std::string ColumnRoleStore::ProjectFile(const std::string& project_root) {
    if (project_root.empty()) return {};
    return (fs::path(project_root) / "datasets" / "column_roles.json").string();
}

bool ColumnRoleStore::Load(std::string* error) {
    datasets_.clear();
    broken_ = false;
    if (file_.empty()) return true;
    std::ifstream in(file_);
    if (!in) return true;  // no file yet
    try {
        const json j = json::parse(in);
        if (j.value("version", 0) != 1) throw std::runtime_error("unknown version");
        for (const auto& d : j.at("datasets")) {
            DatasetUserSettings s;
            s.schema_fingerprint = d.value("schema", std::string());
            if (d.contains("roles"))
                for (const auto& [col, id] : d["roles"].items())
                    if (auto role = RoleFromId(id.get<std::string>())) s.roles[col] = *role;
            if (d.contains("missing_text"))
                for (const auto& [col, texts] : d["missing_text"].items()) s.missing_text[col] = texts.get<std::vector<std::string>>();
            datasets_[d.at("source").get<std::string>()] = std::move(s);
        }
        return true;
    } catch (const std::exception& e) {
        datasets_.clear();
        broken_ = true;
        if (error) *error = file_ + " could not be read (" + e.what() + "); column roles start empty and the file is left as it is.";
        return false;
    }
}

bool ColumnRoleStore::Save(std::string* error) const {
    if (file_.empty()) return true;
    if (broken_) {
        if (error) *error = "not saved: " + file_ + " could not be read, so it is not overwritten";
        return false;
    }
    json j;
    j["version"] = 1;
    j["datasets"] = json::array();
    for (const auto& [source, s] : datasets_) {
        if (s.roles.empty() && s.missing_text.empty()) continue;
        json d;
        d["source"] = source;
        d["schema"] = s.schema_fingerprint;
        json roles = json::object();
        for (const auto& [col, role] : s.roles) roles[col] = RoleId(role);
        d["roles"] = roles;
        json missing = json::object();
        for (const auto& [col, texts] : s.missing_text) missing[col] = texts;
        d["missing_text"] = missing;
        j["datasets"].push_back(d);
    }
    std::error_code ec;
    fs::create_directories(fs::path(file_).parent_path(), ec);
    const std::string tmp = file_ + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            if (error) *error = "could not write " + tmp;
            return false;
        }
        out << j.dump(2) << '\n';
    }
    fs::rename(tmp, file_, ec);
    if (ec) {
        if (error) *error = "could not replace " + file_ + ": " + ec.message();
        return false;
    }
    return true;
}

const DatasetUserSettings* ColumnRoleStore::Find(const std::string& source_key) const {
    auto it = datasets_.find(source_key);
    return it == datasets_.end() ? nullptr : &it->second;
}

void ColumnRoleStore::SetRole(const std::string& source_key, const std::string& column, ColumnRole role) {
    datasets_[source_key].roles[column] = role;
}

void ColumnRoleStore::ClearRole(const std::string& source_key, const std::string& column) {
    auto it = datasets_.find(source_key);
    if (it != datasets_.end()) it->second.roles.erase(column);
}

void ColumnRoleStore::SetMissingText(const std::string& source_key, const std::string& column, std::vector<std::string> texts) {
    if (texts.empty()) {
        auto it = datasets_.find(source_key);
        if (it != datasets_.end()) it->second.missing_text.erase(column);
        return;
    }
    datasets_[source_key].missing_text[column] = std::move(texts);
}

void ColumnRoleStore::SetSchema(const std::string& source_key, const std::string& fingerprint) {
    datasets_[source_key].schema_fingerprint = fingerprint;
}

std::map<std::string, ColumnRole> ColumnRoleStore::RolesFor(const std::string& source_key) const {
    const auto* s = Find(source_key);
    return s ? s->roles : std::map<std::string, ColumnRole>{};
}

}  // namespace cyxwiz
