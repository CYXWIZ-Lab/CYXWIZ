#include "python_node_approvals.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>

namespace cyxwiz {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
constexpr int kApprovalsVersion = 1;
}

std::string SerializePythonNodeApprovals(const std::vector<PythonNodeApproval>& approvals) {
    json list = json::array();
    for (const auto& a : approvals) {
        list.push_back({{"bundle_id", a.bundle_id}, {"content_hash", a.content_hash}, {"folder", a.folder},
                        {"approved_on", a.approved_on}, {"how", a.how}});
    }
    return json{{"version", kApprovalsVersion}, {"approvals", list}}.dump(2);
}

bool ParsePythonNodeApprovals(const std::string& text, std::vector<PythonNodeApproval>& out, std::string& error) {
    out.clear();
    json root;
    try {
        root = json::parse(text);
    } catch (const json::exception& e) {
        error = std::string("not valid JSON: ") + e.what();
        return false;
    }
    if (!root.is_object() || root.value("version", 0) != kApprovalsVersion) {
        error = "not a version 1 approvals file";
        return false;
    }
    const auto list = root.find("approvals");
    if (list == root.end() || !list->is_array()) {
        error = "'approvals' must be a list";
        return false;
    }
    for (const json& item : *list) {
        if (!item.is_object()) continue;
        PythonNodeApproval a;
        a.bundle_id = item.value("bundle_id", "");
        a.content_hash = item.value("content_hash", "");
        a.folder = item.value("folder", "");
        a.approved_on = item.value("approved_on", "");
        a.how = item.value("how", "");
        if (a.bundle_id.empty() || a.content_hash.rfind("sha256:", 0) != 0) continue;  // unusable entry
        out.push_back(std::move(a));
    }
    return true;
}

fs::path PythonNodeApprovalsPath() {
#ifdef _WIN32
    if (const char* appdata = std::getenv("APPDATA")) return fs::path(appdata) / "cyxwiz" / "node_approvals.json";
#else
    if (const char* home = std::getenv("HOME")) return fs::path(home) / ".cyxwiz" / "node_approvals.json";
#endif
    return "node_approvals.json";
}

PythonNodeApprovals::PythonNodeApprovals(fs::path file) : file_(std::move(file)) {}

PythonNodeApprovals& PythonNodeApprovals::User() {
    static PythonNodeApprovals store(PythonNodeApprovalsPath());
    static const bool loaded = store.Load();
    (void)loaded;
    return store;
}

bool PythonNodeApprovals::Load(std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    approvals_.clear();
    std::ifstream file(file_, std::ios::binary);
    if (!file) return true;
    std::ostringstream text;
    text << file.rdbuf();
    std::string parse_error;
    if (!ParsePythonNodeApprovals(text.str(), approvals_, parse_error)) {
        if (error) *error = file_.string() + ": " + parse_error;
        approvals_.clear();
        return false;
    }
    return true;
}

bool PythonNodeApprovals::IsApproved(const std::string& bundle_id, const std::string& content_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(approvals_.begin(), approvals_.end(), [&](const PythonNodeApproval& a) {
        return a.bundle_id == bundle_id && a.content_hash == content_hash;
    });
}

bool PythonNodeApprovals::Approve(PythonNodeApproval approval, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto same = std::find_if(approvals_.begin(), approvals_.end(), [&](const PythonNodeApproval& a) {
        return a.bundle_id == approval.bundle_id && a.content_hash == approval.content_hash;
    });
    if (same != approvals_.end()) {
        *same = std::move(approval);
    } else {
        approvals_.push_back(std::move(approval));
    }
    return SaveLocked(error);
}

bool PythonNodeApprovals::Revoke(const std::string& bundle_id, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    approvals_.erase(std::remove_if(approvals_.begin(), approvals_.end(),
                                    [&](const PythonNodeApproval& a) { return a.bundle_id == bundle_id; }),
                     approvals_.end());
    return SaveLocked(error);
}

std::vector<PythonNodeApproval> PythonNodeApprovals::All() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return approvals_;
}

bool PythonNodeApprovals::SaveLocked(std::string* error) const {
    std::error_code ec;
    if (file_.has_parent_path()) fs::create_directories(file_.parent_path(), ec);
    const fs::path temp = file_.string() + ".tmp";
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        if (!file) {
            if (error) *error = "cannot write " + temp.string();
            return false;
        }
        file << SerializePythonNodeApprovals(approvals_) << '\n';
        if (!file) {
            if (error) *error = "cannot write " + temp.string();
            return false;
        }
    }
    fs::rename(temp, file_, ec);
    if (ec) {
        if (error) *error = "cannot replace " + file_.string() + ": " + ec.message();
        return false;
    }
    return true;
}

}  // namespace cyxwiz
