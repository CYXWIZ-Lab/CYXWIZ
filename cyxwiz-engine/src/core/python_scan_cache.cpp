#include "python_scan_cache.h"

#include <nlohmann/json.hpp>

#include <filesystem>

namespace cyxwiz::pythonsetup {

std::string ScanCacheToJson(const ScanCache& cache) {
    nlohmann::json j;
    j["version"] = ScanCache::kVersion;
    j["required"] = cache.required;
    j["interpreters"] = nlohmann::json::array();
    for (const auto& e : cache.entries) {
        const Candidate& c = e.candidate;
        j["interpreters"].push_back({{"path", c.path},
                                     {"version", c.version},
                                     {"venv", c.venv},
                                     {"pip", c.pip},
                                     {"usable", c.usable},
                                     {"reason", c.reason},
                                     {"home", c.home},
                                     {"size", e.stamp.size},
                                     {"modified", e.stamp.modified}});
    }
    return j.dump(2);
}

std::optional<ScanCache> ScanCacheFromJson(const std::string& text) {
    const auto j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object() || j.value("version", 0) != ScanCache::kVersion) return std::nullopt;
    if (!j.contains("interpreters") || !j["interpreters"].is_array()) return std::nullopt;
    ScanCache cache;
    cache.required = j.value("required", std::string());
    for (const auto& i : j["interpreters"]) {
        if (!i.is_object()) return std::nullopt;
        ScanCache::Entry e;
        e.candidate.path = i.value("path", std::string());
        e.candidate.version = i.value("version", std::string());
        e.candidate.venv = i.value("venv", false);
        e.candidate.pip = i.value("pip", false);
        e.candidate.usable = i.value("usable", false);
        e.candidate.reason = i.value("reason", std::string());
        e.candidate.home = i.value("home", std::string());
        e.stamp.size = i.value("size", std::uintmax_t{0});
        e.stamp.modified = i.value("modified", 0LL);
        if (e.candidate.path.empty()) return std::nullopt;
        cache.entries.push_back(std::move(e));
    }
    return cache;
}

std::optional<FileStamp> StampOfFile(const std::string& path) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (!std::filesystem::is_regular_file(p, ec)) return std::nullopt;
    FileStamp s;
    s.size = std::filesystem::file_size(p, ec);
    if (ec) return std::nullopt;
    const auto t = std::filesystem::last_write_time(p, ec);
    if (ec) return std::nullopt;
    s.modified = static_cast<long long>(t.time_since_epoch().count());
    return s;
}

std::optional<ScanResult> ResultFromCache(const ScanCache& cache, const std::string& configured,
                                          const std::string& required, const StampOf& stamp_of) {
    if (cache.entries.empty() || cache.required != required) return std::nullopt;
    bool any_usable = false, configured_listed = configured.empty();
    for (const auto& e : cache.entries) {
        const auto now = stamp_of(e.candidate.path);
        if (!now || !(*now == e.stamp)) return std::nullopt;  // gone or changed
        any_usable = any_usable || e.candidate.usable;
        configured_listed = configured_listed || e.candidate.path == configured;
    }
    if (!any_usable || !configured_listed) return std::nullopt;
    ScanResult r;
    r.scanned = true;
    r.from_cache = true;
    r.configured_path = configured;
    r.configured_path_set = !configured.empty();
    // The saved path first, as a full scan lists it.
    for (const auto& e : cache.entries)
        if (r.configured_path_set && e.candidate.path == configured) {
            r.configured_ok = e.candidate.usable;
            r.found.push_back(e.candidate);
        }
    for (const auto& e : cache.entries)
        if (!r.configured_path_set || e.candidate.path != configured) r.found.push_back(e.candidate);
    return r;
}

ScanCache CacheOf(const ScanResult& scan, const std::string& required, const StampOf& stamp_of) {
    ScanCache cache;
    cache.required = required;
    for (const auto& c : scan.found) {
        const auto stamp = stamp_of(c.path);
        if (!stamp) continue;
        cache.entries.push_back({c, *stamp});
    }
    return cache;
}

}  // namespace cyxwiz::pythonsetup
