#include "plugin_fingerprint.h"

namespace cyxwiz::plugin {

namespace {

std::string Value(const std::vector<std::pair<std::string, std::string>>& pairs, const std::string& key) {
    for (const auto& [k, v] : pairs) {
        if (k == key) return v;
    }
    return {};
}

std::string MajorMinor(const std::string& version) {
    const auto first = version.find('.');
    if (first == std::string::npos) return version;
    const auto second = version.find('.', first + 1);
    return second == std::string::npos ? version : version.substr(0, second);
}

int Part(const std::string& version, int index) {
    size_t start = 0;
    for (int i = 0; i < index; ++i) {
        start = version.find('.', start);
        if (start == std::string::npos) return -1;
        ++start;
    }
    try {
        return std::stoi(version.substr(start));
    } catch (...) {
        return -1;
    }
}

}  // namespace

std::vector<std::pair<std::string, std::string>> ParseBuildFingerprint(const std::string& text) {
    std::vector<std::pair<std::string, std::string>> pairs;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(';', start);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(start, end - start);
        const auto eq = item.find('=');
        if (eq != std::string::npos && eq > 0) pairs.emplace_back(item.substr(0, eq), item.substr(eq + 1));
        start = end + 1;
    }
    return pairs;
}

std::vector<FingerprintDifference> CompareBuildFingerprints(const std::string& engine, const std::string& plugin) {
    const auto e = ParseBuildFingerprint(engine);
    const auto p = ParseBuildFingerprint(plugin);
    std::vector<FingerprintDifference> differences;
    const auto differ = [&](const char* key, const std::string& pv, const std::string& ev) {
        differences.push_back({key, pv, ev});
    };

    const std::string p_sdk = Value(p, "sdk");
    const std::string e_sdk = Value(e, "sdk");
    if (MajorMinor(p_sdk) != MajorMinor(e_sdk)) differ("sdk", p_sdk, e_sdk);

    const std::string p_api = Value(p, "api");
    const std::string e_api = Value(e, "api");
    if (Part(p_api, 0) != Part(e_api, 0) || Part(p_api, 1) < 0 || Part(p_api, 1) > Part(e_api, 1)) {
        differ("api", p_api, e_api);
    }
    for (const char* key : {"compiler", "stdlib", "build", "ptr"}) {
        const std::string pv = Value(p, key);
        const std::string ev = Value(e, key);
        if (pv != ev) differ(key, pv, ev);
    }
    const std::string p_af = Value(p, "arrayfire");
    const std::string e_af = Value(e, "arrayfire");
    if (!p_af.empty() && p_af != "none" && !e_af.empty() && e_af != "none" && p_af != e_af) {
        differ("arrayfire", p_af, e_af);
    }
    return differences;
}

std::string DescribeFingerprintMismatch(const std::vector<FingerprintDifference>& differences) {
    if (differences.empty()) return {};
    std::string plugin;
    std::string engine;
    for (const auto& d : differences) {
        plugin += (plugin.empty() ? "" : ", ") + d.key + "=" + (d.plugin.empty() ? "(none)" : d.plugin);
        engine += (engine.empty() ? "" : ", ") + d.key + "=" + (d.engine.empty() ? "(none)" : d.engine);
    }
    return "The plugin was built with " + plugin + "; this Engine with " + engine +
           ". Rebuild the plugin against the CyxWiz SDK " + std::to_string(CYXWIZ_VERSION_MAJOR) + "." +
           std::to_string(CYXWIZ_VERSION_MINOR) + " with the same settings.";
}

std::string MissingFingerprintMessage(const std::string& engine) {
    return "The plugin has no build fingerprint: it was built with an SDK older than plugin API " +
           std::to_string(kFingerprintApiMajor) + "." + std::to_string(kFingerprintApiMinor) +
           ". Rebuild it against the CyxWiz SDK " + std::to_string(CYXWIZ_VERSION_MAJOR) + "." +
           std::to_string(CYXWIZ_VERSION_MINOR) + " (this Engine: " + engine + ").";
}

}  // namespace cyxwiz::plugin
