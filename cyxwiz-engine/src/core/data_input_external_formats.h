#pragma once

// Dataset formats that come from outside the Engine's own table (TOFIX181):
// today the formats plugins add (IDataProvider). The core knows them only
// through this hook, so it does not depend on the plugin system; the Engine
// installs the source at startup (InstallPluginDataFormats).
//
// An external format is named "plugin:<loader id>" in a Data Input's
// file_type, lower case like every format name. Each name gets a stable
// number from 100 up for the Data Input dialog, for the whole session.

#include <algorithm>
#include <cctype>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::data_input {

inline constexpr const char* kExternalFormatPrefix = "plugin:";
inline constexpr int kFirstExternalFormatId = 100;

struct ExternalFormat {
    std::string name;                     // "plugin:<loader id>"
    std::string label;                    // shown to the user
    std::vector<std::string> extensions;  // lower case, without the dot
    std::string version;                  // changes when its reader may read differently
};

struct ExternalFormatSource {
    std::function<std::vector<ExternalFormat>()> list;
    // The format that reads this file, or empty.
    std::function<std::string(const std::string& path)> detect;
};

namespace detail {

struct ExternalFormatState {
    std::mutex mutex;
    ExternalFormatSource source;
    std::deque<std::string> names;  // id - kFirstExternalFormatId -> name; never shrinks
};

inline ExternalFormatState& ExternalState() {
    static ExternalFormatState state;
    return state;
}

}  // namespace detail

inline std::string ExternalFormatName(std::string loader_id) {
    std::transform(loader_id.begin(), loader_id.end(), loader_id.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return kExternalFormatPrefix + loader_id;
}

inline bool IsExternalFormatName(const std::string& name) {
    return name.rfind(kExternalFormatPrefix, 0) == 0;
}

inline void SetExternalFormatSource(ExternalFormatSource source) {
    auto& state = detail::ExternalState();
    std::lock_guard lock(state.mutex);
    state.source = std::move(source);
}

inline std::vector<ExternalFormat> ExternalFormats() {
    auto& state = detail::ExternalState();
    std::function<std::vector<ExternalFormat>()> list;
    {
        std::lock_guard lock(state.mutex);
        list = state.source.list;
    }
    return list ? list() : std::vector<ExternalFormat>{};
}

inline bool FindExternalFormat(const std::string& name, ExternalFormat& out) {
    for (auto& format : ExternalFormats()) {
        if (format.name == name) {
            out = std::move(format);
            return true;
        }
    }
    return false;
}

inline std::string DetectExternalFormat(const std::string& path) {
    auto& state = detail::ExternalState();
    std::function<std::string(const std::string&)> detect;
    {
        std::lock_guard lock(state.mutex);
        detect = state.source.detect;
    }
    return detect ? detect(path) : std::string{};
}

// The dialog's number for an external format name (assigned on first use).
inline int ExternalFormatId(const std::string& name) {
    auto& state = detail::ExternalState();
    std::lock_guard lock(state.mutex);
    const auto it = std::find(state.names.begin(), state.names.end(), name);
    if (it != state.names.end()) return kFirstExternalFormatId + static_cast<int>(it - state.names.begin());
    state.names.push_back(name);
    return kFirstExternalFormatId + static_cast<int>(state.names.size()) - 1;
}

// The name of a dialog number, or nullptr; the pointer stays valid for the session.
inline const char* ExternalFormatNameOf(int id) {
    auto& state = detail::ExternalState();
    std::lock_guard lock(state.mutex);
    const int index = id - kFirstExternalFormatId;
    if (index < 0 || index >= static_cast<int>(state.names.size())) return nullptr;
    return state.names[static_cast<size_t>(index)].c_str();
}

}  // namespace cyxwiz::data_input
