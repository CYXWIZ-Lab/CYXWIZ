#include "mj_log.h"

#include <mujoco/mujoco.h>

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace cyxwiz::plugin::mujoco {

namespace {

// Kinds of warning remembered; past this every warning is only counted.
constexpr std::size_t kMaxKinds = 256;

struct LogState {
    std::mutex mutex;  // MuJoCo warns from worker threads too
    PluginContext* ctx = nullptr;
    void (*previous_warning)(const char*) = nullptr;
    void (*previous_error)(const char*) = nullptr;
    std::set<std::string> kinds;
    std::size_t repeats = 0;
};

LogState& State() {
    static LogState state;
    return state;
}

// The text with each run of digits (times, indices, values) as one '#'.
std::string KindOf(const std::string& text) {
    std::string kind;
    kind.reserve(text.size());
    for (const char c : text) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            if (kind.empty() || kind.back() != '#') kind.push_back('#');
        } else {
            kind.push_back(c);
        }
    }
    return kind;
}

void OnWarning(const char* message) {
    const std::string text = message ? message : "";
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (!state.ctx) return;
    if (state.kinds.size() >= kMaxKinds || !state.kinds.insert(KindOf(text)).second) {
        ++state.repeats;
        return;
    }
    state.ctx->LogWarn("MuJoCo: " + text);
}

void OnError(const char* message) {
    const std::string text = message ? message : "";
    {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        if (state.ctx) state.ctx->LogError("MuJoCo stopped with an error and the Engine must close: " + text);
    }
    std::exit(EXIT_FAILURE);
}

}  // namespace

void InstallMuJoCoLogHandlers(PluginContext& ctx) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (!state.ctx) {
        state.previous_warning = mju_user_warning;
        state.previous_error = mju_user_error;
        state.kinds.clear();
        state.repeats = 0;
    }
    state.ctx = &ctx;
    mju_user_warning = OnWarning;
    mju_user_error = OnError;
}

void PluginLog(PluginLogLevel level, const std::string& message) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (!state.ctx) {
        std::fprintf(stderr, "%s\n", message.c_str());
        return;
    }
    switch (level) {
        case PluginLogLevel::Info: state.ctx->LogInfo(message); break;
        case PluginLogLevel::Warn: state.ctx->LogWarn(message); break;
        case PluginLogLevel::Error: state.ctx->LogError(message); break;
    }
}

void RemoveMuJoCoLogHandlers() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (!state.ctx) return;
    mju_user_warning = state.previous_warning;
    mju_user_error = state.previous_error;
    if (state.repeats > 0) {
        state.ctx->LogInfo("MuJoCo warnings: " + std::to_string(state.kinds.size()) + " kinds shown, " +
                           std::to_string(state.repeats) + " repeats not shown");
    }
    state.ctx = nullptr;
    state.previous_warning = nullptr;
    state.previous_error = nullptr;
    state.kinds.clear();
    state.repeats = 0;
}

}  // namespace cyxwiz::plugin::mujoco
