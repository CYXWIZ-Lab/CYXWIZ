#include "cyxwiz/debug_hooks.h"

#include <atomic>
#include <mutex>
#include <utility>

namespace cyxwiz {

namespace {

std::mutex& HookMutex() {
    static std::mutex mutex;
    return mutex;
}

BackendDebugHooks::DebugEventCallback& HookCallback() {
    static BackendDebugHooks::DebugEventCallback callback;
    return callback;
}

// Checked on every layer call; avoids taking the mutex when nothing listens.
std::atomic<bool> g_has_callback{false};

}  // namespace

void BackendDebugHooks::SetDebugEventCallback(DebugEventCallback callback) {
    std::lock_guard<std::mutex> lock(HookMutex());
    g_has_callback = static_cast<bool>(callback);
    HookCallback() = std::move(callback);
}

void BackendDebugHooks::EmitDebugEvent(const std::string& source, const std::string& message) {
    if (!g_has_callback.load(std::memory_order_relaxed)) return;
    DebugEventCallback callback;
    {
        std::lock_guard<std::mutex> lock(HookMutex());
        callback = HookCallback();
    }
    if (callback) callback(source, message);
}

bool BackendDebugHooks::HasDebugEventCallback() {
    return g_has_callback.load(std::memory_order_relaxed);
}

}  // namespace cyxwiz
