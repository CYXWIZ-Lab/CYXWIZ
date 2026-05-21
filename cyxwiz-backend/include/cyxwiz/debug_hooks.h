#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <utility>

namespace cyxwiz {

class BackendDebugHooks {
public:
    using DebugEventCallback = std::function<void(const std::string& source,
                                                  const std::string& message)>;

    static void SetDebugEventCallback(DebugEventCallback callback) {
        std::lock_guard<std::mutex> lock(Mutex());
        Callback() = std::move(callback);
    }

    static void EmitDebugEvent(const std::string& source, const std::string& message) {
        DebugEventCallback callback;
        {
            std::lock_guard<std::mutex> lock(Mutex());
            callback = Callback();
        }
        if (callback) {
            callback(source, message);
        }
    }

    static bool HasDebugEventCallback() {
        std::lock_guard<std::mutex> lock(Mutex());
        return static_cast<bool>(Callback());
    }

private:
    static std::mutex& Mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static DebugEventCallback& Callback() {
        static DebugEventCallback callback;
        return callback;
    }
};

} // namespace cyxwiz
