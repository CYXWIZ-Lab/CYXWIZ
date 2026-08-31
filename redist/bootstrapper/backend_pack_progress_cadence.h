#pragma once

#include <chrono>

namespace cyxwiz::runtime {

class BackendPackProgressCadence {
public:
    bool ShouldPublish(bool force = false) {
        const auto now = std::chrono::steady_clock::now();
        if (!published_ || force ||
            now - last_publication_ >= std::chrono::milliseconds(100)) {
            published_ = true;
            last_publication_ = now;
            return true;
        }
        return false;
    }

private:
    bool published_ = false;
    std::chrono::steady_clock::time_point last_publication_{};
};

}  // namespace cyxwiz::runtime
