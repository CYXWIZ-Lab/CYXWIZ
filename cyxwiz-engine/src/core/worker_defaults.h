#pragma once

#include <algorithm>
#include <thread>

namespace cyxwiz {

inline int GetDefaultNumWorkers() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 4;
    }

    int workers = static_cast<int>(hw);
    if (workers < 2) workers = 2;
    if (workers > 8) workers = 8;
    return workers;
}

inline int ClampNumWorkersToPlatform(int requested) {
    if (requested <= 0) {
        return 0;
    }
    return std::min(requested, GetDefaultNumWorkers());
}

} // namespace cyxwiz
