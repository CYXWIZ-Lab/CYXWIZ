#include "cyxwiz/memory_manager.h"
#include <atomic>
#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace cyxwiz {

static std::atomic<size_t> g_allocated_bytes{0};
static std::atomic<size_t> g_peak_bytes{0};
static std::mutex g_allocation_mutex;
static std::unordered_map<void*, size_t> g_allocation_sizes;

void* MemoryManager::Allocate(size_t bytes) {
    void* ptr = malloc(bytes);
    if (ptr) {
        {
            std::lock_guard<std::mutex> lock(g_allocation_mutex);
            g_allocation_sizes[ptr] = bytes;
        }

        size_t current = g_allocated_bytes.fetch_add(bytes) + bytes;
        size_t peak = g_peak_bytes.load();
        while (current > peak && !g_peak_bytes.compare_exchange_weak(peak, current)) {}
    }
    return ptr;
}

void MemoryManager::Deallocate(void* ptr) {
    if (ptr) {
        size_t bytes = 0;
        {
            std::lock_guard<std::mutex> lock(g_allocation_mutex);
            auto it = g_allocation_sizes.find(ptr);
            if (it != g_allocation_sizes.end()) {
                bytes = it->second;
                g_allocation_sizes.erase(it);
            }
        }

        if (bytes > 0) {
            g_allocated_bytes.fetch_sub(bytes);
        }
        free(ptr);
    }
}

size_t MemoryManager::GetAllocatedBytes() {
    return g_allocated_bytes.load();
}

size_t MemoryManager::GetPeakBytes() {
    return g_peak_bytes.load();
}

void MemoryManager::ResetPeak() {
    g_peak_bytes.store(g_allocated_bytes.load());
}

} // namespace cyxwiz
