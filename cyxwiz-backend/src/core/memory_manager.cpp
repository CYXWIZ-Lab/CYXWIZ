#include "cyxwiz/memory_manager.h"
#include <atomic>
#include <algorithm>

namespace cyxwiz {

static std::atomic<size_t> g_allocated_bytes{0};
static std::atomic<size_t> g_peak_bytes{0};

void* MemoryManager::Allocate(size_t bytes) {
    void* ptr = malloc(bytes);
    if (ptr) {
        g_allocated_bytes += bytes;
        size_t current = g_allocated_bytes.load();
        size_t peak = g_peak_bytes.load();
        while (current > peak && !g_peak_bytes.compare_exchange_weak(peak, current)) {}
    }
    return ptr;
}

void MemoryManager::Deallocate(void* ptr) {
    if (ptr) {
        // Note: We can't track exact deallocation size without metadata
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
