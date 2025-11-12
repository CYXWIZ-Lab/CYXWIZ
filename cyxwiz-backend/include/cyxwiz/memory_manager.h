#pragma once

#include "api_export.h"
#include <cstddef>

namespace cyxwiz {

class CYXWIZ_API MemoryManager {
public:
    static void* Allocate(size_t bytes);
    static void Deallocate(void* ptr);
    static size_t GetAllocatedBytes();
    static size_t GetPeakBytes();
    static void ResetPeak();
};

} // namespace cyxwiz
