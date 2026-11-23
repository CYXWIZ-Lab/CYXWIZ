#include "cyxwiz/device_memory.h"

#include "cyxwiz/cyxwiz.h"
#include "dynamic_library.h"

#include <arrayfire.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <mutex>

namespace cyxwiz {
namespace {

// The last reading and when it was taken, for watcher threads. Readers and
// writers are a few a second, so a mutex is enough.
struct HeldSlot {
    std::mutex mutex;
    DeviceMemoryHeld held;
    std::chrono::steady_clock::time_point taken{};
};

HeldSlot& Slot() {
    static HeldSlot slot;
    return slot;
}

void Remember(const DeviceMemoryHeld& held) {
    auto& slot = Slot();
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.held = held;
    slot.taken = std::chrono::steady_clock::now();
}

// The NVIDIA management library (nvml.dll / libnvidia-ml.so.1), the source
// nvidia-smi reads. It needs no CUDA context. ABI from nvml.h.
struct NvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

class Nvml {
public:
    static Nvml& Instance() {
        static Nvml nvml;
        return nvml;
    }

    bool ready() const { return ready_; }

    // Finds the device by UUID ("" = the only device) and reads its memory.
    bool Read(const std::string& uuid_hex, NvmlMemory& memory) {
        if (!ready_) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        unsigned int count = 0;
        if (get_count_(&count) != 0 || count == 0) return false;
        for (unsigned int index = 0; index < count; ++index) {
            void* device = nullptr;
            if (get_handle_(index, &device) != 0) continue;
            if (uuid_hex.empty()) {
                if (count != 1) return false;
            } else {
                char text[96] = {};
                if (get_uuid_(device, text, sizeof(text)) != 0 || HexOnly(text) != uuid_hex) continue;
            }
            return get_memory_(device, &memory) == 0;
        }
        return false;
    }

private:
    using Init = int (*)();
    using GetCount = int (*)(unsigned int*);
    using GetHandle = int (*)(unsigned int, void**);
    using GetUuid = int (*)(void*, char*, unsigned int);
    using GetMemory = int (*)(void*, NvmlMemory*);

    Nvml()
#ifdef _WIN32
        : library_(L"nvml.dll")
#else
        : library_("libnvidia-ml.so.1")
#endif
    {
        const auto init = detail::FindFunction<Init>(library_, "nvmlInit_v2");
        get_count_ = detail::FindFunction<GetCount>(library_, "nvmlDeviceGetCount_v2");
        get_handle_ = detail::FindFunction<GetHandle>(library_, "nvmlDeviceGetHandleByIndex_v2");
        get_uuid_ = detail::FindFunction<GetUuid>(library_, "nvmlDeviceGetUUID");
        get_memory_ = detail::FindFunction<GetMemory>(library_, "nvmlDeviceGetMemoryInfo");
        // Never shut down: the library stays for the life of the process.
        ready_ = init && get_count_ && get_handle_ && get_uuid_ && get_memory_ && init() == 0;
    }

    // "GPU-8a7b...-..." to its lowercase hex digits.
    static std::string HexOnly(const char* text) {
        std::string hex;
        std::string value(text);
        if (value.rfind("GPU-", 0) == 0) value.erase(0, 4);
        for (const char c : value)
            if (std::isxdigit(static_cast<unsigned char>(c)))
                hex.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        return hex;
    }

    detail::DynamicLibrary library_;
    GetCount get_count_ = nullptr;
    GetHandle get_handle_ = nullptr;
    GetUuid get_uuid_ = nullptr;
    GetMemory get_memory_ = nullptr;
    bool ready_ = false;
    std::mutex mutex_;
};

std::string LowerHex(const std::string& value) {
    std::string hex;
    for (const char c : value)
        if (std::isxdigit(static_cast<unsigned char>(c)))
            hex.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return hex;
}

}  // namespace

DeviceMemoryHeld ReadDeviceMemoryHeld() {
    DeviceMemoryHeld held;
    if (!IsInitialized()) return held;
    try {
        size_t allocated = 0, buffers = 0, in_use = 0, buffers_in_use = 0;
        af::deviceMemInfo(&allocated, &buffers, &in_use, &buffers_in_use);
        held.known = true;
        held.allocated_bytes = allocated;
        held.in_use_bytes = in_use;
        held.buffers = buffers;
        held.buffers_in_use = buffers_in_use;
    } catch (...) {
        // A reading must never disturb the computation it watches.
        return held;
    }
    Remember(held);
    return held;
}

DeviceMemoryHeld LastDeviceMemoryHeld() {
    auto& slot = Slot();
    std::lock_guard<std::mutex> lock(slot.mutex);
    DeviceMemoryHeld held = slot.held;
    if (held.known)
        held.age_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - slot.taken).count();
    return held;
}

DeviceMemoryFree ReadDeviceMemoryFree(const std::string& hardware_uuid) {
    DeviceMemoryFree answer;
    auto& nvml = Nvml::Instance();
    if (!nvml.ready()) {
        answer.source = "Not reported by this driver";
        return answer;
    }
    NvmlMemory memory{};
    if (!nvml.Read(LowerHex(hardware_uuid), memory) || memory.total == 0) {
        answer.source = "Not reported by this driver";
        return answer;
    }
    answer.known = true;
    answer.total_bytes = memory.total;
    answer.free_bytes = std::min(memory.free, memory.total);
    answer.source = "NVIDIA management library";
    return answer;
}

void ReleaseUnusedDeviceMemory() { FreeUnusedDeviceMemory(); }

DeviceMemoryReleased FreeUnusedDeviceMemory() {
    DeviceMemoryReleased released;
    if (!IsInitialized()) return released;
    try {
        released.allocated_before = ReadDeviceMemoryHeld().allocated_bytes;
        af::deviceGC();
        released.allocated_after = ReadDeviceMemoryHeld().allocated_bytes;
        released.ran = true;
    } catch (...) {
        released.ran = false;
    }
    return released;
}

}  // namespace cyxwiz
