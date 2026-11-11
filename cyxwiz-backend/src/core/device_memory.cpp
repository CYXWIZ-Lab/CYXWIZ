#include "cyxwiz/device_memory.h"

#include "cyxwiz/cyxwiz.h"
#include "dynamic_library.h"

#include <arrayfire.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <mutex>

namespace cyxwiz {
namespace {

// The last reading, for watcher threads. Four counters written together under
// a sequence number, so a reader never mixes two readings.
struct HeldSlot {
    std::atomic<std::uint64_t> sequence{0};  // odd while being written
    std::atomic<bool> known{false};
    std::atomic<std::uint64_t> allocated{0};
    std::atomic<std::uint64_t> in_use{0};
    std::atomic<std::uint64_t> buffers{0};
    std::atomic<std::uint64_t> buffers_in_use{0};
};

HeldSlot& Slot() {
    static HeldSlot slot;
    return slot;
}

std::mutex& SlotWriteMutex() {
    static std::mutex mutex;
    return mutex;
}

void Remember(const DeviceMemoryHeld& held) {
    std::lock_guard<std::mutex> lock(SlotWriteMutex());
    auto& slot = Slot();
    slot.sequence.fetch_add(1, std::memory_order_acq_rel);
    slot.known.store(held.known, std::memory_order_relaxed);
    slot.allocated.store(held.allocated_bytes, std::memory_order_relaxed);
    slot.in_use.store(held.in_use_bytes, std::memory_order_relaxed);
    slot.buffers.store(held.buffers, std::memory_order_relaxed);
    slot.buffers_in_use.store(held.buffers_in_use, std::memory_order_relaxed);
    slot.sequence.fetch_add(1, std::memory_order_acq_rel);
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
    DeviceMemoryHeld held;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const auto before = slot.sequence.load(std::memory_order_acquire);
        held.known = slot.known.load(std::memory_order_relaxed);
        held.allocated_bytes = slot.allocated.load(std::memory_order_relaxed);
        held.in_use_bytes = slot.in_use.load(std::memory_order_relaxed);
        held.buffers = slot.buffers.load(std::memory_order_relaxed);
        held.buffers_in_use = slot.buffers_in_use.load(std::memory_order_relaxed);
        if ((before & 1U) == 0 && slot.sequence.load(std::memory_order_acquire) == before) return held;
    }
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

DeviceMemoryReleased ReleaseUnusedDeviceMemory() {
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
