#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#ifdef exception_code
#undef exception_code
#endif
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace cyxwiz {

struct ProcessMemorySnapshot {
    bool detected = false;
    uint64_t resident_bytes = 0;
    uint64_t private_bytes = 0;
    std::string private_metric_name;
    std::string source;
};

inline ProcessMemorySnapshot DetectProcessMemorySnapshot() {
    ProcessMemorySnapshot snapshot;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters))) {
        snapshot.detected = true;
        snapshot.resident_bytes =
            static_cast<uint64_t>(counters.WorkingSetSize);
        snapshot.private_bytes = static_cast<uint64_t>(counters.PrivateUsage);
        snapshot.private_metric_name = "private commit";
        snapshot.source = "Windows GetProcessMemoryInfo";
    }
#elif defined(__APPLE__)
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        snapshot.detected = true;
        snapshot.resident_bytes = static_cast<uint64_t>(info.resident_size);
        snapshot.private_bytes = static_cast<uint64_t>(info.phys_footprint);
        snapshot.private_metric_name = "physical footprint";
        snapshot.source = "macOS task_vm_info";
    }
#elif defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        uint64_t kib = 0;
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream(line.substr(6)) >> kib;
            snapshot.resident_bytes = kib * 1024;
        } else if (line.rfind("RssAnon:", 0) == 0) {
            std::istringstream(line.substr(8)) >> kib;
            snapshot.private_bytes = kib * 1024;
        }
    }
    snapshot.detected = snapshot.resident_bytes > 0;
    if (snapshot.detected) {
        snapshot.private_metric_name = "anonymous resident";
        snapshot.source = "Linux /proc/self/status";
    }
#endif
    return snapshot;
}

} // namespace cyxwiz
