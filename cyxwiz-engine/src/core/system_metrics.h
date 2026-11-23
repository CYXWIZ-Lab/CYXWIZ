#pragma once

// What the machine and this process are using: CPU, memory, threads. Read
// from the operating system on Windows, Linux and macOS. A figure the system
// did not give has its `_known` flag false; nothing is invented.
//
// No ImGui and no backend calls: the Monitor window, its tests and the
// sampler thread all use this file.

#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz::system_metrics {

// CPU time counters of the whole machine, in the system's own ticks.
struct CpuTimes {
    std::uint64_t busy = 0;   // user + system + interrupts...
    std::uint64_t total = 0;  // busy + idle (+ I/O wait)
};

// 0..100: the busy share of the time between two readings. Empty when no time
// passed or the counters went backwards (a suspended machine).
std::optional<double> CpuPercent(const CpuTimes& earlier, const CpuTimes& later);

// 0..100: this process's CPU seconds as a share of every core over the wall
// time between two readings.
std::optional<double> ProcessCpuPercent(double cpu_seconds_earlier, double cpu_seconds_later,
                                        double wall_seconds, int cores);

// ---- Linux text formats (parsed here so every platform tests them)

// The first line of /proc/stat ("cpu  user nice system idle iowait irq softirq steal ...").
std::optional<CpuTimes> ParseProcStat(const std::string& text);

struct MemoryTotals {
    std::uint64_t total_bytes = 0;
    std::uint64_t available_bytes = 0;  // what programs can still take without swapping
};
// /proc/meminfo: MemTotal and MemAvailable (kB).
std::optional<MemoryTotals> ParseMeminfo(const std::string& text);

struct ProcStat {
    std::uint64_t cpu_ticks = 0;  // utime + stime
    int threads = 0;
};
// /proc/self/stat. The program name may hold spaces and brackets, so fields
// are counted from the last ')'.
std::optional<ProcStat> ParseSelfStat(const std::string& text);

// ---- One reading

struct Sample {
    int cores = 0;  // logical processors; 0 when the system did not say

    bool cpu_known = false;
    double cpu_percent = 0.0;  // whole machine, since the previous reading

    bool memory_known = false;
    std::uint64_t memory_total_bytes = 0;
    std::uint64_t memory_available_bytes = 0;
    std::uint64_t memory_used_bytes() const {
        return memory_total_bytes > memory_available_bytes ? memory_total_bytes - memory_available_bytes : 0;
    }

    bool process_memory_known = false;
    std::uint64_t process_memory_bytes = 0;  // resident (working set)

    bool process_cpu_known = false;
    double process_cpu_percent = 0.0;  // share of the whole machine

    bool threads_known = false;
    int threads = 0;
};

// Reads the system. CPU figures compare with the previous Read of the same
// probe, so the first reading has no CPU percentages.
class Probe {
public:
    Sample Read();

private:
    bool has_previous_cpu_ = false;
    bool has_previous_process_ = false;
    CpuTimes previous_cpu_;
    double previous_process_cpu_seconds_ = 0.0;
    double previous_wall_seconds_ = 0.0;
};

}  // namespace cyxwiz::system_metrics
