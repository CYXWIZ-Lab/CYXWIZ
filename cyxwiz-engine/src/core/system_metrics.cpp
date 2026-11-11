#include "system_metrics.h"

#include "process_memory_snapshot.h"

#include <chrono>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <tlhelp32.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace cyxwiz::system_metrics {

std::optional<double> CpuPercent(const CpuTimes& earlier, const CpuTimes& later) {
    if (later.total <= earlier.total || later.busy < earlier.busy) return std::nullopt;
    const double total = static_cast<double>(later.total - earlier.total);
    const double busy = static_cast<double>(later.busy - earlier.busy);
    const double percent = 100.0 * busy / total;
    return percent < 0.0 ? 0.0 : (percent > 100.0 ? 100.0 : percent);
}

std::optional<double> ProcessCpuPercent(double cpu_seconds_earlier, double cpu_seconds_later,
                                        double wall_seconds, int cores) {
    if (wall_seconds <= 0.0 || cores <= 0 || cpu_seconds_later < cpu_seconds_earlier) return std::nullopt;
    const double percent = 100.0 * (cpu_seconds_later - cpu_seconds_earlier) / (wall_seconds * cores);
    return percent > 100.0 ? 100.0 : percent;
}

std::optional<CpuTimes> ParseProcStat(const std::string& text) {
    std::istringstream in(text);
    std::string label;
    if (!(in >> label) || label != "cpu") return std::nullopt;
    // user nice system idle iowait irq softirq steal (guest times are inside user).
    std::vector<std::uint64_t> fields;
    std::uint64_t value = 0;
    while (fields.size() < 8 && in >> value) fields.push_back(value);
    if (fields.size() < 4) return std::nullopt;
    fields.resize(8, 0);
    CpuTimes times;
    const std::uint64_t idle = fields[3] + fields[4];
    times.busy = fields[0] + fields[1] + fields[2] + fields[5] + fields[6] + fields[7];
    times.total = times.busy + idle;
    return times;
}

std::optional<MemoryTotals> ParseMeminfo(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    MemoryTotals totals;
    bool total_seen = false, available_seen = false;
    while (std::getline(in, line)) {
        std::uint64_t kib = 0;
        if (line.rfind("MemTotal:", 0) == 0) {
            total_seen = static_cast<bool>(std::istringstream(line.substr(9)) >> kib);
            totals.total_bytes = kib * 1024;
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            available_seen = static_cast<bool>(std::istringstream(line.substr(13)) >> kib);
            totals.available_bytes = kib * 1024;
        }
    }
    if (!total_seen || !available_seen || totals.total_bytes == 0) return std::nullopt;
    return totals;
}

std::optional<ProcStat> ParseSelfStat(const std::string& text) {
    const size_t close = text.rfind(')');
    if (close == std::string::npos) return std::nullopt;
    std::istringstream in(text.substr(close + 1));
    // After the name: state(3) ... utime(14) stime(15) ... num_threads(20).
    std::vector<std::string> fields;
    std::string field;
    while (fields.size() < 18 && in >> field) fields.push_back(field);
    if (fields.size() < 18) return std::nullopt;
    try {
        ProcStat stat;
        stat.cpu_ticks = std::stoull(fields[11]) + std::stoull(fields[12]);
        stat.threads = std::stoi(fields[17]);
        return stat;
    } catch (...) {
        return std::nullopt;
    }
}

namespace {

double WallSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#ifdef _WIN32
std::uint64_t FileTime(const FILETIME& t) {
    return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
}
#endif

#if defined(__APPLE__)
// mach_host_self() adds a reference each call: take the port once.
host_t Host() {
    static const host_t host = mach_host_self();
    return host;
}
#endif

#if defined(__linux__)
std::string ReadFile(const char* path) {
    std::ifstream in(path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}
#endif

bool ReadCpuTimes(CpuTimes& times) {
#ifdef _WIN32
    FILETIME idle{}, kernel{}, user{};
    if (!GetSystemTimes(&idle, &kernel, &user)) return false;
    // Kernel time includes idle time.
    times.total = FileTime(kernel) + FileTime(user);
    times.busy = times.total - FileTime(idle);
    return true;
#elif defined(__APPLE__)
    host_cpu_load_info_data_t load{};
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(Host(), HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&load), &count) !=
        KERN_SUCCESS)
        return false;
    times.busy = static_cast<std::uint64_t>(load.cpu_ticks[CPU_STATE_USER]) + load.cpu_ticks[CPU_STATE_SYSTEM] +
                 load.cpu_ticks[CPU_STATE_NICE];
    times.total = times.busy + load.cpu_ticks[CPU_STATE_IDLE];
    return true;
#elif defined(__linux__)
    const auto parsed = ParseProcStat(ReadFile("/proc/stat"));
    if (!parsed) return false;
    times = *parsed;
    return true;
#else
    (void)times;
    return false;
#endif
}

bool ReadMemory(MemoryTotals& totals) {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    totals.total_bytes = status.ullTotalPhys;
    totals.available_bytes = status.ullAvailPhys;
    return true;
#elif defined(__APPLE__)
    std::uint64_t total = 0;
    size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) != 0 || total == 0) return false;
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(Host(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
        KERN_SUCCESS)
        return false;
    vm_size_t page = 0;
    if (host_page_size(Host(), &page) != KERN_SUCCESS) return false;
    // Free and inactive pages are what programs can still take (as psutil
    // counts "available").
    totals.total_bytes = total;
    totals.available_bytes = (static_cast<std::uint64_t>(vm.free_count) + vm.inactive_count) * page;
    if (totals.available_bytes > totals.total_bytes) totals.available_bytes = totals.total_bytes;
    return true;
#elif defined(__linux__)
    const auto parsed = ParseMeminfo(ReadFile("/proc/meminfo"));
    if (!parsed) return false;
    totals = *parsed;
    return true;
#else
    (void)totals;
    return false;
#endif
}

// This process's CPU seconds so far and its thread count.
void ReadProcess(bool& cpu_known, double& cpu_seconds, bool& threads_known, int& threads) {
    cpu_known = threads_known = false;
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        cpu_seconds = static_cast<double>(FileTime(kernel) + FileTime(user)) * 1e-7;
        cpu_known = true;
    }
    // The thread list of the whole system; read on the sampler thread only.
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot != INVALID_HANDLE_VALUE) {
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        const DWORD pid = GetCurrentProcessId();
        int count = 0;
        if (Thread32First(snapshot, &entry)) {
            do {
                if (entry.th32OwnerProcessID == pid) ++count;
            } while (Thread32Next(snapshot, &entry));
            threads = count;
            threads_known = count > 0;
        }
        CloseHandle(snapshot);
    }
#elif defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        cpu_seconds = static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
                      static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) * 1e-6;
        cpu_known = true;
    }
    thread_act_array_t list = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &list, &count) == KERN_SUCCESS) {
        threads = static_cast<int>(count);
        threads_known = true;
        for (mach_msg_type_number_t i = 0; i < count; ++i) mach_port_deallocate(mach_task_self(), list[i]);
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(list), count * sizeof(thread_act_t));
    }
#elif defined(__linux__)
    const auto stat = ParseSelfStat(ReadFile("/proc/self/stat"));
    const long ticks_per_second = sysconf(_SC_CLK_TCK);
    if (stat && ticks_per_second > 0) {
        cpu_seconds = static_cast<double>(stat->cpu_ticks) / static_cast<double>(ticks_per_second);
        cpu_known = true;
        threads = stat->threads;
        threads_known = stat->threads > 0;
    }
#else
    (void)cpu_seconds;
    (void)threads;
#endif
}

}  // namespace

Sample Probe::Read() {
    Sample sample;
    sample.cores = static_cast<int>(std::thread::hardware_concurrency());

    MemoryTotals memory;
    if (ReadMemory(memory)) {
        sample.memory_known = true;
        sample.memory_total_bytes = memory.total_bytes;
        sample.memory_available_bytes = memory.available_bytes;
    }

    const ProcessMemorySnapshot process = DetectProcessMemorySnapshot();
    if (process.detected) {
        sample.process_memory_known = true;
        sample.process_memory_bytes = process.resident_bytes;
    }

    CpuTimes cpu;
    const bool cpu_read = ReadCpuTimes(cpu);
    bool process_cpu_read = false;
    double process_cpu_seconds = 0.0;
    ReadProcess(process_cpu_read, process_cpu_seconds, sample.threads_known, sample.threads);
    const double wall = WallSeconds();

    if (has_previous_cpu_ && cpu_read) {
        if (const auto percent = CpuPercent(previous_cpu_, cpu)) {
            sample.cpu_known = true;
            sample.cpu_percent = *percent;
        }
    }
    if (has_previous_process_ && process_cpu_read) {
        if (const auto percent = ProcessCpuPercent(previous_process_cpu_seconds_, process_cpu_seconds,
                                                   wall - previous_wall_seconds_, sample.cores)) {
            sample.process_cpu_known = true;
            sample.process_cpu_percent = *percent;
        }
    }
    // A reading that failed leaves the next one without a comparison.
    has_previous_cpu_ = cpu_read;
    has_previous_process_ = process_cpu_read;
    previous_cpu_ = cpu;
    previous_process_cpu_seconds_ = process_cpu_seconds;
    previous_wall_seconds_ = wall;
    return sample;
}

}  // namespace cyxwiz::system_metrics
