// What the machine and the process use, and the sampler that keeps a history.
//
//   test_system_metrics            the checks; exit code 0 when all pass
//   test_system_metrics --print    one reading as JSON, then waits for a line
//                                  on stdin (compare_system_metrics.py reads
//                                  it and asks psutil about this process)

#include "core/monitor_sampler.h"
#include "core/system_metrics.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
#include <process.h>
#define CYXWIZ_GETPID _getpid
#else
#include <unistd.h>
#define CYXWIZ_GETPID getpid
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::cerr << "FAILED: " << what << "\n";
    }
}

bool Near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

using namespace cyxwiz;
using namespace cyxwiz::system_metrics;

void TestArithmetic() {
    Check(!CpuPercent({10, 100}, {10, 100}).has_value(), "no time passed: no CPU figure");
    Check(!CpuPercent({10, 100}, {5, 200}).has_value(), "counters going backwards: no CPU figure");
    const auto quarter = CpuPercent({100, 1000}, {150, 1200});
    Check(quarter && Near(*quarter, 25.0, 1e-12), "50 busy ticks of 200 is 25 %");
    const auto full = CpuPercent({0, 0}, {300, 300});
    Check(full && Near(*full, 100.0, 1e-12), "all busy is 100 %");

    const auto one_core = ProcessCpuPercent(2.0, 3.0, 1.0, 12);
    Check(one_core && Near(*one_core, 100.0 / 12.0, 1e-12), "one busy core of 12 is 8.33 % of the machine");
    Check(!ProcessCpuPercent(2.0, 3.0, 0.0, 12).has_value(), "no wall time: no figure");
    Check(!ProcessCpuPercent(2.0, 3.0, 1.0, 0).has_value(), "unknown cores: no figure");
    const auto capped = ProcessCpuPercent(0.0, 30.0, 1.0, 12);
    Check(capped && Near(*capped, 100.0, 1e-12), "never above 100 %");
}

void TestLinuxText() {
    // /proc/stat of a real 4-core machine.
    const auto stat = ParseProcStat("cpu  4705 150 1120 16250 520 30 45 10 0 0\ncpu0 1393 20 300 4000 100 5 9 2 0 0\n");
    Check(stat.has_value(), "/proc/stat parses");
    if (stat) {
        Check(stat->busy == 4705 + 150 + 1120 + 30 + 45 + 10, "busy = user + nice + system + irq + softirq + steal");
        Check(stat->total == stat->busy + 16250 + 520, "total adds idle and iowait");
    }
    Check(ParseProcStat("cpu  1 2 3 4\n").has_value(), "an old kernel's four fields parse");
    Check(!ParseProcStat("intr 1 2 3\n").has_value(), "not the cpu line: refused");
    Check(!ParseProcStat("").has_value(), "empty: refused");

    const auto mem = ParseMeminfo("MemTotal:       16303428 kB\nMemFree:          210000 kB\nMemAvailable:    8151714 kB\nBuffers: 1 kB\n");
    Check(mem && mem->total_bytes == 16303428ULL * 1024 && mem->available_bytes == 8151714ULL * 1024,
          "/proc/meminfo: MemTotal and MemAvailable in bytes");
    Check(!ParseMeminfo("MemTotal: 100 kB\nMemFree: 50 kB\n").has_value(),
          "no MemAvailable: not known (MemFree is not what programs can take)");

    // A program name with spaces and brackets.
    const auto self = ParseSelfStat(
        "4242 (cyxwiz (engine) x) S 1 4242 4242 0 -1 4194560 1000 0 0 0 730 270 0 0 20 0 78 0 12345 100000 2000 "
        "18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0\n");
    Check(self && self->cpu_ticks == 1000 && self->threads == 78, "/proc/self/stat: utime + stime and num_threads");
    Check(!ParseSelfStat("4242 (short) S 1 2 3\n").has_value(), "a cut line: refused");
    Check(!ParseSelfStat("no bracket").has_value(), "no name: refused");
}

void TestLiveReading() {
    Probe probe;
    const Sample first = probe.Read();
    Check(!first.cpu_known && !first.process_cpu_known, "the first reading has no CPU percentages");
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    Check(first.cores > 0, "cores are known");
    Check(first.memory_known && first.memory_total_bytes > (256ULL << 20), "memory total is known and above 256 MB");
    Check(first.memory_available_bytes <= first.memory_total_bytes, "available memory is within the total");
    Check(first.process_memory_known && first.process_memory_bytes > (1ULL << 20) &&
              first.process_memory_bytes < first.memory_total_bytes,
          "process memory is between 1 MB and the machine's total");
    Check(first.threads_known && first.threads >= 1, "the process has at least one thread");

    // Burn one core for a moment: the process's share must show it.
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
    volatile double sink = 0.0;
    while (std::chrono::steady_clock::now() < until) sink = sink + std::sqrt(sink + 1.0);
    const Sample second = probe.Read();
    Check(second.cpu_known && second.cpu_percent >= 0.0 && second.cpu_percent <= 100.0, "machine CPU is 0..100 %");
    Check(second.process_cpu_known, "process CPU is known on the second reading");
    if (second.process_cpu_known && second.cores > 0) {
        const double one_core = 100.0 / second.cores;
        Check(second.process_cpu_percent > 0.4 * one_core && second.process_cpu_percent <= 100.0,
              "a busy loop shows as about one core: " + std::to_string(second.process_cpu_percent) + " % of " +
                  std::to_string(second.cores) + " cores");
    }

    // Threads are counted.
    std::atomic<bool> release{false};
    std::thread extra([&release] {
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });
    const Sample with_thread = probe.Read();
    release.store(true);
    extra.join();
    Check(with_thread.threads_known && with_thread.threads >= second.threads + 1,
          "a new thread is counted: " + std::to_string(second.threads) + " -> " + std::to_string(with_thread.threads));
#endif
}

void TestSampler() {
    std::atomic<int> reads{0};
    MonitorSampler sampler([&reads](MonitorSample& sample) {
        ++reads;
        sample.tables_known = true;
        sample.tables_bytes = 4242;
    });
    Check(!sampler.running() && sampler.size() == 0, "a new sampler is stopped and empty");
    sampler.Start(0.01, 1.0);  // both below their limits
    Check(Near(sampler.interval(), MonitorSampler::kMinInterval, 1e-12) && Near(sampler.keep(), MonitorSampler::kMinKeep, 1e-12),
          "settings are held to their limits");
    std::this_thread::sleep_for(std::chrono::milliseconds(1400));
    const auto history = sampler.History();
    Check(history.size() >= 3 && history.size() <= 8, "about four readings in 1.4 s at 0.25 s: " + std::to_string(history.size()));
    bool ordered = true, extra = true;
    for (size_t i = 0; i < history.size(); ++i) {
        if (i > 0 && history[i].seconds <= history[i - 1].seconds) ordered = false;
        if (!history[i].tables_known || history[i].tables_bytes != 4242) extra = false;
    }
    Check(ordered, "readings are in time order");
    Check(extra && reads.load() >= static_cast<int>(history.size()), "the Engine's reader fills every reading");
    MonitorSample latest;
    Check(sampler.Latest(latest) && !history.empty() && latest.seconds >= history.back().seconds, "Latest is the newest");
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    Check(latest.system.cpu_known, "a sampled reading has the machine's CPU");
#endif
    const auto generation = sampler.generation();
    sampler.Clear();
    Check(sampler.size() == 0 && sampler.generation() == generation + 1, "Clear empties the history and is seen");
    sampler.Stop();
    Check(!sampler.running(), "Stop ends the thread");
    const size_t after_stop = sampler.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    Check(sampler.size() == after_stop, "nothing is read after Stop");
    sampler.Start(0.25, 60.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    Check(sampler.size() >= 1, "a stopped sampler starts again");
}

int Print() {
    Probe probe;
    probe.Read();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const Sample s = probe.Read();
    std::cout << "{\"pid\": " << CYXWIZ_GETPID() << ", \"cores\": " << s.cores
              << ", \"memory_known\": " << (s.memory_known ? "true" : "false")
              << ", \"memory_total\": " << s.memory_total_bytes << ", \"memory_available\": " << s.memory_available_bytes
              << ", \"process_memory_known\": " << (s.process_memory_known ? "true" : "false")
              << ", \"process_memory\": " << s.process_memory_bytes
              << ", \"threads_known\": " << (s.threads_known ? "true" : "false") << ", \"threads\": " << s.threads
              << ", \"cpu_known\": " << (s.cpu_known ? "true" : "false") << ", \"cpu_percent\": " << s.cpu_percent << "}"
              << std::endl;
    std::string line;
    std::getline(std::cin, line);  // stays alive while the script asks psutil
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--print") return Print();
    TestArithmetic();
    TestLinuxText();
    TestLiveReading();
    TestSampler();
    if (g_failures != 0) {
        std::cerr << g_failures << " of " << g_checks << " checks failed\n";
        return 1;
    }
    std::cout << "system metrics: " << g_checks << " checks passed\n";
    return 0;
}
