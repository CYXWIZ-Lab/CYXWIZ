// What the Monitor window shows of a reading (TOFIX129 D13). The main scenario
// is the dev PC on 2026-11-11: 12 cores, 47.9 GB, a GTX 1050 Ti on CUDA.

#include "core/monitor_presentation.h"

#include <iostream>
#include <string>

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

void Same(const std::string& got, const std::string& want, const std::string& what) {
    Check(got == want, what + ": got \"" + got + "\", want \"" + want + "\"");
}

using namespace cyxwiz;
using namespace cyxwiz::monitor_view;

constexpr std::uint64_t kGB = 1024ULL * 1024 * 1024;
constexpr std::uint64_t kMB = 1024ULL * 1024;
const std::string kDot = " \xC2\xB7 ";

MonitorSample DevPc() {
    MonitorSample s;
    s.seconds = 900.0;
    s.system.cores = 12;
    s.system.cpu_known = true;
    s.system.cpu_percent = 14.34;
    s.system.memory_known = true;
    s.system.memory_total_bytes = 51387342848ULL;
    s.system.memory_available_bytes = 33011183616ULL;
    s.system.process_memory_known = true;
    s.system.process_memory_bytes = 837 * kMB;
    s.system.process_cpu_known = true;
    s.system.process_cpu_percent = 0.94;
    s.system.threads_known = true;
    s.system.threads = 78;
    s.device_held_known = true;
    s.device_allocated_bytes = 0;
    s.device_in_use_bytes = 0;
    s.device_free_known = true;
    s.device_total_bytes = 4294967296ULL;
    s.device_free_bytes = 4143149056ULL;
    s.tables_known = true;
    s.tables_bytes = 0;
    s.tensors_known = true;
    s.tensors_bytes = 0;
    return s;
}

void TestNames() {
    Same(TabLabel(Tab::LoadedData), "Loaded data", "tab label");
    for (const Tab tab : kTabs) Check(TabFromId(TabId(tab)) == tab, std::string("tab id round trip: ") + TabId(tab));
    Check(TabFromId("nonsense") == Tab::System, "an unknown tab id opens System");
}

void TestSizes() {
    Same(SizeText(0), "0 B", "zero");
    Same(SizeText(1023), "1023 B", "below one KB");
    Same(SizeText(1024), "1.00 KB", "one KB");
    Same(SizeText(837 * kMB), "837 MB", "837 MB");
    Same(SizeText(51387342848ULL), "47.9 GB", "the dev PC's memory");
    Same(SizeText(4294967296ULL), "4.00 GB", "the card's memory");
    Same(SizeText(static_cast<std::uint64_t>(1.92 * static_cast<double>(kGB))), "1.92 GB", "two decimals below 10");
    Same(SizeText(1023 * kMB + 900 * 1024), "1.00 GB", "999.5 MB and above reads as the next unit");
    Same(PercentText(14.34), "14.3", "one decimal");
    Same(PercentText(140.0), "100.0", "never above 100");
    Same(DurationText(900), "15 min", "15 minutes");
    Same(DurationText(3600), "1 hour", "one hour");
    Same(DurationText(300), "5 min", "5 minutes");
    Same(DurationText(45), "45 s", "seconds");
    Check(LevelOf(35.0) == Level::Normal && LevelOf(75.0) == Level::Caution && LevelOf(95.0) == Level::High, "meter levels");
}

void TestCards() {
    const MonitorSample pc = DevPc();
    const Card cpu = CpuCard(&pc);
    Same(cpu.subtitle, "12 cores", "CPU subtitle");
    Same(cpu.big + " " + cpu.unit, "14.3 %", "CPU headline");
    Check(cpu.has_meter && cpu.level == Level::Normal, "CPU meter");

    const Card memory = MemoryCard(&pc);
    Same(memory.subtitle, "17.1 GB of 47.9 GB", "memory subtitle");
    Same(memory.big, "35.8", "memory percent");
    Same(memory.unit, "%" + kDot + "30.7 GB free", "memory free");

    DeviceLabel card;
    card.present = true;
    card.name = "NVIDIA GeForce GTX 1050 Ti";
    card.backend = "CUDA";
    card.device_id = 0;
    const Card device = DeviceCard(&pc, card);
    Same(device.subtitle, "NVIDIA GeForce GTX 1050 Ti" + kDot + "CUDA" + kDot + "device 0", "device subtitle");
    Check(device.rows.size() == 3, "device rows: held, device memory, other programs");
    if (device.rows.size() == 3) {
        Same(device.rows[0].second, "0 B allocated" + kDot + "0 B in use", "held by the Engine");
        Same(device.rows[1].second, "3.86 GB free of 4.00 GB", "device memory");
        Same(device.rows[2].second, "145 MB", "other programs = used - held");
    }
    Check(device.has_meter && device.meter_percent > 3.4 && device.meter_percent < 3.6, "device meter is the used share");

    const Card process = ProcessCard(&pc);
    Check(process.rows.size() == 3, "process rows");
    if (process.rows.size() == 3) {
        Same(process.rows[0].second, "837 MB", "process memory");
        Same(process.rows[1].second, "0.9 % of the machine", "process CPU");
        Same(process.rows[2].second, "78", "threads");
    }

    // A driver that does not report free memory: said so, no meter, no guess.
    MonitorSample intel = pc;
    intel.device_free_known = false;
    intel.device_allocated_bytes = 412 * kMB;
    intel.device_in_use_bytes = 96 * kMB;
    const Card unknown = DeviceCard(&intel, card);
    Check(unknown.rows.size() == 2 && !unknown.has_meter, "no free figure: two rows and no meter");
    if (unknown.rows.size() == 2) {
        Same(unknown.rows[0].second, "412 MB allocated" + kDot + "96.0 MB in use", "held is still exact");
        Same(unknown.rows[1].second, kNotReported, "device memory not reported");
    }

    // Held bytes larger than the device's used bytes (two readings a moment apart).
    MonitorSample race = pc;
    race.device_allocated_bytes = 1 * kGB;
    const Card raced = DeviceCard(&race, card);
    Check(raced.rows.size() == 3 && raced.rows[2].second == "0 B", "other programs never goes below zero");

    // No graphics device.
    const Card none = DeviceCard(&pc, DeviceLabel{});
    Same(none.subtitle, "CPU only", "no device");
    Check(none.rows.empty() && !none.has_meter, "no device: no rows");

    // A system that gives nothing (an OS this build does not know).
    MonitorSample blank;
    Same(CpuCard(&blank).big, kNotAvailable, "CPU not available");
    Same(MemoryCard(&blank).big, kNotAvailable, "memory not available");
    const Card blank_process = ProcessCard(&blank);
    Check(blank_process.rows.size() == 3 && blank_process.rows[0].second == kNotAvailable &&
              blank_process.rows[1].second == kNotAvailable && blank_process.rows[2].second == kNotAvailable,
          "process figures not available");
    Check(!CpuCard(&blank).has_meter && !MemoryCard(&blank).has_meter, "no meter without a figure");

    // Before the first reading.
    Same(CpuCard(nullptr).note, "Waiting for the first reading", "no reading yet");
    Check(ProcessCard(nullptr).rows.empty() && DeviceCard(nullptr, card).rows.empty(), "no rows before a reading");
}

void TestHistory() {
    // 15 minutes, one reading a second; training starts at minute 5.
    std::vector<MonitorSample> history;
    for (int i = 0; i <= 900; ++i) {
        MonitorSample s = DevPc();
        s.seconds = i;
        const bool training = i >= 300;
        s.system.process_memory_bytes = training ? static_cast<std::uint64_t>(1.92 * static_cast<double>(kGB)) : 837 * kMB;
        s.device_allocated_bytes = training ? static_cast<std::uint64_t>(2.16 * static_cast<double>(kGB)) : 0;
        s.device_in_use_bytes = training ? static_cast<std::uint64_t>(1.60 * static_cast<double>(kGB)) : 0;
        s.tables_bytes = i >= 200 ? 440 * kMB : 0;
        s.tensors_known = i >= 100;  // not known at first
        s.tensors_bytes = 12 * kMB;
        history.push_back(s);
    }
    const auto rows = MemoryRows(history);
    Check(rows.size() == 5, "five memory rows");
    if (rows.size() == 5) {
        Same(rows[0].what, "Engine process", "row 1");
        Same(rows[0].now + " / " + rows[0].lowest + " / " + rows[0].highest, "1.92 GB / 837 MB / 1.92 GB", "process now / lowest / highest");
        Same(rows[0].measured_by, "the operating system", "who measures the process");
        Same(rows[1].now + " / " + rows[1].lowest, "2.16 GB / 0 B", "device allocated");
        Same(rows[2].now, "1.60 GB", "device in use");
        Same(rows[3].highest, "440 MB", "tables");
        Same(rows[4].lowest, "12.0 MB", "tensors: the readings without the figure are left out");
        Same(rows[1].measured_by, "ArrayFire's memory manager", "who measures the device");
    }
    const Points process = SeriesPoints(history, Series::Process);
    Check(process.x.size() == 901 && process.x.front() == -900.0 && process.x.back() == 0.0, "x runs from -900 s to now");
    const Points tensors = SeriesPoints(history, Series::Tensors);
    Check(tensors.x.size() == 801 && tensors.x.front() == -800.0, "unknown readings are not plotted as zero");
    const Points cpu = SeriesPoints(history, Series::Cpu);
    Check(!cpu.y.empty() && cpu.y.back() == 14.34, "CPU series is in percent");
    Same(HistorySummary(history), "901 readings over 15 min", "history summary");
    Same(HistorySummary({}), "No readings yet", "empty history");
    Check(SeriesPoints({}, Series::Process).x.empty() && MemoryRows({}).size() == 5 && MemoryRows({})[0].now == kNotAvailable,
          "an empty history draws nothing and says so");

    std::vector<MonitorSample> two(history.begin(), history.begin() + 2);
    two[1].system.cpu_known = false;
    two[1].device_free_known = false;
    const std::string csv = HistoryCsv(two);
    const std::string header = csv.substr(0, csv.find('\n'));
    Same(header,
         "seconds,cpu_percent,memory_used_bytes,memory_total_bytes,process_memory_bytes,process_cpu_percent,threads,"
         "device_allocated_bytes,device_in_use_bytes,device_free_bytes,device_total_bytes,tables_bytes,tensors_bytes",
         "CSV header");
    const size_t second = csv.find('\n', csv.find('\n') + 1);
    const std::string line2 = csv.substr(csv.find('\n', csv.find('\n') + 1) + 1, std::string::npos);
    (void)second;
    Same(line2, "1.000,,18376159232,51387342848,877658112,0.94,78,0,0,,,0,\n", "a figure that was not known is an empty cell");
}

void TestReleased() {
    Same(ReleasedText(true, 412 * kMB, 96),
         "Freed 412 MB of device memory the Engine was holding but not using. Python collected 96 objects.", "freed");
    Same(ReleasedText(true, 0, 0),
         "Nothing to free on the device: the Engine holds no memory it is not using. Python had nothing to collect.", "nothing");
    Same(ReleasedText(false, 0, -1),
         "The compute device is not started yet, so it holds nothing to free. Python is busy, so its collection was skipped.",
         "not started, Python busy");
    Same(ReleasedText(true, 1 * kMB, 1), "Freed 1.00 MB of device memory the Engine was holding but not using. Python collected 1 object.",
         "one object");
}

}  // namespace

int main() {
    TestNames();
    TestSizes();
    TestCards();
    TestHistory();
    TestReleased();
    if (g_failures != 0) {
        std::cerr << g_failures << " of " << g_checks << " checks failed\n";
        return 1;
    }
    std::cout << "monitor presentation: " << g_checks << " checks passed\n";
    return 0;
}
