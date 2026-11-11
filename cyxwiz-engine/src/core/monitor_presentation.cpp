#include "monitor_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace cyxwiz::monitor_view {

namespace {

constexpr const char* kWaiting = "Waiting for the first reading";

std::string Fixed(double value, int decimals) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

std::string Count(long long n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + digits : digits;
}

}  // namespace

const char* TabLabel(Tab tab) {
    switch (tab) {
        case Tab::System: return "System";
        case Tab::Memory: return "Memory";
        case Tab::LoadedData: return "Loaded data";
        case Tab::Frames: return "Frames";
    }
    return "System";
}

const char* TabId(Tab tab) {
    switch (tab) {
        case Tab::System: return "system";
        case Tab::Memory: return "memory";
        case Tab::LoadedData: return "loaded_data";
        case Tab::Frames: return "frames";
    }
    return "system";
}

Tab TabFromId(const std::string& id) {
    for (const Tab tab : kTabs)
        if (id == TabId(tab)) return tab;
    return Tab::System;
}

std::string SizeText(std::uint64_t bytes) {
    if (bytes < 1024) return std::to_string(bytes) + " B";
    static const char* const units[] = {"KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes) / 1024.0;
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    // Three significant digits: 1.92 GB, 17.1 GB, 837 MB. 999.6 MB rounds to
    // 1000, which reads better as the next unit.
    if (value >= 999.5 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    const int decimals = value >= 99.95 ? 0 : (value >= 9.995 ? 1 : 2);
    return Fixed(value, decimals) + " " + units[unit];
}

std::string PercentText(double percent) { return Fixed(std::clamp(percent, 0.0, 100.0), 1); }

std::string DurationText(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    const long long whole = std::llround(seconds);
    if (whole >= 3600 && whole % 3600 == 0) return std::to_string(whole / 3600) + (whole == 3600 ? " hour" : " hours");
    if (whole >= 120 || (whole >= 60 && whole % 60 == 0)) return std::to_string(std::llround(seconds / 60.0)) + " min";
    return std::to_string(whole) + " s";
}

Level LevelOf(double percent) { return percent < 70.0 ? Level::Normal : (percent < 90.0 ? Level::Caution : Level::High); }

Card CpuCard(const MonitorSample* sample) {
    Card card;
    card.title = "CPU";
    if (!sample) {
        card.note = kWaiting;
        return card;
    }
    const auto& s = sample->system;
    if (s.cores > 0) card.subtitle = std::to_string(s.cores) + (s.cores == 1 ? " core" : " cores");
    if (!s.cpu_known) {
        card.big = kNotAvailable;
        return card;
    }
    card.big = PercentText(s.cpu_percent);
    card.unit = "%";
    card.has_meter = true;
    card.meter_percent = s.cpu_percent;
    card.level = LevelOf(s.cpu_percent);
    return card;
}

Card MemoryCard(const MonitorSample* sample) {
    Card card;
    card.title = "Memory";
    if (!sample) {
        card.note = kWaiting;
        return card;
    }
    const auto& s = sample->system;
    if (!s.memory_known || s.memory_total_bytes == 0) {
        card.big = kNotAvailable;
        return card;
    }
    const double percent = 100.0 * static_cast<double>(s.memory_used_bytes()) / static_cast<double>(s.memory_total_bytes);
    card.subtitle = SizeText(s.memory_used_bytes()) + " of " + SizeText(s.memory_total_bytes);
    card.big = PercentText(percent);
    card.unit = "% \xC2\xB7 " + SizeText(s.memory_available_bytes) + " free";
    card.has_meter = true;
    card.meter_percent = percent;
    card.level = LevelOf(percent);
    return card;
}

Card DeviceCard(const MonitorSample* sample, const DeviceLabel& device) {
    Card card;
    card.title = "Compute device";
    if (!device.present) {
        card.subtitle = "CPU only";
        card.note = "No graphics device is in use; training runs on the processor.";
        return card;
    }
    card.subtitle = device.name;
    if (!device.backend.empty()) card.subtitle += " \xC2\xB7 " + device.backend;
    card.subtitle += " \xC2\xB7 device " + std::to_string(device.device_id);
    if (!sample) {
        card.note = kWaiting;
        return card;
    }
    card.rows.emplace_back("Held by the Engine",
                           sample->device_held_known
                               ? SizeText(sample->device_allocated_bytes) + " allocated \xC2\xB7 " +
                                     SizeText(sample->device_in_use_bytes) + " in use"
                               : std::string("Not read yet"));
    if (sample->device_free_known && sample->device_total_bytes > 0) {
        const std::uint64_t total = sample->device_total_bytes;
        const std::uint64_t free_bytes = std::min(sample->device_free_bytes, total);
        card.rows.emplace_back("Device memory", SizeText(free_bytes) + " free of " + SizeText(total));
        if (sample->device_held_known) {
            const std::uint64_t used = total - free_bytes;
            const std::uint64_t others = used > sample->device_allocated_bytes ? used - sample->device_allocated_bytes : 0;
            card.rows.emplace_back("Other programs", SizeText(others));
        }
        card.has_meter = true;
        card.meter_percent = 100.0 * static_cast<double>(total - free_bytes) / static_cast<double>(total);
        card.level = LevelOf(card.meter_percent);
    } else {
        card.rows.emplace_back("Device memory", kNotReported);
        card.note = "This driver does not say how much device memory is free. What the Engine holds is still exact.";
    }
    return card;
}

Card ProcessCard(const MonitorSample* sample) {
    Card card;
    card.title = "This Engine";
    card.subtitle = "process";
    if (!sample) {
        card.note = kWaiting;
        return card;
    }
    const auto& s = sample->system;
    card.rows.emplace_back("Memory", s.process_memory_known ? SizeText(s.process_memory_bytes) : std::string(kNotAvailable));
    card.rows.emplace_back("CPU", s.process_cpu_known ? PercentText(s.process_cpu_percent) + " % of the machine"
                                                      : std::string(kNotAvailable));
    card.rows.emplace_back("Threads", s.threads_known ? std::to_string(s.threads) : std::string(kNotAvailable));
    return card;
}

const char* SeriesLabel(Series series) {
    switch (series) {
        case Series::Cpu: return "CPU";
        case Series::Memory: return "Memory";
        case Series::Process: return "Engine process";
        case Series::DeviceAllocated: return "Device, allocated";
        case Series::DeviceInUse: return "Device, in use";
        case Series::Tables: return "Tables (Arrow)";
        case Series::Tensors: return "Tensors on the host";
    }
    return "";
}

const char* SeriesMeasuredBy(Series series) {
    switch (series) {
        case Series::Cpu:
        case Series::Memory:
        case Series::Process: return "the operating system";
        case Series::DeviceAllocated:
        case Series::DeviceInUse: return "ArrayFire's memory manager";
        case Series::Tables: return "Arrow's memory pool";
        case Series::Tensors: return "the tensor allocator";
    }
    return "";
}

bool SeriesValue(const MonitorSample& sample, Series series, double& value) {
    const auto& s = sample.system;
    switch (series) {
        case Series::Cpu:
            value = s.cpu_percent;
            return s.cpu_known;
        case Series::Memory:
            if (!s.memory_known || s.memory_total_bytes == 0) return false;
            value = 100.0 * static_cast<double>(s.memory_used_bytes()) / static_cast<double>(s.memory_total_bytes);
            return true;
        case Series::Process:
            value = static_cast<double>(s.process_memory_bytes);
            return s.process_memory_known;
        case Series::DeviceAllocated:
            value = static_cast<double>(sample.device_allocated_bytes);
            return sample.device_held_known;
        case Series::DeviceInUse:
            value = static_cast<double>(sample.device_in_use_bytes);
            return sample.device_held_known;
        case Series::Tables:
            value = static_cast<double>(sample.tables_bytes);
            return sample.tables_known;
        case Series::Tensors:
            value = static_cast<double>(sample.tensors_bytes);
            return sample.tensors_known;
    }
    return false;
}

Points SeriesPoints(const std::vector<MonitorSample>& history, Series series) {
    Points points;
    if (history.empty()) return points;
    const double newest = history.back().seconds;
    points.x.reserve(history.size());
    points.y.reserve(history.size());
    for (const auto& sample : history) {
        double value = 0.0;
        if (!SeriesValue(sample, series, value)) continue;
        points.x.push_back(sample.seconds - newest);
        points.y.push_back(value);
    }
    return points;
}

std::vector<MemoryRow> MemoryRows(const std::vector<MonitorSample>& history) {
    std::vector<MemoryRow> rows;
    for (const Series series : kMemorySeries) {
        MemoryRow row;
        row.series = series;
        row.what = SeriesLabel(series);
        row.measured_by = SeriesMeasuredBy(series);
        double lowest = 0.0, highest = 0.0, now = 0.0;
        bool any = false, now_known = false;
        for (const auto& sample : history) {
            double value = 0.0;
            const bool known = SeriesValue(sample, series, value);
            if (&sample == &history.back()) {
                now_known = known;
                now = value;
            }
            if (!known) continue;
            lowest = any ? std::min(lowest, value) : value;
            highest = any ? std::max(highest, value) : value;
            any = true;
        }
        row.known = any;
        row.now = now_known ? SizeText(static_cast<std::uint64_t>(now)) : std::string(kNotAvailable);
        row.lowest = any ? SizeText(static_cast<std::uint64_t>(lowest)) : std::string("-");
        row.highest = any ? SizeText(static_cast<std::uint64_t>(highest)) : std::string("-");
        rows.push_back(std::move(row));
    }
    return rows;
}

std::string HistorySummary(const std::vector<MonitorSample>& history) {
    if (history.empty()) return "No readings yet";
    const double span = history.back().seconds - history.front().seconds;
    return Count(static_cast<long long>(history.size())) + (history.size() == 1 ? " reading" : " readings") +
           (history.size() > 1 ? " over " + DurationText(span) : std::string());
}

std::string HistoryCsv(const std::vector<MonitorSample>& history) {
    std::ostringstream out;
    out << "seconds,cpu_percent,memory_used_bytes,memory_total_bytes,process_memory_bytes,process_cpu_percent,threads,"
           "device_allocated_bytes,device_in_use_bytes,device_free_bytes,device_total_bytes,tables_bytes,tensors_bytes\n";
    const auto cell = [&out](bool known, const std::string& text, bool last = false) {
        if (known) out << text;
        out << (last ? "\n" : ",");
    };
    for (const auto& sample : history) {
        const auto& s = sample.system;
        out << Fixed(sample.seconds, 3) << ",";
        cell(s.cpu_known, Fixed(s.cpu_percent, 2));
        cell(s.memory_known, std::to_string(s.memory_used_bytes()));
        cell(s.memory_known, std::to_string(s.memory_total_bytes));
        cell(s.process_memory_known, std::to_string(s.process_memory_bytes));
        cell(s.process_cpu_known, Fixed(s.process_cpu_percent, 2));
        cell(s.threads_known, std::to_string(s.threads));
        cell(sample.device_held_known, std::to_string(sample.device_allocated_bytes));
        cell(sample.device_held_known, std::to_string(sample.device_in_use_bytes));
        cell(sample.device_free_known, std::to_string(sample.device_free_bytes));
        cell(sample.device_free_known, std::to_string(sample.device_total_bytes));
        cell(sample.tables_known, std::to_string(sample.tables_bytes));
        cell(sample.tensors_known, std::to_string(sample.tensors_bytes), true);
    }
    return out.str();
}

std::string ReleasedText(bool device_ran, std::uint64_t freed_bytes, long long python_objects) {
    std::string text;
    if (!device_ran) text = "The compute device is not started yet, so it holds nothing to free.";
    else if (freed_bytes == 0) text = "Nothing to free on the device: the Engine holds no memory it is not using.";
    else text = "Freed " + SizeText(freed_bytes) + " of device memory the Engine was holding but not using.";
    if (python_objects == kPythonNotStarted) text += " Python has not been started, so it holds nothing.";
    else if (python_objects < 0) text += " Python is busy, so its collection was skipped.";
    else if (python_objects == 0) text += " Python had nothing to collect.";
    else text += " Python collected " + Count(python_objects) + (python_objects == 1 ? " object." : " objects.");
    return text;
}

}  // namespace cyxwiz::monitor_view
