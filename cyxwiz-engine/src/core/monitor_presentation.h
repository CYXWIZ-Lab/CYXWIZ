#pragma once

// Tools > Monitoring (TOFIX129 D13): what the one Monitor window shows of a
// sampler's readings, worked out without ImGui so the window and its tests
// agree. Four tabs: System, Memory, Loaded data, Frames.

#include "monitor_sampler.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::monitor_view {

enum class Tab { System, Memory, LoadedData, Frames };

constexpr Tab kTabs[] = {Tab::System, Tab::Memory, Tab::LoadedData, Tab::Frames};

const char* TabLabel(Tab tab);
// The tab's id in menus and Preferences ("system", "memory", "loaded_data", "frames").
const char* TabId(Tab tab);
// The tab of an id; System for one it does not know.
Tab TabFromId(const std::string& id);

// "0 B", "837 MB", "17.1 GB" (powers of 1024, three significant digits).
std::string SizeText(std::uint64_t bytes);
// "14.3"
std::string PercentText(double percent);
// "15 min", "1 hour", "90 s"
std::string DurationText(double seconds);

// How full a meter is.
enum class Level { Normal, Caution, High };
Level LevelOf(double percent);  // below 70 normal, below 90 caution, else high

// One card of the System tab.
struct Card {
    std::string title;
    std::string subtitle;                                     // right of the title
    std::string big;                                          // the headline figure ("14.3"), empty when rows carry the card
    std::string unit;                                         // after the headline ("% · 30.8 GB free")
    bool has_meter = false;
    double meter_percent = 0.0;
    Level level = Level::Normal;
    std::vector<std::pair<std::string, std::string>> rows;    // label, value
    std::string note;                                         // faint line under the card
};

// The compute device the Engine uses, as the window names it.
struct DeviceLabel {
    bool present = false;
    std::string name;     // "NVIDIA GeForce GTX 1050 Ti"
    std::string backend;  // "CUDA"
    int device_id = 0;
};

inline constexpr const char* kNotAvailable = "Not available";
inline constexpr const char* kNotReported = "Not reported by this driver";

// `sample` may be null (no reading yet): every figure then reads "Waiting for
// the first reading".
Card CpuCard(const MonitorSample* sample);
Card MemoryCard(const MonitorSample* sample);
Card DeviceCard(const MonitorSample* sample, const DeviceLabel& device);
Card ProcessCard(const MonitorSample* sample);

// ---- The series of the trends and of the Memory tab
enum class Series { Cpu, Memory, Process, DeviceAllocated, DeviceInUse, Tables, Tensors };

constexpr Series kMemorySeries[] = {Series::Process, Series::DeviceAllocated, Series::DeviceInUse, Series::Tables,
                                    Series::Tensors};

const char* SeriesLabel(Series series);       // "Engine process"
const char* SeriesMeasuredBy(Series series);  // "the operating system"
// True and the value when the reading has this figure (bytes, or percent for
// Cpu and Memory).
bool SeriesValue(const MonitorSample& sample, Series series, double& value);

// Points for a plot: x in seconds before the newest reading (negative), y the
// value; readings without the figure are left out.
struct Points {
    std::vector<double> x;
    std::vector<double> y;
};
Points SeriesPoints(const std::vector<MonitorSample>& history, Series series);

// One row of the Memory tab's table.
struct MemoryRow {
    Series series = Series::Process;
    std::string what;
    std::string now;      // "1.92 GB" or "Not available"
    std::string lowest;
    std::string highest;
    std::string measured_by;
    bool known = false;
};
std::vector<MemoryRow> MemoryRows(const std::vector<MonitorSample>& history);

// "900 readings over 15 min" / "No readings yet"
std::string HistorySummary(const std::vector<MonitorSample>& history);

// The history as CSV, one line per reading; a figure that was not known is an
// empty cell.
std::string HistoryCsv(const std::vector<MonitorSample>& history);

// ---- Free unused memory
// What the window says after the button: device memory given back and Python
// objects collected (-1: Python was busy and was not asked).
std::string ReleasedText(bool device_ran, std::uint64_t freed_bytes, long long python_objects);

}  // namespace cyxwiz::monitor_view
