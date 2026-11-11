#include "monitor_panel.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_catalog.h"
#include "../../core/engine_config.h"
#include "../../core/execution_device_context.h"
#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/device.h>
#include <cyxwiz/device_memory.h>
#include <cyxwiz/memory_manager.h>

#include <arrow/memory_pool.h>
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace view = monitor_view;
using view::Tab;

namespace {

constexpr double kKeepSeconds[] = {300.0, 900.0, 3600.0};
// Below this width the cards and the plot's legend stack.
constexpr float kTwoColumnWidth = 620.0f;

std::filesystem::path Utf8Path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

int KeepChoice(double seconds) {
    int best = 0;
    for (int i = 0; i < 3; ++i)
        if (std::fabs(kKeepSeconds[i] - seconds) < std::fabs(kKeepSeconds[best] - seconds)) best = i;
    return best;
}

ImVec4 LevelColour(const ui::Tokens& t, view::Level level) {
    switch (level) {
        case view::Level::Normal: return t.success;
        case view::Level::Caution: return t.warning;
        case view::Level::High: return t.error;
    }
    return t.success;
}

// A filled bar: how much of something is used.
void Meter(double percent, const ImVec4& colour) {
    const auto& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 8.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.bg_input), 4.0f);
    const float filled = width * static_cast<float>(std::clamp(percent, 0.0, 100.0) / 100.0);
    if (filled > 1.0f) draw->AddRectFilled(at, ImVec2(at.x + filled, at.y + height), ui::ToU32(colour), 4.0f);
    ImGui::Dummy(ImVec2(width, height));
}

// A small trend under a card: the series over the kept history, from zero
// (or 0..100 for a percentage) to its highest value.
void Trend(const char* id, const view::Points& points, const ImVec4& colour, bool percent, double span_seconds) {
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 34.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::PushID(id);
    ImGui::Dummy(ImVec2(width, height));
    ImGui::PopID();
    if (points.x.size() < 2 || width < 8.0f || span_seconds <= 0.0) return;
    double top = percent ? 100.0 : 0.0;
    if (!percent)
        for (const double y : points.y) top = std::max(top, y);
    if (top <= 0.0) top = 1.0;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    std::vector<ImVec2> line;
    line.reserve(points.x.size());
    for (size_t i = 0; i < points.x.size(); ++i) {
        const double fx = std::clamp(1.0 + points.x[i] / span_seconds, 0.0, 1.0);
        const double fy = std::clamp(points.y[i] / top, 0.0, 1.0);
        line.emplace_back(at.x + static_cast<float>(fx) * width, at.y + height - 1.0f - static_cast<float>(fy) * (height - 3.0f));
    }
    // The area under the line, one quad per step.
    const ImU32 fill = ui::ToU32(ui::WithAlpha(colour, 0.20f));
    const float base = at.y + height;
    for (size_t i = 1; i < line.size(); ++i)
        draw->AddQuadFilled(ImVec2(line[i - 1].x, base), line[i - 1], line[i], ImVec2(line[i].x, base), fill);
    draw->AddPolyline(line.data(), static_cast<int>(line.size()), ui::ToU32(colour), ImDrawFlags_None, 1.5f);
}

void RenderCard(const view::Card& card, const char* id, const view::Points* trend, const ImVec4& trend_colour,
                bool trend_percent, double span_seconds) {
    const auto& t = ui::CurrentTokens();
    ui::BeginCard(id);
    ui::CardHeader(card.title.c_str(), card.subtitle.empty() ? nullptr : card.subtitle.c_str());
    if (!card.big.empty()) {
        {
            ui::FontScope heading(ui::Font::Heading);
            ImGui::TextUnformatted(card.big.c_str());
        }
        if (!card.unit.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", card.unit.c_str());
        }
    }
    if (!card.rows.empty()) MonitorRows("##rows", card.rows);
    if (card.has_meter) Meter(card.meter_percent, LevelColour(t, card.level));
    if (trend) Trend("##trend", *trend, trend_colour, trend_percent, span_seconds);
    if (!card.note.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_faint, "%s", card.note.c_str());
        ImGui::PopTextWrapPos();
    }
    ui::EndCard();
}

const char* DeviceTypeName(DeviceType type) {
    switch (type) {
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::ONEAPI: return "oneAPI";
        case DeviceType::METAL: return "Metal";
        default: return "";
    }
}

}  // namespace

void MonitorRows(const char* id, const std::vector<std::pair<std::string, std::string>>& rows) {
    const auto& t = ui::CurrentTokens();
    float label_width = 0.0f;
    for (const auto& row : rows) label_width = std::max(label_width, ImGui::CalcTextSize(row.first.c_str()).x);
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingFixedFit)) return;
    ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, label_width + t.space_lg);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    for (const auto& [label, value] : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", label.c_str());
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(value.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::EndTable();
}

MonitorPanel::MonitorPanel() : device_for_sampler_(std::make_shared<DeviceForSampler>()) {
    const core::MonitorConfig config = core::EngineConfig::Instance().GetMonitor();
    tab_ = view::TabFromId(config.tab);
    interval_ = std::clamp(config.interval_seconds, MonitorSampler::kMinInterval, MonitorSampler::kMaxInterval);
    keep_choice_ = KeepChoice(config.keep_seconds);

    // The sampler's reader: counters only, no ArrayFire call (its thread owns
    // no device).
    auto device = device_for_sampler_;
    sampler_ = std::make_unique<MonitorSampler>([device](MonitorSample& sample) {
        const DeviceMemoryHeld held = LastDeviceMemoryHeld();
        sample.device_held_known = held.known;
        sample.device_allocated_bytes = held.allocated_bytes;
        sample.device_in_use_bytes = held.in_use_bytes;
        std::string uuid;
        bool present = false;
        {
            std::lock_guard<std::mutex> lock(device->mutex);
            present = device->present;
            uuid = device->hardware_uuid;
        }
        if (present && !uuid.empty()) {
            const DeviceMemoryFree free_memory = ReadDeviceMemoryFree(uuid);
            sample.device_free_known = free_memory.known;
            sample.device_total_bytes = free_memory.total_bytes;
            sample.device_free_bytes = free_memory.free_bytes;
        }
        sample.tables_known = true;
        sample.tables_bytes = static_cast<std::uint64_t>(std::max<int64_t>(0, arrow::default_memory_pool()->bytes_allocated()));
        sample.tensors_known = true;
        sample.tensors_bytes = static_cast<std::uint64_t>(MemoryManager::GetAllocatedBytes());
    });
    sampler_->Start(interval_, kKeepSeconds[keep_choice_]);

    // The loaded-data account is rebuilt only when a dataset changes.
    catalog_subscription_ =
        DatasetCatalog::Instance().Subscribe(alive_, [this](const DatasetChange&) { loaded_stale_ = true; });
}

MonitorPanel::~MonitorPanel() {
    DatasetCatalog::Instance().Unsubscribe(catalog_subscription_);
    sampler_->Stop();
}

void MonitorPanel::Open(const std::string& tab_id) {
    tab_ = view::TabFromId(tab_id);
    select_tab_ = true;
    visible_ = true;
}

// What the Engine holds on the device is read here, on the interface thread
// (ArrayFire's device is per thread), once per sample interval. While a run
// owns the device its own thread reports instead (the training trace).
void MonitorPanel::ReadHeldMemory() {
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last_held_read_).count() < interval_) return;
    last_held_read_ = now;
    if (HasActiveExecutionDeviceContext()) return;
    ReadDeviceMemoryHeld();
    RefreshDevice();
}

// The process device's name and hardware id: asked when the selection
// changes, never on a timer.
void MonitorPanel::RefreshDevice() {
    const auto selected = Device::GetProcessDevice();
    const int type = selected ? static_cast<int>(selected->type) : -1;
    const int id = selected ? selected->device_id : -1;
    if (type == device_type_seen_ && id == device_id_seen_) return;
    device_type_seen_ = type;
    device_id_seen_ = id;
    device_ = {};
    std::string uuid;
    if (selected && selected->type != DeviceType::CPU) {
        try {
            const DeviceInfo info = Device(selected->type, selected->device_id).GetInfo();
            device_.present = true;
            device_.name = info.name;
            std::replace(device_.name.begin(), device_.name.end(), '_', ' ');
            device_.backend = DeviceTypeName(selected->type);
            device_.device_id = selected->device_id;
            if (info.hardware_uuid_known) uuid = info.hardware_uuid;
        } catch (const std::exception& e) {
            spdlog::warn("Monitor: the compute device could not be described ({})", e.what());
        }
    }
    std::lock_guard<std::mutex> lock(device_for_sampler_->mutex);
    device_for_sampler_->present = device_.present;
    device_for_sampler_->hardware_uuid = uuid;
}

void MonitorPanel::RefreshHistory() {
    const std::uint64_t generation = sampler_->generation();
    if (generation == history_generation_) return;
    history_generation_ = generation;
    history_ = sampler_->History();
}

void MonitorPanel::Render() {
    ReadHeldMemory();
    RenderClearCaches();
    // Python's answer to Free unused memory.
    if (released_waiting_ && python_collected_ && python_collected_->load() != -2) {
        released_ = view::ReleasedText(released_device_ran_, released_bytes_, python_collected_->load());
        released_waiting_ = false;
        spdlog::info("Free unused memory: {}", released_);
    }
    if (!visible_) return;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(std::min(980.0f, viewport->WorkSize.x * 0.9f), std::min(680.0f, viewport->WorkSize.y * 0.9f)),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(viewport->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    if (!ImGui::Begin(ICON_FA_GAUGE_HIGH " Monitor###Monitor", &visible_)) {
        ImGui::End();
        return;
    }
    RefreshHistory();
    RenderBar();
    if (ImGui::BeginTabBar("##monitor_tabs")) {
        for (const Tab tab : view::kTabs) {
            const ImGuiTabItemFlags flags = select_tab_ && tab == tab_ ? ImGuiTabItemFlags_SetSelected : 0;
            if (!ImGui::BeginTabItem(view::TabLabel(tab), nullptr, flags)) continue;
            if (!select_tab_) tab_ = tab;
            switch (tab) {
                case Tab::System: RenderSystem(); break;
                case Tab::Memory: RenderMemory(); break;
                case Tab::LoadedData: RenderLoadedData(); break;
                case Tab::Frames: RenderFrames(); break;
            }
            ImGui::EndTabItem();
        }
        select_tab_ = false;
        ImGui::EndTabBar();
    }
    ImGui::End();
}

// One row above the tabs: the sample interval, how long readings are kept,
// the last message, and "Save as my default" at the right.
void MonitorPanel::RenderBar() {
    const auto& t = ui::CurrentTokens();
    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Sample every").x + 150.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Sample every");
    ImGui::SameLine();
    ui::NumberFieldOptions options;
    options.min = MonitorSampler::kMinInterval;
    options.max = MonitorSampler::kMaxInterval;
    options.step = 1.0;
    options.format = "%.2g";
    options.width = 56.0f;
    options.hint = "Seconds between readings";
    if (ui::NumberField("##interval", &interval_, options)) sampler_->SetInterval(interval_);
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "s");

    row.Next(ImGui::CalcTextSize("Keep").x + 110.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Keep");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    const std::string keep_label = view::DurationText(kKeepSeconds[keep_choice_]);
    if (ImGui::BeginCombo("##keep", keep_label.c_str())) {
        for (int i = 0; i < 3; ++i) {
            if (ImGui::Selectable(view::DurationText(kKeepSeconds[i]).c_str(), i == keep_choice_)) {
                keep_choice_ = i;
                sampler_->SetKeep(kKeepSeconds[i]);
            }
        }
        ImGui::EndCombo();
    }

    if (!message_.empty()) {
        row.Next(ImGui::CalcTextSize(message_.c_str()).x + 40.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(message_error_ ? t.error : t.success, "%s", message_.c_str());
        ImGui::SameLine();
        if (ui::InlineLink("OK")) message_.clear();
    }
    const float save_width = ui::ButtonWidth("Save as my default", ui::ButtonSize::Small);
    if (!ui::SameLineRight(save_width)) ImGui::NewLine();
    if (ui::LinkButton("Save as my default")) SaveDefaults();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("This tab, the sample interval and the history length open this way next time\n"
                          "(also in Preferences > General).");
}

void MonitorPanel::SaveDefaults() {
    auto& config = core::EngineConfig::Instance();
    core::MonitorConfig c = config.GetMonitor();
    c.tab = view::TabId(tab_);
    c.interval_seconds = interval_;
    c.keep_seconds = kKeepSeconds[keep_choice_];
    config.SetMonitor(c);
    message_error_ = !config.Save();
    message_ = message_error_ ? "Could not save the Engine settings." : "Saved as your default.";
}

// ---------------------------------------------------------------- System

void MonitorPanel::RenderSystem() {
    const auto& t = ui::CurrentTokens();
    const MonitorSample* latest = history_.empty() ? nullptr : &history_.back();
    const double span = kKeepSeconds[keep_choice_];
    const view::Points cpu = view::SeriesPoints(history_, view::Series::Cpu);
    const view::Points memory = view::SeriesPoints(history_, view::Series::Memory);
    const view::Points process = view::SeriesPoints(history_, view::Series::Process);

    const view::Card cards[] = {view::CpuCard(latest), view::MemoryCard(latest), view::DeviceCard(latest, device_),
                                view::ProcessCard(latest)};
    const view::Points* trends[] = {&cpu, &memory, nullptr, &process};
    const bool trend_percent[] = {true, true, false, false};
    const char* ids[] = {"cpu", "memory", "device", "process"};
    const int columns = ImGui::GetContentRegionAvail().x >= kTwoColumnWidth ? 2 : 1;
    if (ImGui::BeginTable("##cards", columns, ImGuiTableFlags_SizingStretchSame)) {
        for (int i = 0; i < 4; ++i) {
            ImGui::TableNextColumn();
            const ImVec4 colour = i == 3 ? t.series[0] : LevelColour(t, cards[i].level);
            RenderCard(cards[i], ids[i], trends[i], colour, trend_percent[i], span);
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ui::FlowRow row;
    row.Next(ui::ButtonWidth("Free unused memory", ui::ButtonSize::Regular));
    if (ui::SecondaryButton("Free unused memory", !released_waiting_, "Waiting for Python")) FreeUnusedMemory();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Returns the device memory the Engine keeps for reuse to the driver,\nand asks Python to collect its garbage.");
    row.Next(ui::ButtonWidth("Clear caches...", ui::ButtonSize::Regular));
    if (ui::SecondaryButton("Clear caches...")) OpenClearCaches();
    row.Next(ui::ButtonWidth("Open the run profiler", ui::ButtonSize::Small));
    if (ui::LinkButton("Open the run profiler") && open_profiler_) open_profiler_();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Studio Debugger > Runtime: where a training run spends its time.");
    if (!released_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", released_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void MonitorPanel::FreeUnusedMemory() {
    const DeviceMemoryReleased released = cyxwiz::ReleaseUnusedDeviceMemory();
    released_device_ran_ = released.ran;
    released_bytes_ = released.freed_bytes();
    python_collected_ = std::make_shared<std::atomic<long long>>(-2);
    auto answer = python_collected_;
    const bool asked = collect_python_ && collect_python_([answer](long long objects) { answer->store(objects); });
    if (asked) {
        released_waiting_ = true;
        released_ = "Freeing...";
    } else {
        released_waiting_ = false;
        released_ = view::ReleasedText(released_device_ran_, released_bytes_, view::kPythonBusy);
        spdlog::info("Free unused memory: {}", released_);
    }
}

// ---------------------------------------------------------------- Memory

void MonitorPanel::RenderMemory() {
    const auto& t = ui::CurrentTokens();
    // The series to draw.
    ui::FlowRow row;
    int index = 0;
    for (const view::Series series : view::kMemorySeries) {
        const char* label = view::SeriesLabel(series);
        row.Next(ImGui::GetFrameHeight() * 2.0f + ImGui::CalcTextSize(label).x + 16.0f);
        ImGui::PushID(index);
        ImGui::Checkbox("##on", &series_on_[index]);
        ImGui::SameLine(0.0f, 4.0f);
        ImGui::ColorButton("##colour", t.series[index], ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                           ImVec2(10.0f, 10.0f));
        ImGui::SameLine(0.0f, 4.0f);
        ImGui::TextUnformatted(label);
        ImGui::PopID();
        ++index;
    }

    const double span = kKeepSeconds[keep_choice_];
    const float table_height = ImGui::GetTextLineHeightWithSpacing() * 7.0f + ImGui::GetFrameHeightWithSpacing() * 1.5f;
    const float plot_height = std::max(180.0f, ImGui::GetContentRegionAvail().y - table_height);
    if (history_.size() < 2) {
        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        ImGui::TextColored(t.text_dim, "The first readings are being taken.");
    } else if (ImPlot::BeginPlot("##memory", ImVec2(-1, plot_height), ImPlotFlags_NoMenus | ImPlotFlags_NoLegend)) {
        double top = 1.0;
        std::vector<view::Points> drawn(5);
        index = 0;
        for (const view::Series series : view::kMemorySeries) {
            if (series_on_[index]) {
                drawn[static_cast<size_t>(index)] = view::SeriesPoints(history_, series);
                for (double& y : drawn[static_cast<size_t>(index)].y) {
                    y /= (1024.0 * 1024.0 * 1024.0);
                    top = std::max(top, y);
                }
            }
            ++index;
        }
        ImPlot::SetupAxes("minutes ago", "GB", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_X1, -span / 60.0, 0.0, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, top * 1.08, ImPlotCond_Always);
        index = 0;
        for (const view::Series series : view::kMemorySeries) {
            auto& points = drawn[static_cast<size_t>(index)];
            if (series_on_[index] && points.x.size() >= 2) {
                for (double& x : points.x) x /= 60.0;
                ImPlot::SetNextLineStyle(t.series[index], 2.0f);
                ImPlot::PlotLine(view::SeriesLabel(series), points.x.data(), points.y.data(), static_cast<int>(points.x.size()));
            }
            ++index;
        }
        ImPlot::EndPlot();
    }

    const auto rows = view::MemoryRows(history_);
    if (ImGui::BeginTable("##memory_rows", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("What", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("Now", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("Lowest", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("Highest", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("Measured by", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableHeadersRow();
        for (const auto& r : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.what.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(r.known ? t.text : t.text_dim, "%s", r.now.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.lowest.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.highest.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", r.measured_by.c_str());
        }
        ImGui::EndTable();
    }

    ui::FlowRow actions;
    actions.Next(ui::ButtonWidth("Export CSV...", ui::ButtonSize::Regular));
    if (ui::SecondaryButton("Export CSV...", !history_.empty(), "No readings yet")) ExportCsv();
    actions.Next(ui::ButtonWidth("Clear history", ui::ButtonSize::Regular));
    if (ui::SecondaryButton("Clear history", !history_.empty(), "No readings yet")) sampler_->Clear();
    const std::string summary = view::HistorySummary(history_) + " \xC2\xB7 read in the background, also while this window is closed.";
    actions.Next(ImGui::CalcTextSize(summary.c_str()).x);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "%s", summary.c_str());
}

void MonitorPanel::ExportCsv() {
    if (const auto path = FileDialogs::SaveFile("Save the readings as CSV", {{"CSV", "csv"}}, nullptr, "monitor_readings.csv")) {
        std::ofstream out(Utf8Path(*path), std::ios::binary);
        out << view::HistoryCsv(history_);
        message_error_ = !out;
        message_ = out ? "Saved " + *path : "Could not write " + *path;
    }
}

// ---------------------------------------------------------------- Clear caches

void MonitorPanel::OpenClearCaches() {
    caches_open_request_ = true;
    caches_note_.clear();
    caches_selected_.clear();
    MeasureCaches();
}

void MonitorPanel::MeasureCaches() {
    if (caches_measuring_) return;
    caches_measuring_ = true;
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    auto result = std::make_shared<std::vector<StorageCache>>();
    std::weak_ptr<int> alive = alive_;
    AsyncTaskManager::Instance().RunAsync(
        "Measure the caches",
        [root, result](LambdaTask&) { *result = MeasureStorageCaches(root, DataRegistry::Instance().FileInUse()); },
        nullptr,
        [this, alive, result](bool, const std::string&) {
            if (alive.expired()) return;
            caches_ = std::move(*result);
            caches_measuring_ = false;
        },
        alive_);
}

// The same caches as Preferences > Storage, picked together. Opens at any
// time, also before the first run.
void MonitorPanel::RenderClearCaches() {
    static const char* const kTitle = "Clear caches";
    if (caches_open_request_) {
        ImGui::OpenPopup(kTitle);
        caches_open_request_ = false;
    }
    ui::DialogOptions options;
    options.size = ImVec2(640.0f, 380.0f);
    if (!ui::BeginDialog(kTitle, options)) return;
    const auto& t = ui::CurrentTokens();
    const auto selected = [this](const std::string& id) {
        return std::find(caches_selected_.begin(), caches_selected_.end(), id) != caches_selected_.end();
    };
    std::uint64_t total = 0;
    if (caches_measuring_ && caches_.empty()) {
        ImGui::TextColored(t.text_dim, "Measuring the caches...");
    } else if (ImGui::BeginTable("##caches", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Cache", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("What it holds", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableHeadersRow();
        for (const auto& cache : caches_) {
            ImGui::PushID(cache.id.c_str());
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool on = selected(cache.id);
            ImGui::BeginDisabled(cache.bytes == 0 || caches_clearing_);
            if (ImGui::Checkbox(cache.label.c_str(), &on)) {
                if (on) caches_selected_.push_back(cache.id);
                else caches_selected_.erase(std::remove(caches_selected_.begin(), caches_selected_.end(), cache.id), caches_selected_.end());
            }
            ImGui::EndDisabled();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(view::SizeText(cache.bytes).c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            std::string detail = cache.detail;
            if (cache.in_use > 0)
                detail += " " + StorageItemsText(cache.in_use, cache.unit) + " in use by loaded data will be kept.";
            ImGui::TextColored(t.text_dim, "%s", detail.c_str());
            ImGui::PopTextWrapPos();
            ImGui::PopID();
            if (selected(cache.id)) total += cache.bytes;
        }
        ImGui::EndTable();
    }
    if (ProjectManager::Instance().GetProjectRoot().empty())
        ImGui::TextColored(t.text_faint, "No project is open: only this PC's temporary cache is listed.");
    if (!caches_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.success, "%s", caches_note_.c_str());
        ImGui::PopTextWrapPos();
    }
    const std::string label = total > 0 ? "Clear " + view::SizeText(total) : std::string("Clear selected");
    const ui::DialogResult result = ui::EndDialog((label + "###clear").c_str(), "Close", total > 0 && !caches_clearing_,
                                                  caches_clearing_ ? "Clearing..." : "Tick a cache that holds something", true, false);
    if (result != ui::DialogResult::Primary) return;

    caches_clearing_ = true;
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    const std::vector<std::string> ids = caches_selected_;
    auto freed = std::make_shared<StorageClearResult>();
    std::weak_ptr<int> alive = alive_;
    AsyncTaskManager::Instance().RunAsync(
        "Clear caches",
        [ids, root, freed](LambdaTask& task) {
            for (const auto& id : ids) {
                const StorageClearResult one = ClearStorageCache(id, root, DataRegistry::Instance().FileInUse());
                freed->removed += one.removed;
                freed->freed_bytes += one.freed_bytes;
                freed->kept += one.kept;
            }
            task.MarkCompleted(monitor_view::SizeText(freed->freed_bytes) + " freed");
        },
        nullptr,
        [this, alive, freed](bool, const std::string&) {
            if (alive.expired()) return;
            caches_clearing_ = false;
            caches_selected_.clear();
            caches_note_ = "Cleared " + view::SizeText(freed->freed_bytes) +
                           (freed->kept > 0 ? "; " + std::to_string(freed->kept) + " kept (in use)." : ".");
            spdlog::info("Clear caches: {}", caches_note_);
            MeasureCaches();
        },
        alive_);
}

}  // namespace cyxwiz
