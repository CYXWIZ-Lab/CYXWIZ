#include "training_trace_collector.h"

#include <cyxwiz/memory_manager.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

std::filesystem::path TraceDir() {
    return std::filesystem::current_path() / ".cyxwiz" / "debug_runs";
}

std::filesystem::path CurrentTracePath() {
    return TraceDir() / "current_training_trace.json";
}

void PopulateMemorySnapshot(TrainingTraceEvent& event) {
    event.cpu_allocated_bytes = static_cast<uint64_t>(MemoryManager::GetAllocatedBytes());
    event.cpu_peak_bytes = static_cast<uint64_t>(MemoryManager::GetPeakBytes());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        size_t alloc_bytes = 0;
        size_t alloc_buffers = 0;
        size_t lock_bytes = 0;
        size_t lock_buffers = 0;
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
        event.af_allocated_bytes = static_cast<uint64_t>(alloc_bytes);
        event.af_alloc_buffers = static_cast<uint64_t>(alloc_buffers);
        event.af_locked_bytes = static_cast<uint64_t>(lock_bytes);
        event.af_lock_buffers = static_cast<uint64_t>(lock_buffers);
    } catch (...) {
        // Memory tracing must never affect training.
    }
#endif
}

nlohmann::json EventToJson(const TrainingTraceEvent& event) {
    return {
        {"timestamp", event.timestamp},
        {"run_id", event.run_id},
        {"stage", event.stage},
        {"thread_id", event.thread_id},
        {"epoch", event.epoch},
        {"batch", event.batch},
        {"total_batches", event.total_batches},
        {"loss", event.loss},
        {"accuracy", event.accuracy},
        {"duration_ms", event.duration_ms},
        {"cpu_allocated_bytes", event.cpu_allocated_bytes},
        {"cpu_peak_bytes", event.cpu_peak_bytes},
        {"af_allocated_bytes", event.af_allocated_bytes},
        {"af_locked_bytes", event.af_locked_bytes},
        {"af_alloc_buffers", event.af_alloc_buffers},
        {"af_lock_buffers", event.af_lock_buffers},
        {"status", event.status},
        {"message", event.message}
    };
}

TrainingTraceEvent EventFromJson(const nlohmann::json& j) {
    TrainingTraceEvent event;
    event.timestamp = j.value("timestamp", "");
    event.run_id = j.value("run_id", "");
    event.stage = j.value("stage", "");
    event.thread_id = j.value("thread_id", "");
    event.epoch = j.value("epoch", 0);
    event.batch = j.value("batch", 0);
    event.total_batches = j.value("total_batches", 0);
    event.loss = j.value("loss", 0.0f);
    event.accuracy = j.value("accuracy", 0.0f);
    event.duration_ms = j.value("duration_ms", 0.0f);
    event.cpu_allocated_bytes = j.value("cpu_allocated_bytes", uint64_t{0});
    event.cpu_peak_bytes = j.value("cpu_peak_bytes", uint64_t{0});
    event.af_allocated_bytes = j.value("af_allocated_bytes", uint64_t{0});
    event.af_locked_bytes = j.value("af_locked_bytes", uint64_t{0});
    event.af_alloc_buffers = j.value("af_alloc_buffers", uint64_t{0});
    event.af_lock_buffers = j.value("af_lock_buffers", uint64_t{0});
    event.status = j.value("status", "ok");
    event.message = j.value("message", "");
    return event;
}

} // namespace

TrainingTraceCollector& TrainingTraceCollector::Instance() {
    static TrainingTraceCollector collector;
    return collector;
}

void TrainingTraceCollector::StartRun(const std::string& run_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    run_id_ = run_id;
    status_ = "running";
    events_.clear();
    warnings_.clear();
    events_since_write_ = 0;
    if (settings_.persist_enabled) {
        WriteLocked();
    }
}

void TrainingTraceCollector::RecordStage(TrainingTraceStage stage,
                                         int epoch,
                                         int batch,
                                         int total_batches,
                                         float loss,
                                         float accuracy,
                                         float duration_ms,
                                         const std::string& status,
                                         const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = CrashRunRecorder::StageName(stage);
    event.thread_id = ThreadIdString();
    event.epoch = epoch;
    event.batch = batch;
    event.total_batches = total_batches;
    event.loss = loss;
    event.accuracy = accuracy;
    event.duration_ms = duration_ms;
    event.status = status;
    event.message = message;
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (status != "ok" && !message.empty()) {
        warnings_.push_back(message);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    events_since_write_++;
    const int write_interval = std::max(1, settings_.persist_every_n_events);
    if (settings_.persist_enabled &&
        (events_since_write_ >= static_cast<size_t>(write_interval) ||
         status != "ok")) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::RecordRuntimeWarning(const std::string& source,
                                                  const std::string& message) {
    RecordRuntimeEvent(source, message, "warning");
}

void TrainingTraceCollector::RecordRuntimeEvent(const std::string& stage,
                                                const std::string& message,
                                                const std::string& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (run_id_.empty()) {
        return;
    }

    if (status != "ok") {
        std::string warning = stage;
        if (!warning.empty() && !message.empty()) {
            warning += ": ";
        }
        warning += message;
        warnings_.push_back(warning);
        if (warnings_.size() > 50) {
            warnings_.erase(warnings_.begin());
        }
    }

    TrainingTraceEvent event;
    event.timestamp = NowLocal();
    event.run_id = run_id_;
    event.stage = stage.empty() ? "RuntimeEvent" : stage;
    event.thread_id = ThreadIdString();
    event.status = status.empty() ? "ok" : status;
    event.message = message;
    if (!events_.empty()) {
        const auto& latest = events_.back();
        event.epoch = latest.epoch;
        event.batch = latest.batch;
        event.total_batches = latest.total_batches;
        event.loss = latest.loss;
        event.accuracy = latest.accuracy;
    }
    PopulateMemorySnapshot(event);
    events_.push_back(event);
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }

    if (settings_.persist_enabled) {
        WriteLocked();
        events_since_write_ = 0;
    }
}

void TrainingTraceCollector::FinishRun(const std::string& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
    if (settings_.persist_enabled) {
        WriteLocked();
    }
}

void TrainingTraceCollector::Configure(const TrainingTraceSettings& settings) {
    std::lock_guard<std::mutex> lock(mutex_);
    settings_ = settings;
    if (settings_.persist_every_n_events < 1) {
        settings_.persist_every_n_events = 1;
    }
    if (settings_.max_recent_events < 20) {
        settings_.max_recent_events = 20;
    }
    while (events_.size() > settings_.max_recent_events) {
        events_.pop_front();
    }
    if (settings_.persist_enabled) {
        WriteLocked();
    }
}

TrainingTraceSettings TrainingTraceCollector::GetSettings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return settings_;
}

TrainingTraceSummary TrainingTraceCollector::Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    TrainingTraceSummary summary;
    summary.available = !run_id_.empty();
    summary.run_id = run_id_;
    summary.status = status_;
    summary.warnings = warnings_;
    summary.recent_events.assign(events_.begin(), events_.end());
    if (!events_.empty()) {
        const auto& latest = events_.back();
        summary.latest_stage = latest.stage;
        summary.latest_timestamp = latest.timestamp;
        summary.latest_epoch = latest.epoch;
        summary.latest_batch = latest.batch;
        summary.latest_total_batches = latest.total_batches;
        summary.latest_loss = latest.loss;
        summary.latest_accuracy = latest.accuracy;
    }
    return summary;
}

std::optional<TrainingTraceSummary> TrainingTraceCollector::LoadLastTrace() {
    const auto path = CurrentTracePath();
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    try {
        std::ifstream file(path);
        nlohmann::json j;
        file >> j;

        TrainingTraceSummary summary;
        summary.available = true;
        summary.run_id = j.value("run_id", "");
        summary.status = j.value("status", "");
        summary.warnings = j.value("warnings", std::vector<std::string>{});
        if (j.contains("events") && j["events"].is_array()) {
            for (const auto& item : j["events"]) {
                summary.recent_events.push_back(EventFromJson(item));
            }
        }
        if (!summary.recent_events.empty()) {
            const auto& latest = summary.recent_events.back();
            summary.latest_stage = latest.stage;
            summary.latest_timestamp = latest.timestamp;
            summary.latest_epoch = latest.epoch;
            summary.latest_batch = latest.batch;
            summary.latest_total_batches = latest.total_batches;
            summary.latest_loss = latest.loss;
            summary.latest_accuracy = latest.accuracy;
        }
        return summary;
    } catch (...) {
        return std::nullopt;
    }
}

void TrainingTraceCollector::WriteLocked() const {
    try {
        std::filesystem::create_directories(TraceDir());
        nlohmann::json events = nlohmann::json::array();
        for (const auto& event : events_) {
            events.push_back(EventToJson(event));
        }
        nlohmann::json j = {
            {"run_id", run_id_},
            {"status", status_},
            {"events", events},
            {"warnings", warnings_}
        };
        std::ofstream file(CurrentTracePath(), std::ios::trunc);
        file << std::setw(2) << j << '\n';
    } catch (...) {
        // Debug tracing must never break training.
    }
}

std::string TrainingTraceCollector::NowLocal() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

std::string TrainingTraceCollector::ThreadIdString() {
    std::ostringstream out;
    out << std::this_thread::get_id();
    return out.str();
}

} // namespace cyxwiz
