#pragma once

#include "crash_run_recorder.h"
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct TrainingTraceEvent {
    std::string timestamp;
    std::string run_id;
    std::string stage;
    std::string thread_id;
    int epoch = 0;
    int batch = 0;
    int total_batches = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    float duration_ms = 0.0f;
    uint64_t cpu_allocated_bytes = 0;
    uint64_t cpu_peak_bytes = 0;
    uint64_t af_allocated_bytes = 0;
    uint64_t af_locked_bytes = 0;
    uint64_t af_alloc_buffers = 0;
    uint64_t af_lock_buffers = 0;
    std::string status = "ok";
    std::string message;
};

struct TrainingTraceSummary {
    bool available = false;
    std::string run_id;
    std::string status;
    std::string latest_stage;
    std::string latest_timestamp;
    int latest_epoch = 0;
    int latest_batch = 0;
    int latest_total_batches = 0;
    float latest_loss = 0.0f;
    float latest_accuracy = 0.0f;
    std::vector<TrainingTraceEvent> recent_events;
    std::vector<std::string> warnings;
};

struct TrainingTraceSettings {
    bool persist_enabled = true;
    int persist_every_n_events = 10;
    size_t max_recent_events = 200;
};

class TrainingTraceCollector {
public:
    static TrainingTraceCollector& Instance();

    void StartRun(const std::string& run_id);
    void RecordStage(TrainingTraceStage stage,
                     int epoch,
                     int batch,
                     int total_batches,
                     float loss = 0.0f,
                     float accuracy = 0.0f,
                     float duration_ms = 0.0f,
                     const std::string& status = "ok",
                     const std::string& message = "");
    void RecordRuntimeWarning(const std::string& source,
                              const std::string& message);
    void RecordRuntimeEvent(const std::string& stage,
                            const std::string& message,
                            const std::string& status = "ok");
    void FinishRun(const std::string& status);

    void Configure(const TrainingTraceSettings& settings);
    TrainingTraceSettings GetSettings() const;
    TrainingTraceSummary Snapshot() const;
    static std::optional<TrainingTraceSummary> LoadLastTrace();

private:
    TrainingTraceCollector() = default;

    void WriteLocked() const;
    static std::string NowLocal();
    static std::string ThreadIdString();

    mutable std::mutex mutex_;
    std::string run_id_;
    std::string status_ = "idle";
    std::deque<TrainingTraceEvent> events_;
    std::vector<std::string> warnings_;
    TrainingTraceSettings settings_;
    size_t events_since_write_ = 0;
};

} // namespace cyxwiz
