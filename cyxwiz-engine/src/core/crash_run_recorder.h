#pragma once

#include "graph_compiler.h"
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class TrainingTraceStage {
    Start,
    GetNextBatch,
    Forward,
    ComputeLoss,
    Backward,
    UpdateParameters,
    BatchCallback,
    UIPlotUpdate,
    EpochComplete,
    Complete,
    EarlyStopped,
    Failed,
    Cancelled
};

struct CrashRunSummary {
    bool available = false;
    bool suspected_crash = false;
    std::string run_id;
    std::string status;
    std::string dataset_name;
    std::string backend;
    std::string last_stage;
    std::string last_event_time;
    int epoch = 0;
    int last_executed_epoch = 0;
    int batch = 0;
    int total_batches = 0;
    int epochs = 0;
    int batch_size = 0;
    size_t sample_count = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    std::string file_path;
    std::string warning;
    std::string terminal_reason;
    std::string failure_reason;
    std::string checkpoint_used;
    int restored_checkpoint_epoch = 0;
    int restored_checkpoint_step = 0;
    std::string active_model_provenance;
    std::vector<std::string> panel_events;
    bool windows_crash_available = false;
    std::string windows_fault_module;
    std::string windows_exception_code;
    std::string windows_crash_time;
    std::string windows_report_id;
    std::string windows_report_path;
};

class CrashRunRecorder {
public:
    static CrashRunRecorder& Instance();

    void StartTrainingRun(const TrainingConfiguration& config,
                          int epochs,
                          int batch_size,
                          size_t sample_count);
    void UpdateSampleCount(size_t sample_count);
    void UpdateLastExecutedEpoch(int epoch);

    void MarkStage(TrainingTraceStage stage,
                   int epoch,
                   int batch,
                   int total_batches,
                   float loss = 0.0f,
                   float accuracy = 0.0f);
    void MarkPanelEvent(const std::string& action,
                        const std::string& detail = "");
    void MarkBackendEvent(const std::string& source,
                          const std::string& detail = "");
    void MarkActiveModelCheckpoint(const std::string& checkpoint_path,
                                   int checkpoint_epoch,
                                   int checkpoint_step,
                                   const std::string& provenance);

    void MarkCompleted();
    void MarkEarlyStopped(const std::string& reason);
    void MarkCancelled();
    void MarkFailed(const std::string& reason);

    static std::optional<CrashRunSummary> LoadLastRun();
    static const char* StageName(TrainingTraceStage stage);

private:
    CrashRunRecorder() = default;

    void WriteLocked();
    static std::string NowIso8601();
    static std::string ThreadIdString();
    static std::string DomainName(PreprocessingDomain domain);
    static std::string BackendName();

    mutable std::mutex mutex_;
    bool active_ = false;
    std::string run_id_;
    std::string status_;
    std::string dataset_name_;
    std::string backend_;
    std::string last_stage_;
    std::string last_event_time_;
    std::string last_thread_id_;
    std::string failure_reason_;
    std::string terminal_reason_;
    std::string checkpoint_used_;
    int restored_checkpoint_epoch_ = 0;
    int restored_checkpoint_step_ = 0;
    std::string active_model_provenance_;
    int epoch_ = 0;
    int last_executed_epoch_ = 0;
    int batch_ = 0;
    int total_batches_ = 0;
    int epochs_ = 0;
    int batch_size_ = 0;
    size_t sample_count_ = 0;
    float loss_ = 0.0f;
    float accuracy_ = 0.0f;
    std::string domain_;
    std::string file_path_;
    std::vector<std::string> panel_events_;
};

} // namespace cyxwiz
