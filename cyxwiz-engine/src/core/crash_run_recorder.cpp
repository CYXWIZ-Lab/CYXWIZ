#include "crash_run_recorder.h"
#include "debug_run_paths.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <arrayfire.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

namespace cyxwiz {

namespace {

std::filesystem::path DebugRunDir() {
    return GetDebugRunRoot();
}

std::filesystem::path CurrentRunPath() {
    return DebugRunDir() / "current_run.json";
}

#ifdef _WIN32
std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string EnvPath(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string{};
}

std::vector<std::filesystem::path> WerRoots() {
    std::vector<std::filesystem::path> roots;
    const std::string program_data = EnvPath("ProgramData");
    if (!program_data.empty()) {
        roots.push_back(std::filesystem::path(program_data) / "Microsoft" / "Windows" / "WER" / "ReportArchive");
        roots.push_back(std::filesystem::path(program_data) / "Microsoft" / "Windows" / "WER" / "ReportQueue");
    }
    const std::string local_app_data = EnvPath("LOCALAPPDATA");
    if (!local_app_data.empty()) {
        roots.push_back(std::filesystem::path(local_app_data) / "Microsoft" / "Windows" / "WER" / "ReportArchive");
        roots.push_back(std::filesystem::path(local_app_data) / "Microsoft" / "Windows" / "WER" / "ReportQueue");
    }
    return roots;
}

std::map<std::string, std::string> ParseWerFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::map<std::string, std::string> values;
    std::string line;
    std::string pending_sig_name;
    while (std::getline(file, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        values[key] = value;

        if (key.find("Sig[") == 0 && key.find("].Name") != std::string::npos) {
            pending_sig_name = value;
        } else if (!pending_sig_name.empty() &&
                   key.find("Sig[") == 0 &&
                   key.find("].Value") != std::string::npos) {
            values[pending_sig_name] = value;
            pending_sig_name.clear();
        }
    }
    return values;
}
#endif

void AttachLatestWerReport(CrashRunSummary& summary) {
#ifdef _WIN32
    const std::string target = "cyxwiz-engine.exe";
    std::filesystem::path best_path;
    std::filesystem::file_time_type best_time{};
    bool found = false;

    for (const auto& root : WerRoots()) {
        if (!std::filesystem::exists(root)) {
            continue;
        }

        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(
            root,
            std::filesystem::directory_options::skip_permission_denied,
            ec);
        const std::filesystem::recursive_directory_iterator end;
        for (; !ec && it != end; it.increment(ec)) {
            const auto& entry = *it;
            const std::string path_lower = ToLower(entry.path().string());
            if (path_lower.find("cyxwiz") == std::string::npos) {
                continue;
            }
            if (!entry.is_regular_file(ec) ||
                ToLower(entry.path().extension().string()) != ".wer") {
                continue;
            }

            std::ifstream probe(entry.path());
            const std::string content((std::istreambuf_iterator<char>(probe)),
                                      std::istreambuf_iterator<char>());
            if (ToLower(content).find(target) == std::string::npos) {
                continue;
            }

            const auto write_time = entry.last_write_time(ec);
            if (!found || (!ec && write_time > best_time)) {
                best_path = entry.path();
                best_time = write_time;
                found = true;
            }
        }
    }

    if (!found) {
        return;
    }

    const auto values = ParseWerFile(best_path);
    auto get = [&values](const char* key) -> std::string {
        auto it = values.find(key);
        return it == values.end() ? std::string{} : it->second;
    };

    summary.windows_crash_available = true;
    summary.windows_fault_module = get("Fault Module Name");
    if (summary.windows_fault_module.empty()) {
        summary.windows_fault_module = get("Fault Module");
    }
    summary.windows_exception_code = get("Exception Code");
    summary.windows_crash_time = get("EventTime");
    summary.windows_report_id = get("ReportIdentifier");
    summary.windows_report_path = best_path.string();
#else
    (void)summary;
#endif
}

} // namespace

CrashRunRecorder& CrashRunRecorder::Instance() {
    static CrashRunRecorder recorder;
    return recorder;
}

const char* CrashRunRecorder::StageName(TrainingTraceStage stage) {
    switch (stage) {
        case TrainingTraceStage::Start: return "Start";
        case TrainingTraceStage::GetNextBatch: return "GetNextBatch";
        case TrainingTraceStage::Forward: return "Forward";
        case TrainingTraceStage::ComputeLoss: return "ComputeLoss";
        case TrainingTraceStage::Backward: return "Backward";
        case TrainingTraceStage::UpdateParameters: return "UpdateParameters";
        case TrainingTraceStage::BatchCallback: return "BatchCallback";
        case TrainingTraceStage::UIPlotUpdate: return "UIPlotUpdate";
        case TrainingTraceStage::EpochComplete: return "EpochComplete";
        case TrainingTraceStage::Complete: return "Complete";
        case TrainingTraceStage::EarlyStopped: return "EarlyStopped";
        case TrainingTraceStage::Failed: return "Failed";
        case TrainingTraceStage::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

void CrashRunRecorder::StartTrainingRun(const TrainingConfiguration& config,
                                        int epochs,
                                        int batch_size,
                                        size_t sample_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    run_id_ = "train-" + std::to_string(millis);
    status_ = "running";
    dataset_name_ = config.dataset_name;
    backend_ = BackendName();
    last_stage_ = StageName(TrainingTraceStage::Start);
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    failure_reason_.clear();
    terminal_reason_.clear();
    checkpoint_used_.clear();
    restored_checkpoint_epoch_ = 0;
    restored_checkpoint_step_ = 0;
    active_model_provenance_.clear();
    epoch_ = 0;
    last_executed_epoch_ = 0;
    batch_ = 0;
    total_batches_ = 0;
    epochs_ = epochs;
    batch_size_ = batch_size;
    sample_count_ = sample_count;
    loss_ = 0.0f;
    accuracy_ = 0.0f;
    domain_ = DomainName(config.preprocessing_domain);
    file_path_ = CurrentRunPath().string();
    panel_events_.clear();
    active_ = true;

    WriteLocked();
}

void CrashRunRecorder::UpdateSampleCount(size_t sample_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    sample_count_ = sample_count;
    WriteLocked();
}

void CrashRunRecorder::MarkStage(TrainingTraceStage stage,
                                 int epoch,
                                 int batch,
                                 int total_batches,
                                 float loss,
                                 float accuracy) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }

    last_stage_ = StageName(stage);
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    epoch_ = epoch;
    batch_ = batch;
    total_batches_ = total_batches;
    loss_ = loss;
    accuracy_ = accuracy;

    WriteLocked();
}

void CrashRunRecorder::MarkPanelEvent(const std::string& action,
                                      const std::string& detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }

    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    std::string event = last_event_time_ + " [" + last_thread_id_ + "] " + action;
    if (!detail.empty()) {
        event += ": " + detail;
    }
    panel_events_.push_back(std::move(event));
    constexpr size_t kMaxPanelEvents = 80;
    if (panel_events_.size() > kMaxPanelEvents) {
        panel_events_.erase(panel_events_.begin(),
                            panel_events_.begin() + static_cast<std::ptrdiff_t>(
                                panel_events_.size() - kMaxPanelEvents));
    }
    WriteLocked();
}

void CrashRunRecorder::MarkBackendEvent(const std::string& source,
                                        const std::string& detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }

    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    std::string event = last_event_time_ + " [" + last_thread_id_ + "] " + source;
    if (!detail.empty()) {
        event += ": " + detail;
    }
    panel_events_.push_back(std::move(event));
    constexpr size_t kMaxPanelEvents = 80;
    if (panel_events_.size() > kMaxPanelEvents) {
        panel_events_.erase(panel_events_.begin(),
                            panel_events_.begin() + static_cast<std::ptrdiff_t>(
                                panel_events_.size() - kMaxPanelEvents));
    }
    WriteLocked();
}

void CrashRunRecorder::UpdateLastExecutedEpoch(int epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    last_executed_epoch_ = std::max(0, epoch);
    WriteLocked();
}

void CrashRunRecorder::MarkActiveModelCheckpoint(
    const std::string& checkpoint_path,
    int checkpoint_epoch,
    int checkpoint_step,
    const std::string& provenance) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    checkpoint_used_ = checkpoint_path;
    restored_checkpoint_epoch_ = checkpoint_epoch;
    restored_checkpoint_step_ = checkpoint_step;
    active_model_provenance_ = provenance;
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    WriteLocked();
}

void CrashRunRecorder::MarkCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    status_ = "completed";
    terminal_reason_ = "completed_all_epochs";
    last_stage_ = StageName(TrainingTraceStage::Complete);
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    WriteLocked();
    active_ = false;
}

void CrashRunRecorder::MarkEarlyStopped(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    status_ = "early_stopped";
    terminal_reason_ = reason;
    last_stage_ = StageName(TrainingTraceStage::EarlyStopped);
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    WriteLocked();
    active_ = false;
}

void CrashRunRecorder::MarkCancelled() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    status_ = "cancelled";
    terminal_reason_ = "user_cancelled";
    last_stage_ = StageName(TrainingTraceStage::Cancelled);
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    WriteLocked();
    active_ = false;
}

void CrashRunRecorder::MarkFailed(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    status_ = "failed";
    last_stage_ = StageName(TrainingTraceStage::Failed);
    last_event_time_ = NowIso8601();
    last_thread_id_ = ThreadIdString();
    failure_reason_ = reason;
    terminal_reason_ = reason;
    WriteLocked();
    active_ = false;
}

std::optional<CrashRunSummary> CrashRunRecorder::LoadLastRun() {
    const auto path = CurrentRunPath();
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    try {
        std::ifstream file(path);
        nlohmann::json j;
        file >> j;

        CrashRunSummary summary;
        summary.available = true;
        summary.run_id = j.value("run_id", "");
        summary.status = j.value("status", "");
        summary.dataset_name = j.value("dataset_name", "");
        summary.backend = j.value("backend", "");
        summary.last_stage = j.value("last_stage", "");
        summary.last_event_time = j.value("last_event_time", "");
        summary.epoch = j.value("epoch", 0);
        summary.last_executed_epoch =
            j.value("last_executed_epoch", summary.epoch);
        summary.batch = j.value("batch", 0);
        summary.total_batches = j.value("total_batches", 0);
        summary.epochs = j.value("epochs", 0);
        summary.batch_size = j.value("batch_size", 0);
        summary.sample_count = j.value("sample_count", static_cast<size_t>(0));
        summary.loss = j.value("loss", 0.0f);
        summary.accuracy = j.value("accuracy", 0.0f);
        summary.file_path = path.string();
        summary.terminal_reason = j.value("terminal_reason", "");
        summary.failure_reason = j.value("failure_reason", "");
        summary.checkpoint_used = j.value("checkpoint_used", "");
        summary.restored_checkpoint_epoch =
            j.value("restored_checkpoint_epoch", 0);
        summary.restored_checkpoint_step =
            j.value("restored_checkpoint_step", 0);
        summary.active_model_provenance =
            j.value("active_model_provenance", "");
        summary.panel_events = j.value("panel_events", std::vector<std::string>{});

        if (summary.status == "running") {
            summary.suspected_crash = true;
            summary.status = "suspected crash";
            summary.warning =
                "The last run never wrote a clean completion marker. "
                "Check Windows Error Reporting for a matching APPCRASH.";
        }
        AttachLatestWerReport(summary);
        if (summary.windows_crash_available && summary.suspected_crash) {
            summary.warning += " A local WER report was found and attached below.";
        }
        return summary;
    } catch (const std::exception& e) {
        spdlog::warn("CrashRunRecorder: failed to load {}: {}", path.string(), e.what());
        return std::nullopt;
    }
}

void CrashRunRecorder::WriteLocked() {
    try {
        std::filesystem::create_directories(DebugRunDir());

        nlohmann::json j = {
            {"run_id", run_id_},
            {"status", status_},
            {"dataset_name", dataset_name_},
            {"domain", domain_},
            {"backend", backend_},
            {"last_stage", last_stage_},
            {"last_event_time", last_event_time_},
            {"last_thread_id", last_thread_id_},
            {"epoch", epoch_},
            {"last_executed_epoch", last_executed_epoch_},
            {"batch", batch_},
            {"total_batches", total_batches_},
            {"epochs", epochs_},
            {"batch_size", batch_size_},
            {"sample_count", sample_count_},
            {"loss", loss_},
            {"accuracy", accuracy_},
            {"failure_reason", failure_reason_},
            {"terminal_reason", terminal_reason_},
            {"checkpoint_used", checkpoint_used_},
            {"restored_checkpoint_epoch", restored_checkpoint_epoch_},
            {"restored_checkpoint_step", restored_checkpoint_step_},
            {"active_model_provenance", active_model_provenance_},
            {"panel_events", panel_events_}
        };

        std::ofstream file(CurrentRunPath(), std::ios::trunc);
        file << std::setw(2) << j << '\n';
    } catch (const std::exception& e) {
        spdlog::warn("CrashRunRecorder: failed to write heartbeat: {}", e.what());
    }
}

std::string CrashRunRecorder::NowIso8601() {
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

std::string CrashRunRecorder::ThreadIdString() {
    std::ostringstream out;
    out << std::this_thread::get_id();
    return out.str();
}

std::string CrashRunRecorder::DomainName(PreprocessingDomain domain) {
    switch (domain) {
        case PreprocessingDomain::Tabular: return "Tabular";
        case PreprocessingDomain::Image: return "Image";
        case PreprocessingDomain::Audio: return "Audio";
        case PreprocessingDomain::Text: return "Text";
        case PreprocessingDomain::TimeSeries: return "TimeSeries";
        case PreprocessingDomain::General: return "General";
    }
    return "Unknown";
}

std::string CrashRunRecorder::BackendName() {
    try {
        switch (af::getActiveBackend()) {
            case AF_BACKEND_CPU: return "CPU";
            case AF_BACKEND_CUDA: return "CUDA";
            case AF_BACKEND_OPENCL: return "OpenCL";
            case AF_BACKEND_ONEAPI: return "oneAPI";
            default: return "Unknown";
        }
    } catch (...) {
        return "Unknown";
    }
}

} // namespace cyxwiz
