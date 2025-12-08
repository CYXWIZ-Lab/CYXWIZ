#pragma once

#include "../panel.h"
#include "../../core/model_format.h"
#include "../../core/model_exporter.h"
#include <string>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

namespace cyxwiz {

// Forward declarations
class SequentialModel;
class Optimizer;
struct TrainingMetrics;

/**
 * Export Dialog Panel
 * UI for exporting trained models to various formats
 */
class ExportDialog : public Panel {
public:
    ExportDialog();
    ~ExportDialog() override;

    void Render() override;

    // Dialog control
    void Open();
    void Close();
    bool IsOpen() const { return is_open_; }

    // Set model data for export
    void SetModelData(
        SequentialModel* model,
        const Optimizer* optimizer,
        const TrainingMetrics* metrics,
        const std::string& graph_json
    );

    // Callbacks
    using ExportCompleteCallback = std::function<void(const ExportResult&)>;
    void SetExportCompleteCallback(ExportCompleteCallback callback) {
        export_complete_callback_ = callback;
    }

private:
    void RenderFormatSelection();
    void RenderOptions();
    void RenderMetadata();
    void RenderProgress();
    void RenderButtons();

    // File dialog helper
    std::string SaveFileDialog(const char* filter, const char* default_ext);

    // Start export in background thread
    void StartExport();

    // State
    bool is_open_ = false;

    // Model data (non-owning pointers)
    SequentialModel* model_ = nullptr;
    const Optimizer* optimizer_ = nullptr;
    const TrainingMetrics* metrics_ = nullptr;
    std::string graph_json_;

    // Export settings
    ModelFormat selected_format_ = ModelFormat::CyxModel;
    ExportOptions export_options_;

    // UI buffers
    char output_path_[512] = "";
    char model_name_[256] = "";
    char author_[256] = "";
    char description_[1024] = "";

    // Options
    bool include_optimizer_state_ = true;
    bool include_training_history_ = true;
    bool include_graph_ = true;
    int quantization_index_ = 0;
    int opset_version_ = 17;
    bool compress_ = true;

    // Export state
    std::atomic<bool> is_exporting_{false};
    std::atomic<int> export_progress_{0};
    std::atomic<int> export_total_{0};
    std::string export_status_;
    std::mutex status_mutex_;
    std::unique_ptr<std::thread> export_thread_;

    ExportResult last_result_;
    bool show_result_ = false;

    // Callbacks
    ExportCompleteCallback export_complete_callback_;
};

} // namespace cyxwiz
