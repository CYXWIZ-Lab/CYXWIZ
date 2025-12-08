#pragma once

#include "../panel.h"
#include "../../core/model_format.h"
#include "../../core/model_importer.h"
#include <string>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

namespace cyxwiz {

// Forward declarations
class SequentialModel;

/**
 * Import Dialog Panel
 * UI for importing trained models from various formats
 */
class ImportDialog : public Panel {
public:
    ImportDialog();
    ~ImportDialog() override;

    void Render() override;

    // Dialog control
    void Open();
    void Close();
    bool IsOpen() const { return is_open_; }

    // Callbacks
    using ImportCompleteCallback = std::function<void(const ImportResult&, const std::string& graph_json)>;
    void SetImportCompleteCallback(ImportCompleteCallback callback) {
        import_complete_callback_ = callback;
    }

    // Get the last imported graph JSON (for loading into NodeEditor)
    const std::string& GetImportedGraphJson() const { return imported_graph_json_; }

    // Transfer learning getters
    bool IsTransferLearningEnabled() const { return enable_transfer_learning_; }
    int GetFreezeMode() const { return freeze_mode_; }
    int GetUnfreezeLastN() const { return unfreeze_last_n_; }
    const std::vector<bool>& GetLayerTrainable() const { return layer_trainable_; }

private:
    void RenderFileSelection();
    void RenderProbeInfo();
    void RenderOptions();
    void RenderProgress();
    void RenderButtons();

    // File dialog helper
    std::string OpenFileDialog(const char* filter, const char* title);

    // Probe selected file
    void ProbeFile();

    // Start import in background thread
    void StartImport();

    // State
    bool is_open_ = false;

    // File selection
    char input_path_[512] = "";
    bool file_probed_ = false;
    ProbeResult probe_result_;

    // Import options
    ImportOptions import_options_;
    bool load_optimizer_state_ = false;
    bool load_training_history_ = false;
    bool strict_mode_ = true;
    bool allow_shape_mismatch_ = false;

    // Transfer learning options
    bool enable_transfer_learning_ = false;
    int freeze_mode_ = 0;  // 0=None, 1=All except last N, 2=Custom
    int unfreeze_last_n_ = 2;  // Number of layers to keep trainable at the end
    std::vector<bool> layer_trainable_;  // Per-layer trainable status for custom mode

    // Import state
    std::atomic<bool> is_importing_{false};
    std::atomic<bool> is_probing_{false};
    std::atomic<int> import_progress_{0};
    std::atomic<int> import_total_{0};
    std::string import_status_;
    std::mutex status_mutex_;
    std::unique_ptr<std::thread> import_thread_;

    ImportResult last_result_;
    std::string imported_graph_json_;
    bool show_result_ = false;

    // Callbacks
    ImportCompleteCallback import_complete_callback_;
};

} // namespace cyxwiz
