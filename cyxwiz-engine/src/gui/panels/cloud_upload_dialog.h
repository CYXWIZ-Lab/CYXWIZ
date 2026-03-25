#pragma once

#include "../panel.h"
#include "../../network/datastream_client.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace gui {

/**
 * Cloud Upload Dialog
 *
 * Upload local files to CyxCloud for creating datasets.
 * Features:
 * - Drag and drop file support
 * - Multiple file selection
 * - Upload progress tracking
 * - Automatic hash computation
 */
class CloudUploadDialog : public cyxwiz::Panel {
public:
    CloudUploadDialog();
    ~CloudUploadDialog() override;

    void Render() override;

    // Set DataStream client
    void SetDataStreamClient(network::DataStreamClient* client) { datastream_client_ = client; }

    // Show the dialog
    void Show() { visible_ = true; ClearState(); }

    // Callbacks
    using UploadCompleteCallback = std::function<void(const std::vector<std::string>& file_ids)>;
    void SetUploadCompleteCallback(UploadCompleteCallback callback) { on_complete_ = callback; }

private:
    // File to upload
    struct PendingFile {
        std::string path;
        std::string filename;
        int64_t size_bytes = 0;
        bool selected = true;

        // Upload state
        enum class State { Pending, Uploading, Complete, Error };
        State state = State::Pending;
        float progress = 0.0f;
        std::string file_id;  // Set on success
        std::string error;    // Set on error
    };

    void RenderFileList();
    void RenderDropZone();
    void RenderUploadProgress();
    void RenderActions();

    void AddFile(const std::string& path);
    void AddFiles(const std::vector<std::string>& paths);
    void RemoveFile(size_t index);
    void ClearState();

    void StartUpload();
    void UploadNextFile();
    void UploadFile(PendingFile& file);

    // Open file dialog
    void ShowFileBrowser();

    network::DataStreamClient* datastream_client_ = nullptr;
    UploadCompleteCallback on_complete_;

    // Pending files
    std::vector<PendingFile> pending_files_;
    std::mutex files_mutex_;

    // Upload state
    std::atomic<bool> is_uploading_{false};
    std::atomic<bool> should_cancel_{false};
    size_t current_upload_index_ = 0;
    std::vector<std::string> completed_file_ids_;

    // Background upload thread (prevents use-after-free)
    std::thread upload_thread_;

    // UI state
    bool show_file_browser_ = false;
    char file_path_buffer_[512] = "";

    // Error display
    std::string last_error_;
    float error_time_ = 0.0f;
};

} // namespace gui
