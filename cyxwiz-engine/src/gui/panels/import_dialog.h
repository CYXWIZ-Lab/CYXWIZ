#pragma once

#include "../panel.h"
#include "../../core/model_format.h"
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * Open Model File (File > Open Model File..., Ctrl+I): reads a model file on a
 * worker, shows what it holds, and opens its graph in Studio. The weights are
 * not loaded.
 */
class ImportDialog : public Panel {
public:
    ImportDialog();
    ~ImportDialog() override;

    void Render() override;

    void Open();
    void Close();
    bool IsOpen() const { return is_open_; }

    // Called with the file's graph; returns false when Studio could not open it.
    using OpenGraphCallback = std::function<bool(
        const ProbeResult& probe,
        const std::string& graph_json,
        const std::string& input_path)>;
    void SetOpenGraphCallback(OpenGraphCallback callback) { open_graph_callback_ = std::move(callback); }

private:
    // One read of a file on a worker: a probe, or the graph's extraction. The
    // worker writes only into its job, so a reopened dialog never races it.
    struct Job {
        std::string path;
        bool extract = false;
        std::atomic<bool> done{false};
        ProbeResult probe;
        std::optional<std::string> graph_json;
    };

    void RenderFileField();
    void RenderFile();
    void RenderButtons();

    void StartJob(const std::string& path, bool extract);
    void CollectJob();

    bool is_open_ = false;
    char input_path_[512] = "";

    // The probed file (probed_path_ is the file probe_ describes).
    std::string probed_path_;
    ProbeResult probe_;
    std::string error_;  // a failed graph extraction or open

    std::shared_ptr<Job> job_;
    std::vector<std::pair<std::thread, std::shared_ptr<Job>>> workers_;

    OpenGraphCallback open_graph_callback_;
};

} // namespace cyxwiz
