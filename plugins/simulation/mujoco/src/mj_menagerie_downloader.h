#pragma once

// MjMenagerieDownloader — Downloads MuJoCo Menagerie models from GitHub
//
// Uses GitHub API to list files in a model directory, then downloads each
// file from raw.githubusercontent.com. Supports async download with
// progress reporting and cancellation.

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>

namespace cyxwiz::plugin::mujoco {

class MjMenagerieDownloader {
public:
    MjMenagerieDownloader() = default;
    ~MjMenagerieDownloader();

    // Non-copyable
    MjMenagerieDownloader(const MjMenagerieDownloader&) = delete;
    MjMenagerieDownloader& operator=(const MjMenagerieDownloader&) = delete;

    // Download a model directory from Menagerie GitHub repo.
    // repo_path: directory name in menagerie repo (e.g. "franka_emika_panda")
    // dest_dir:  local directory to save files (e.g. ~/.cyxwiz/menagerie/franka_emika_panda)
    // Returns true on success.
    bool DownloadModel(const std::string& repo_path, const std::string& dest_dir);

    // Async wrapper — runs download in background thread
    using DoneCallback = std::function<void(bool success, const std::string& error)>;
    void DownloadModelAsync(const std::string& repo_path, const std::string& dest_dir,
                            DoneCallback done_cb = {});

    void CancelDownload();

    // State queries
    bool IsDownloading() const { return downloading_; }
    float GetProgress() const { return progress_; }
    std::string GetError() const;
    std::string GetStatusText() const;

private:
    // Download a single file from raw.githubusercontent.com
    bool DownloadFile(const std::string& url_path, const std::string& local_path);

    // List files in a GitHub directory via API
    struct FileEntry {
        std::string path;     // relative path within model dir
        std::string download_url;
        bool is_dir = false;
    };
    std::vector<FileEntry> ListGitHubDir(const std::string& api_path);

    std::atomic<bool> downloading_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<float> progress_{0.0f};

    mutable std::mutex status_mutex_;
    std::string error_;
    std::string status_text_;

    std::thread download_thread_;
};

} // namespace cyxwiz::plugin::mujoco
