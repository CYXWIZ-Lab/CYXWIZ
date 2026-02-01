#include "mj_menagerie_downloader.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace cyxwiz::plugin::mujoco {

static const char* kGitHubApiHost = "api.github.com";
static const char* kGitHubRawHost = "raw.githubusercontent.com";
static const char* kMenagerieRepo = "google-deepmind/mujoco_menagerie";
static const char* kBranch = "main";

MjMenagerieDownloader::~MjMenagerieDownloader() {
    CancelDownload();
}

void MjMenagerieDownloader::CancelDownload() {
    cancel_requested_ = true;
    if (download_thread_.joinable()) {
        download_thread_.join();
    }
    cancel_requested_ = false;
}

std::string MjMenagerieDownloader::GetError() const {
    std::lock_guard lock(status_mutex_);
    return error_;
}

std::string MjMenagerieDownloader::GetStatusText() const {
    std::lock_guard lock(status_mutex_);
    return status_text_;
}

std::vector<MjMenagerieDownloader::FileEntry> MjMenagerieDownloader::ListGitHubDir(
    const std::string& api_path)
{
    std::vector<FileEntry> entries;

    httplib::SSLClient cli(kGitHubApiHost, 443);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);

    httplib::Headers headers = {
        {"User-Agent", "CyxWiz-Engine/1.0"},
        {"Accept", "application/vnd.github.v3+json"}
    };

    std::string url = "/repos/" + std::string(kMenagerieRepo) + "/contents/" + api_path + "?ref=" + kBranch;
    auto res = cli.Get(url, headers);

    if (!res || res->status != 200) {
        spdlog::error("MenagerieDownloader: GitHub API request failed for '{}' (status: {})",
                      api_path, res ? res->status : 0);
        return entries;
    }

    try {
        auto json = nlohmann::json::parse(res->body);
        for (const auto& item : json) {
            FileEntry entry;
            entry.path = item.value("path", "");
            entry.is_dir = (item.value("type", "") == "dir");
            if (!entry.is_dir) {
                entry.download_url = item.value("download_url", "");
            }
            entries.push_back(std::move(entry));
        }
    } catch (const std::exception& e) {
        spdlog::error("MenagerieDownloader: Failed to parse GitHub API response: {}", e.what());
    }

    return entries;
}

bool MjMenagerieDownloader::DownloadFile(const std::string& url_path, const std::string& local_path) {
    // url_path is the full path after the host, e.g. /google-deepmind/mujoco_menagerie/main/franka_emika_panda/scene.xml
    httplib::SSLClient cli(kGitHubRawHost, 443);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(60);

    httplib::Headers headers = {
        {"User-Agent", "CyxWiz-Engine/1.0"}
    };

    auto res = cli.Get(url_path, headers);
    if (!res || res->status != 200) {
        spdlog::error("MenagerieDownloader: Failed to download '{}' (status: {})",
                      url_path, res ? res->status : 0);
        return false;
    }

    // Ensure parent directory exists
    fs::path p(local_path);
    fs::create_directories(p.parent_path());

    std::ofstream ofs(local_path, std::ios::binary);
    if (!ofs.is_open()) {
        spdlog::error("MenagerieDownloader: Cannot write to '{}'", local_path);
        return false;
    }

    ofs.write(res->body.data(), static_cast<std::streamsize>(res->body.size()));
    return true;
}

bool MjMenagerieDownloader::DownloadModel(const std::string& repo_path, const std::string& dest_dir) {
    downloading_ = true;
    progress_ = 0.0f;

    {
        std::lock_guard lock(status_mutex_);
        error_.clear();
        status_text_ = "Listing files...";
    }

    // Recursively list all files in the model directory
    struct DirToScan {
        std::string api_path;
        std::string local_prefix;
    };
    std::vector<DirToScan> dirs_to_scan;
    dirs_to_scan.push_back({repo_path, ""});

    std::vector<std::pair<std::string, std::string>> files_to_download;  // {raw_url_path, local_path}

    while (!dirs_to_scan.empty() && !cancel_requested_) {
        auto dir = dirs_to_scan.back();
        dirs_to_scan.pop_back();

        auto entries = ListGitHubDir(dir.api_path);
        for (const auto& entry : entries) {
            if (cancel_requested_) break;

            // Compute relative path within model dir
            std::string rel_path = entry.path;
            if (rel_path.starts_with(repo_path + "/")) {
                rel_path = rel_path.substr(repo_path.size() + 1);
            }

            if (entry.is_dir) {
                dirs_to_scan.push_back({entry.path, rel_path});
            } else {
                // Build raw.githubusercontent.com path
                std::string raw_path = "/" + std::string(kMenagerieRepo) + "/" + kBranch + "/" + entry.path;
                std::string local_file = (fs::path(dest_dir) / rel_path).string();
                files_to_download.push_back({raw_path, local_file});
            }
        }
    }

    if (cancel_requested_) {
        downloading_ = false;
        return false;
    }

    if (files_to_download.empty()) {
        std::lock_guard lock(status_mutex_);
        error_ = "No files found for model '" + repo_path + "'";
        downloading_ = false;
        return false;
    }

    spdlog::info("MenagerieDownloader: Downloading {} files for '{}'",
                 files_to_download.size(), repo_path);

    // Download each file
    int downloaded = 0;
    int total = static_cast<int>(files_to_download.size());

    for (const auto& [raw_path, local_path] : files_to_download) {
        if (cancel_requested_) break;

        {
            std::lock_guard lock(status_mutex_);
            // Show just the filename
            fs::path p(local_path);
            status_text_ = "Downloading " + p.filename().string() +
                           " (" + std::to_string(downloaded + 1) + "/" + std::to_string(total) + ")";
        }

        if (!DownloadFile(raw_path, local_path)) {
            std::lock_guard lock(status_mutex_);
            error_ = "Failed to download: " + raw_path;
            downloading_ = false;
            return false;
        }

        downloaded++;
        progress_ = static_cast<float>(downloaded) / static_cast<float>(total);
    }

    {
        std::lock_guard lock(status_mutex_);
        status_text_ = cancel_requested_ ? "Cancelled" : "Complete";
    }

    downloading_ = false;
    bool success = !cancel_requested_;

    if (success) {
        spdlog::info("MenagerieDownloader: Successfully downloaded '{}' ({} files)",
                     repo_path, downloaded);
    }
    return success;
}

void MjMenagerieDownloader::DownloadModelAsync(const std::string& repo_path,
                                                const std::string& dest_dir,
                                                DoneCallback done_cb) {
    CancelDownload();  // Cancel any previous download

    download_thread_ = std::thread([this, repo_path, dest_dir, done_cb]() {
        bool ok = DownloadModel(repo_path, dest_dir);
        if (done_cb) {
            done_cb(ok, GetError());
        }
    });
}

} // namespace cyxwiz::plugin::mujoco
