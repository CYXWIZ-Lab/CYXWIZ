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

// =============================================================================
// URL Import — download from arbitrary URL
// =============================================================================

// Parse a URL into host, port, path components
static bool ParseUrl(const std::string& url, std::string& host, std::string& path, bool& is_https) {
    std::string work = url;
    is_https = true;

    if (work.starts_with("https://")) {
        work = work.substr(8);
    } else if (work.starts_with("http://")) {
        work = work.substr(7);
        is_https = false;
    } else {
        return false;
    }

    auto slash_pos = work.find('/');
    if (slash_pos == std::string::npos) {
        host = work;
        path = "/";
    } else {
        host = work.substr(0, slash_pos);
        path = work.substr(slash_pos);
    }
    return !host.empty();
}

// Check if URL is a GitHub tree/blob URL and extract owner/repo/branch/path
static bool ParseGitHubUrl(const std::string& url,
                            std::string& owner, std::string& repo,
                            std::string& branch, std::string& dir_path,
                            bool& is_dir) {
    // Patterns:
    //   https://github.com/owner/repo/tree/branch/path  (directory)
    //   https://github.com/owner/repo/blob/branch/path  (file)
    std::string prefix = "https://github.com/";
    if (!url.starts_with(prefix)) return false;

    std::string rest = url.substr(prefix.size());
    // owner/repo/tree|blob/branch/path...
    auto parts_split = [](const std::string& s, char delim) {
        std::vector<std::string> parts;
        size_t start = 0;
        while (start < s.size()) {
            auto pos = s.find(delim, start);
            if (pos == std::string::npos) {
                parts.push_back(s.substr(start));
                break;
            }
            parts.push_back(s.substr(start, pos - start));
            start = pos + 1;
        }
        return parts;
    };

    auto parts = parts_split(rest, '/');
    if (parts.size() < 4) return false;

    owner = parts[0];
    repo = parts[1];
    std::string type = parts[2]; // "tree" or "blob"
    branch = parts[3];

    if (type != "tree" && type != "blob") return false;
    is_dir = (type == "tree");

    // Reconstruct path
    dir_path.clear();
    for (size_t i = 4; i < parts.size(); i++) {
        if (!dir_path.empty()) dir_path += "/";
        dir_path += parts[i];
    }

    return !owner.empty() && !repo.empty();
}

bool MjMenagerieDownloader::DownloadFromUrl(const std::string& url, const std::string& dest_dir) {
    downloading_ = true;
    progress_ = 0.0f;

    {
        std::lock_guard lock(status_mutex_);
        error_.clear();
        status_text_ = "Analyzing URL...";
    }

    // Case 1: GitHub tree/blob URL
    std::string owner, repo, branch, dir_path;
    bool is_dir = false;
    if (ParseGitHubUrl(url, owner, repo, branch, dir_path, is_dir)) {
        if (is_dir) {
            // Download entire directory via GitHub API
            std::string full_repo = owner + "/" + repo;

            // List files via API
            httplib::SSLClient cli(kGitHubApiHost, 443);
            cli.set_connection_timeout(10);
            cli.set_read_timeout(30);

            httplib::Headers headers = {
                {"User-Agent", "CyxWiz-Engine/1.0"},
                {"Accept", "application/vnd.github.v3+json"}
            };

            // Recursively collect files
            struct DirEntry { std::string api_path; };
            std::vector<DirEntry> dirs_to_scan;
            dirs_to_scan.push_back({dir_path});
            std::vector<std::pair<std::string, std::string>> files; // {raw_path, rel_path}

            while (!dirs_to_scan.empty() && !cancel_requested_) {
                auto d = dirs_to_scan.back();
                dirs_to_scan.pop_back();

                std::string api_url = "/repos/" + full_repo + "/contents/" + d.api_path + "?ref=" + branch;
                auto res = cli.Get(api_url, headers);
                if (!res || res->status != 200) {
                    std::lock_guard lock(status_mutex_);
                    error_ = "GitHub API failed (status: " + std::to_string(res ? res->status : 0) + ")";
                    downloading_ = false;
                    return false;
                }

                try {
                    auto json = nlohmann::json::parse(res->body);
                    for (const auto& item : json) {
                        std::string item_path = item.value("path", "");
                        std::string item_type = item.value("type", "");
                        if (item_type == "dir") {
                            dirs_to_scan.push_back({item_path});
                        } else {
                            std::string rel = item_path;
                            if (rel.starts_with(dir_path + "/"))
                                rel = rel.substr(dir_path.size() + 1);
                            std::string raw = "/" + full_repo + "/" + branch + "/" + item_path;
                            files.push_back({raw, rel});
                        }
                    }
                } catch (...) {
                    std::lock_guard lock(status_mutex_);
                    error_ = "Failed to parse GitHub API response";
                    downloading_ = false;
                    return false;
                }
            }

            if (files.empty() || cancel_requested_) {
                downloading_ = false;
                return false;
            }

            // Download files
            int total = static_cast<int>(files.size());
            int done = 0;
            spdlog::info("UrlImport: Downloading {} files from {}/{}", total, full_repo, dir_path);

            for (const auto& [raw_path, rel_path] : files) {
                if (cancel_requested_) break;
                {
                    std::lock_guard lock(status_mutex_);
                    fs::path p(rel_path);
                    status_text_ = "Downloading " + p.filename().string() +
                                   " (" + std::to_string(done + 1) + "/" + std::to_string(total) + ")";
                }
                std::string local = (fs::path(dest_dir) / rel_path).string();
                if (!DownloadFile(raw_path, local)) {
                    std::lock_guard lock(status_mutex_);
                    error_ = "Failed to download: " + raw_path;
                    downloading_ = false;
                    return false;
                }
                done++;
                progress_ = static_cast<float>(done) / static_cast<float>(total);
            }

            {
                std::lock_guard lock(status_mutex_);
                status_text_ = "Complete";
            }
            downloading_ = false;
            return !cancel_requested_;
        } else {
            // Single file from GitHub blob — convert to raw URL
            std::string raw_path = "/" + owner + "/" + repo + "/" + branch + "/" + dir_path;
            std::string filename = fs::path(dir_path).filename().string();
            std::string local = (fs::path(dest_dir) / filename).string();

            {
                std::lock_guard lock(status_mutex_);
                status_text_ = "Downloading " + filename;
            }

            bool ok = DownloadFile(raw_path, local);
            progress_ = ok ? 1.0f : 0.0f;
            {
                std::lock_guard lock(status_mutex_);
                status_text_ = ok ? "Complete" : "Failed";
                if (!ok) error_ = "Failed to download file";
            }
            downloading_ = false;
            return ok;
        }
    }

    // Case 2: Direct URL (raw.githubusercontent.com or any HTTPS host)
    std::string host, path;
    bool is_https = true;
    if (!ParseUrl(url, host, path, is_https)) {
        std::lock_guard lock(status_mutex_);
        error_ = "Invalid URL format";
        downloading_ = false;
        return false;
    }

    {
        std::lock_guard lock(status_mutex_);
        std::string filename = fs::path(path).filename().string();
        status_text_ = "Downloading " + filename;
    }

    // Download single file
    std::string filename = fs::path(path).filename().string();
    if (filename.empty()) filename = "model.xml";
    std::string local = (fs::path(dest_dir) / filename).string();

    fs::create_directories(dest_dir);

    httplib::SSLClient cli(host, 443);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(60);

    httplib::Headers headers = {{"User-Agent", "CyxWiz-Engine/1.0"}};
    auto res = cli.Get(path, headers);

    if (!res || (res->status != 200 && res->status != 301 && res->status != 302)) {
        std::lock_guard lock(status_mutex_);
        error_ = "Download failed (status: " + std::to_string(res ? res->status : 0) + ")";
        downloading_ = false;
        return false;
    }

    // Handle redirect
    if (res->status == 301 || res->status == 302) {
        auto loc = res->get_header_value("Location");
        if (!loc.empty()) {
            downloading_ = false;
            return DownloadFromUrl(loc, dest_dir);
        }
    }

    fs::path p(local);
    fs::create_directories(p.parent_path());
    std::ofstream ofs(local, std::ios::binary);
    if (!ofs.is_open()) {
        std::lock_guard lock(status_mutex_);
        error_ = "Cannot write to " + local;
        downloading_ = false;
        return false;
    }
    ofs.write(res->body.data(), static_cast<std::streamsize>(res->body.size()));

    progress_ = 1.0f;
    {
        std::lock_guard lock(status_mutex_);
        status_text_ = "Complete";
    }
    downloading_ = false;
    spdlog::info("UrlImport: Downloaded '{}' to '{}'", url, local);
    return true;
}

void MjMenagerieDownloader::DownloadFromUrlAsync(const std::string& url,
                                                  const std::string& dest_dir,
                                                  DoneCallback done_cb) {
    CancelDownload();
    download_thread_ = std::thread([this, url, dest_dir, done_cb]() {
        bool ok = DownloadFromUrl(url, dest_dir);
        if (done_cb) done_cb(ok, GetError());
    });
}

} // namespace cyxwiz::plugin::mujoco
