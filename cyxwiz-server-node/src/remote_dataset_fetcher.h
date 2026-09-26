#pragma once

// Fetches a remote job's dataset files from the Engine over the training
// stream (TOFIX118 P2, owner decision 2026-09-26: whole files, then local
// training with the shared core). Replaces the per-batch RemoteDataLoader.
//
// Each file is requested with DatasetFileRequest and arrives as ordered
// DatasetFileChunk messages (routed here by the stream's command reader). The
// fetcher checks offsets, total size and the whole file's SHA-256, resumes
// from the last received byte when chunks stop arriving, and renames the file
// into place only when it verifies.

#include "execution.pb.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>

namespace cyxwiz {
namespace server_node {

class RemoteDatasetFetcher {
public:
    using WriteUpdate = std::function<bool(const cyxwiz::protocol::TrainingUpdate&)>;

    RemoteDatasetFetcher(WriteUpdate write, std::string job_id, std::filesystem::path directory,
                         int chunk_timeout_ms = 60000, int max_resumes = 3);

    // Downloads one dataset of the job into the directory; false with a reason.
    bool Fetch(const std::string& dataset_name, std::filesystem::path& file, std::string& error);

    // Called by the stream reader for each chunk from the Engine.
    void OnChunk(const cyxwiz::protocol::DatasetFileChunk& chunk);

    // Unblocks a running Fetch (job stopped or Engine gone).
    void Cancel();

private:
    bool Request(const std::string& dataset_name, int64_t offset, int32_t& request_id);

    WriteUpdate write_;
    std::string job_id_;
    std::filesystem::path directory_;
    int chunk_timeout_ms_;
    int max_resumes_;

    std::mutex mutex_;
    std::condition_variable arrived_;
    std::deque<cyxwiz::protocol::DatasetFileChunk> chunks_;
    std::atomic<bool> cancelled_{false};
    int32_t next_request_id_ = 1;
};

// Plans the job from its graph (the shared core's PlanGraphJobDatasets),
// fetches every training/validation input and rewrites the job so it trains
// locally: model_definition bound to the fetched files (supplied Test inputs
// removed - they stay on the Engine), dataset_uri cleared.
bool FetchRemoteJobDatasets(RemoteDatasetFetcher& fetcher, cyxwiz::protocol::JobConfig& config,
                            std::string& error);

}  // namespace server_node
}  // namespace cyxwiz
