#pragma once

// Serves a remote job's dataset files to its Server Node (TOFIX118 P2, owner
// decision 2026-09-26: whole prepared files, not per-batch streaming).
//
// RegisterJob plans the job from its graph (PlanGraphJobDatasets): the
// training/validation Data Inputs are shipped, a supplied Test input never
// leaves the Engine, and graphs the node could not train are refused before
// submission. Datasets are read through the graph dataset catalog the Engine
// registry installs: an Arrow table is exported once to Arrow IPC in the job's
// cache folder (chosen by the caller, e.g. inside the project - not the system
// drive), a Parquet-backed dataset ships its source file. Requests are
// answered with ordered chunks carrying the whole file's size and SHA-256.

#include "execution.pb.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>

namespace network {

class DatasetFileServer {
public:
    using SendChunk = std::function<bool(const cyxwiz::protocol::DatasetFileChunk&)>;

    DatasetFileServer() = default;
    ~DatasetFileServer();

    // False with a reason when the job cannot be served (graph refused by the
    // headless runner, or a dataset the graph needs is not loaded). Exports
    // go to cache_dir (created on first use, deleted on unregister).
    bool RegisterJob(const std::string& job_id, const std::string& graph_json,
                     const std::filesystem::path& cache_dir, std::string& error);
    // Forgets the job and deletes its exported files.
    void UnregisterJob(const std::string& job_id);
    bool HasJob(const std::string& job_id) const;

    // Sends the requested file from request.offset in chunk_bytes pieces;
    // an error is one chunk with status != SUCCESS. False if a send failed.
    bool HandleRequest(const cyxwiz::protocol::DatasetFileRequest& request, const SendChunk& send,
                       size_t chunk_bytes = 1024 * 1024);

private:
    struct PreparedFile {
        std::filesystem::path path;
        std::string format;  // "arrow_ipc" | "parquet"
        std::string sha256;
        std::uintmax_t size = 0;
        bool exported = false;  // created by us (deleted on unregister)
    };
    struct Job {
        std::filesystem::path cache_dir;
        std::set<std::string> ship;
        std::set<std::string> kept_private;
        std::map<std::string, PreparedFile> files;
    };

    bool Prepare(Job& job, const std::string& dataset_name, PreparedFile& file, std::string& error);

    std::map<std::string, Job> jobs_;
    mutable std::mutex mutex_;
};

}  // namespace network
