#pragma once

// Fetches a job's dataset file from CyxCloud (TOFIX164 slice 3): the
// gateway's DataService.StreamData, one file per call, in order, into the
// job's data directory. The same cache, verification and bind step as the
// Engine-shipped files (RemoteDatasetFetcher); only the source differs.
//
// The gateway ends the stream with an error instead of skipping a chunk it
// cannot serve, and its last chunk carries the SHA-256 of every streamed byte
// and the byte count; the fetcher verifies both before the file is renamed
// into place.

#include "job.pb.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace grpc {
class Channel;
}

namespace cyxwiz {
namespace server_node {

class CloudDatasetFetcher {
public:
    // chunk_timeout_ms bounds the wait for each message from the gateway.
    explicit CloudDatasetFetcher(std::filesystem::path directory, int chunk_timeout_ms = 60000);

    // Downloads one dataset file into the directory; false with a reason that
    // names the gateway, the file and what went wrong.
    bool Fetch(const cyxwiz::protocol::CloudDatasetSource& source, std::filesystem::path& file, std::string& error);

    // Unblocks a running Fetch (job stopped).
    void Cancel();

    // Bytes received by the last Fetch (for the job's timing report).
    uint64_t bytes_received() const { return bytes_received_; }

private:
    std::shared_ptr<grpc::Channel> ChannelFor(const std::string& gateway_address);

    std::filesystem::path directory_;
    int chunk_timeout_ms_;
    std::atomic<bool> cancelled_{false};
    uint64_t bytes_received_ = 0;
};

// A file name that is safe on every file system, from a dataset name.
std::string SafeCloudFileName(const std::string& dataset_name);

// The extension the fetched file gets, from the file's name in CyxCloud:
// .parquet for Parquet, .arrow for Arrow IPC (.arrow, .feather, .ipc), and
// .arrow for anything else (the Engine's plan admits only those formats).
std::string CloudFileExtension(const std::string& file_name);

}  // namespace server_node
}  // namespace cyxwiz
