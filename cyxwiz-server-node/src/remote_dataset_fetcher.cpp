#include "remote_dataset_fetcher.h"

#include <spdlog/spdlog.h>

#include <cctype>
#include <chrono>
#include <fstream>
#include <map>

// Shared training core (after the protocol headers)
#include "core/graph_training_job.h"
#include "core/sha256_digest.h"

#ifdef STATUS_ERROR
#undef STATUS_ERROR
#endif

namespace cyxwiz {
namespace server_node {

namespace fs = std::filesystem;

namespace {

// Dataset names become file names: keep only path-safe characters.
std::string SafeName(const std::string& text) {
    std::string safe;
    for (const char c : text) {
        safe.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_');
    }
    return safe.empty() ? std::string("_") : safe;
}

}  // namespace

RemoteDatasetFetcher::RemoteDatasetFetcher(WriteUpdate write, std::string job_id, fs::path directory,
                                           int chunk_timeout_ms, int max_resumes)
    : write_(std::move(write)),
      job_id_(std::move(job_id)),
      directory_(std::move(directory)),
      chunk_timeout_ms_(chunk_timeout_ms),
      max_resumes_(max_resumes) {}

void RemoteDatasetFetcher::OnChunk(const cyxwiz::protocol::DatasetFileChunk& chunk) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        chunks_.push_back(chunk);
    }
    arrived_.notify_all();
}

void RemoteDatasetFetcher::Cancel() {
    cancelled_ = true;
    arrived_.notify_all();
}

bool RemoteDatasetFetcher::Request(const std::string& dataset_name, int64_t offset, int32_t& request_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request_id = next_request_id_++;
        chunks_.clear();  // anything left belongs to an abandoned request
    }
    cyxwiz::protocol::TrainingUpdate update;
    update.set_job_id(job_id_);
    update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());
    auto* request = update.mutable_dataset_file_request();
    request->set_job_id(job_id_);
    request->set_dataset_name(dataset_name);
    request->set_offset(offset);
    request->set_request_id(request_id);
    return write_(update);
}

bool RemoteDatasetFetcher::Fetch(const std::string& dataset_name, fs::path& file, std::string& error) {
    std::error_code ec;
    fs::create_directories(directory_, ec);
    const fs::path part = directory_ / (SafeName(dataset_name) + ".part");
    std::ofstream output(part, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        error = "cannot write " + part.string();
        return false;
    }

    Sha256Hasher hasher;
    int64_t received = 0;
    int64_t total = -1;
    std::string expected_sha;
    std::string format;
    int resumes = 0;
    int32_t request_id = 0;
    if (!Request(dataset_name, 0, request_id)) {
        error = "could not ask the Engine for dataset '" + dataset_name + "' (stream closed)";
        return false;
    }
    spdlog::info("Job {}: fetching dataset '{}' from the Engine", job_id_, dataset_name);

    while (true) {
        cyxwiz::protocol::DatasetFileChunk chunk;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const bool got = arrived_.wait_for(lock, std::chrono::milliseconds(chunk_timeout_ms_), [this] {
                return cancelled_.load() || !chunks_.empty();
            });
            if (cancelled_) {
                error = "the job was stopped while dataset '" + dataset_name + "' was downloading";
                return false;
            }
            if (!got) {
                lock.unlock();
                if (++resumes > max_resumes_) {
                    error = "the Engine stopped sending dataset '" + dataset_name + "' (" +
                            std::to_string(received) + " bytes received)";
                    return false;
                }
                spdlog::warn("Job {}: dataset '{}' stalled at {} bytes; resuming ({}/{})", job_id_, dataset_name,
                             received, resumes, max_resumes_);
                if (!Request(dataset_name, received, request_id)) {
                    error = "could not resume dataset '" + dataset_name + "' (stream closed)";
                    return false;
                }
                continue;
            }
            chunk = std::move(chunks_.front());
            chunks_.pop_front();
        }
        if (chunk.request_id() != request_id || chunk.dataset_name() != dataset_name) continue;  // stale
        if (chunk.status() != cyxwiz::protocol::STATUS_SUCCESS) {
            error = "the Engine refused dataset '" + dataset_name + "': " + chunk.error().message();
            return false;
        }
        if (total < 0) {
            total = chunk.total_size();
            expected_sha = chunk.sha256();
            format = chunk.format();
        } else if (chunk.total_size() != total || chunk.sha256() != expected_sha || chunk.format() != format) {
            error = "dataset '" + dataset_name + "' changed on the Engine during the download";
            return false;
        }
        if (chunk.offset() != received) {
            error = "dataset '" + dataset_name + "' arrived out of order (offset " + std::to_string(chunk.offset()) +
                    ", expected " + std::to_string(received) + ")";
            return false;
        }
        output.write(chunk.data().data(), static_cast<std::streamsize>(chunk.data().size()));
        if (!output || !hasher.Update(chunk.data(), error)) {
            error = "cannot store dataset '" + dataset_name + "': " + (error.empty() ? part.string() : error);
            return false;
        }
        received += static_cast<int64_t>(chunk.data().size());
        if (received > total) {
            error = "dataset '" + dataset_name + "' is larger than announced";
            return false;
        }
        if (!chunk.last()) continue;

        output.close();
        std::string digest;
        if (received != total) {
            error = "dataset '" + dataset_name + "' ended early (" + std::to_string(received) + " of " +
                    std::to_string(total) + " bytes)";
            return false;
        }
        if (!hasher.Finish(digest, error)) return false;
        if (digest != expected_sha) {
            error = "dataset '" + dataset_name + "' failed its SHA-256 check";
            return false;
        }
        file = directory_ / (SafeName(dataset_name) + (format == "parquet" ? ".parquet" : ".arrow"));
        fs::remove(file, ec);
        fs::rename(part, file, ec);
        if (ec) {
            error = "cannot place dataset '" + dataset_name + "': " + ec.message();
            return false;
        }
        spdlog::info("Job {}: dataset '{}' received ({} bytes, SHA-256 verified)", job_id_, dataset_name, total);
        return true;
    }
}

bool FetchRemoteJobDatasets(RemoteDatasetFetcher& fetcher, cyxwiz::protocol::JobConfig& config,
                            std::string& error) {
    cyxwiz::GraphJobDatasetPlan plan;
    if (!cyxwiz::PlanGraphJobDatasets(config.model_definition(), plan, error)) return false;
    std::map<std::string, std::string> files;
    for (const auto& name : plan.ship) {
        fs::path file;
        if (!fetcher.Fetch(name, file, error)) return false;
        files[name] = file.string();
    }
    std::string bound;
    if (!cyxwiz::BindGraphJobDatasets(config.model_definition(), files, plan.kept_private, bound, error)) {
        return false;
    }
    if (!plan.kept_private.empty()) {
        spdlog::info("Job {}: {} Test input(s) stay on the Engine; the node trains without them", config.job_id(),
                     plan.kept_private.size());
    }
    config.set_model_definition(bound);
    config.set_dataset_uri("");
    return true;
}

}  // namespace server_node
}  // namespace cyxwiz
