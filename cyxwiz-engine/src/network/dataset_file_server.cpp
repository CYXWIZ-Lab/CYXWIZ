#include "dataset_file_server.h"

#include "../core/arrow_dataset.h"
#include "../core/graph_compiler_dataset_hooks.h"
#include "../core/graph_training_job.h"
#include "../core/sha256_digest.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <vector>

// Windows macros that collide with the protobuf StatusCode enum.
#ifdef STATUS_ERROR
#undef STATUS_ERROR
#endif

namespace network {

namespace fs = std::filesystem;

namespace {

cyxwiz::protocol::DatasetFileChunk ErrorChunk(const cyxwiz::protocol::DatasetFileRequest& request, int code,
                                              const std::string& message) {
    cyxwiz::protocol::DatasetFileChunk chunk;
    chunk.set_status(cyxwiz::protocol::STATUS_ERROR);
    chunk.set_request_id(request.request_id());
    chunk.set_dataset_name(request.dataset_name());
    chunk.set_last(true);
    chunk.mutable_error()->set_code(code);
    chunk.mutable_error()->set_message(message);
    return chunk;
}

// Dataset names become file names: keep only path-safe characters.
std::string SafeName(const std::string& text) {
    std::string safe;
    for (const char c : text) {
        safe.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_');
    }
    return safe.empty() ? std::string("_") : safe;
}

}  // namespace

DatasetFileServer::~DatasetFileServer() {
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [id, job] : jobs_) ids.push_back(id);
    }
    for (const auto& id : ids) UnregisterJob(id);
}

bool DatasetFileServer::RegisterJob(const std::string& job_id, const std::string& graph_json,
                                    const fs::path& cache_dir, std::string& error) {
    cyxwiz::GraphJobDatasetPlan plan;
    if (!cyxwiz::PlanGraphJobDatasets(graph_json, plan, error)) return false;
    for (const auto& name : plan.ship) {
        if (!cyxwiz::GraphArrowDataset(name) && !cyxwiz::GraphParquetDataset(name)) {
            error = "dataset '" + name + "' is not loaded in the Engine (apply its Data Input first)";
            return false;
        }
    }
    Job job;
    job.cache_dir = cache_dir;
    job.ship.insert(plan.ship.begin(), plan.ship.end());
    job.kept_private.insert(plan.kept_private.begin(), plan.kept_private.end());
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_[job_id] = std::move(job);
    spdlog::info("DatasetFileServer: job {} ships {} dataset(s), keeps {} Test input(s) on the Engine", job_id,
                 plan.ship.size(), plan.kept_private.size());
    return true;
}

void DatasetFileServer::UnregisterJob(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = jobs_.find(job_id);
    if (it == jobs_.end()) return;
    for (const auto& [name, file] : it->second.files) {
        if (!file.exported) continue;
        std::error_code ec;
        fs::remove(file.path, ec);
    }
    std::error_code ec;
    fs::remove(it->second.cache_dir, ec);  // only if now empty
    jobs_.erase(it);
}

bool DatasetFileServer::HasJob(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jobs_.count(job_id) > 0;
}

bool DatasetFileServer::Prepare(Job& job, const std::string& dataset_name, PreparedFile& file, std::string& error) {
    if (const auto cached = job.files.find(dataset_name); cached != job.files.end()) {
        file = cached->second;
        return true;
    }
    PreparedFile prepared;
    if (auto arrow = cyxwiz::GraphArrowDataset(dataset_name)) {
        std::error_code ec;
        fs::create_directories(job.cache_dir, ec);
        prepared.path = job.cache_dir / (SafeName(dataset_name) + ".arrow");
        prepared.format = "arrow_ipc";
        prepared.exported = true;
        if (!arrow->ExportFeather(prepared.path.string())) {
            error = "could not export dataset '" + dataset_name + "' to Arrow IPC";
            return false;
        }
    } else if (cyxwiz::GraphParquetDataset(dataset_name)) {
        const auto source = cyxwiz::GraphDatasetSourcePath(dataset_name);
        if (!source || source->empty()) {
            error = "dataset '" + dataset_name + "' has no Parquet source file";
            return false;
        }
        prepared.path = *source;
        prepared.format = "parquet";
    } else {
        error = "dataset '" + dataset_name + "' is no longer loaded in the Engine";
        return false;
    }
    std::error_code ec;
    prepared.size = fs::file_size(prepared.path, ec);
    if (ec) {
        error = "dataset file is unreadable: " + prepared.path.string();
        return false;
    }
    if (!cyxwiz::Sha256File(prepared.path, prepared.sha256, error)) return false;
    job.files[dataset_name] = prepared;
    file = prepared;
    return true;
}

bool DatasetFileServer::HandleRequest(const cyxwiz::protocol::DatasetFileRequest& request, const SendChunk& send,
                                      size_t chunk_bytes) {
    PreparedFile file;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = jobs_.find(request.job_id());
        if (it == jobs_.end()) {
            return send(ErrorChunk(request, 1001, "no datasets registered for job " + request.job_id()));
        }
        Job& job = it->second;
        if (job.kept_private.count(request.dataset_name()) > 0) {
            return send(ErrorChunk(request, 1002,
                                   "dataset '" + request.dataset_name() + "' is the job's Test input and stays "
                                   "on the Engine"));
        }
        if (job.ship.count(request.dataset_name()) == 0) {
            return send(ErrorChunk(request, 1003,
                                   "dataset '" + request.dataset_name() + "' is not an input of the job's graph"));
        }
        std::string error;
        if (!Prepare(job, request.dataset_name(), file, error)) {
            spdlog::error("DatasetFileServer: {}", error);
            return send(ErrorChunk(request, 1004, error));
        }
    }
    const auto offset = static_cast<std::uintmax_t>(std::max<int64_t>(0, request.offset()));
    if (offset > file.size) {
        return send(ErrorChunk(request, 1005, "offset is past the end of the file"));
    }
    std::ifstream input(file.path, std::ios::binary);
    if (!input.is_open()) {
        return send(ErrorChunk(request, 1004, "dataset file is unreadable: " + file.path.string()));
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::string buffer(std::max<size_t>(1, chunk_bytes), '\0');
    std::uintmax_t position = offset;
    do {
        const auto want = static_cast<std::streamsize>(std::min<std::uintmax_t>(buffer.size(), file.size - position));
        input.read(buffer.data(), want);
        if (input.gcount() != want) {
            return send(ErrorChunk(request, 1004, "dataset file changed while it was sent"));
        }
        cyxwiz::protocol::DatasetFileChunk chunk;
        chunk.set_status(cyxwiz::protocol::STATUS_SUCCESS);
        chunk.set_request_id(request.request_id());
        chunk.set_dataset_name(request.dataset_name());
        chunk.set_format(file.format);
        chunk.set_offset(static_cast<int64_t>(position));
        chunk.set_data(buffer.data(), static_cast<size_t>(want));
        chunk.set_total_size(static_cast<int64_t>(file.size));
        chunk.set_sha256(file.sha256);
        position += static_cast<std::uintmax_t>(want);
        chunk.set_last(position >= file.size);
        if (!send(chunk)) return false;
    } while (position < file.size);
    spdlog::info("DatasetFileServer: sent '{}' for job {} ({} bytes from offset {})", request.dataset_name(),
                 request.job_id(), file.size - offset, offset);
    return true;
}

}  // namespace network
