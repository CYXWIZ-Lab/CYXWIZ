#include "cloud_job_datasets.h"

#include "datastream_client.h"

#include <spdlog/spdlog.h>

#include <map>

namespace network {

namespace {
constexpr int64_t kTokenMarginSeconds = 5 * 60;
constexpr int64_t kTokenDefaultSeconds = 24 * 60 * 60;
constexpr int64_t kTokenMinimumSeconds = 15 * 60;
}  // namespace

int64_t CloudAccessTtlSeconds(int64_t reservation_end_unix, int64_t now_unix) {
    if (reservation_end_unix <= 0 || now_unix <= 0) return kTokenDefaultSeconds;
    const int64_t remaining = reservation_end_unix - now_unix + kTokenMarginSeconds;
    return remaining < kTokenMinimumSeconds ? kTokenMinimumSeconds : remaining;
}

bool FillCloudJobDatasets(const cyxwiz::GraphJobDatasetPlan& plan, DataStreamClient* client,
                          const std::string& gateway_address, int64_t ttl_seconds,
                          cyxwiz::protocol::JobConfig& config, std::string& error) {
    if (plan.cloud.empty()) {
        config.clear_cloud_datasets();
        return true;
    }
    if (!client || !client->IsConnected()) {
        error = "dataset '" + plan.cloud.front().dataset_name +
                "' lives in CyxCloud but the Engine has no CyxCloud connection (open the Cloud browser while signed in)";
        return false;
    }
    if (gateway_address.empty()) {
        error = "no CyxCloud gateway address is configured (set the server in Preferences > Servers)";
        return false;
    }

    // One token per dataset; several files of one dataset share it.
    std::map<std::string, std::string> tokens;
    google::protobuf::RepeatedPtrField<cyxwiz::protocol::CloudDatasetSource> sources;
    for (const auto& entry : plan.cloud) {
        auto token = tokens.find(entry.dataset_id);
        if (token == tokens.end()) {
            DataStreamAccessToken issued;
            if (!client->CreateAccessToken(entry.dataset_id, issued, /*node_id=*/"", ttl_seconds)) {
                error = "CyxCloud refused an access token for dataset '" + entry.dataset_name + "': " +
                        client->GetLastError();
                return false;
            }
            token = tokens.emplace(entry.dataset_id, issued.token).first;
        }
        auto* source = sources.Add();
        source->set_dataset_name(entry.dataset_name);
        source->set_gateway_address(gateway_address);
        source->set_file_id(entry.file_id);
        source->set_access_token(token->second);
        source->set_size_bytes(entry.size_bytes);
        source->set_file_name(entry.file_name);
    }
    config.mutable_cloud_datasets()->Swap(&sources);
    spdlog::info("Job {}: {} dataset(s) come from CyxCloud at {} ({} token(s), {} s)", config.job_id(),
                 plan.cloud.size(), gateway_address, tokens.size(), ttl_seconds);
    return true;
}

}  // namespace network
