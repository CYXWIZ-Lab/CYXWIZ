#pragma once

// The Engine's side of a CyxCloud dataset in a remote job (TOFIX164 slice
// 4a): for every cloud input the plan names, the job carries the gateway
// address, the file and an access token the node presents to the gateway.
// The Engine ships nothing for those inputs.

#include "../core/graph_training_job.h"
#include "job.pb.h"

#include <cstdint>
#include <string>

namespace network {

class DataStreamClient;

// One access token (scope stream, ttl_seconds) per CyxCloud dataset in
// plan.cloud, written into config.cloud_datasets with `gateway_address` (the
// address the node will call). Files of one dataset share its token. False
// with the reason (no CyxCloud connection, gateway refusal); config is then
// left as it was. An empty plan.cloud clears config.cloud_datasets.
bool FillCloudJobDatasets(const cyxwiz::GraphJobDatasetPlan& plan, DataStreamClient* client,
                          const std::string& gateway_address, int64_t ttl_seconds,
                          cyxwiz::protocol::JobConfig& config, std::string& error);

// How long a job's dataset access tokens live: until the reservation ends
// plus a margin, or a day when the end is unknown (TOFIX164).
int64_t CloudAccessTtlSeconds(int64_t reservation_end_unix, int64_t now_unix);

}  // namespace network
