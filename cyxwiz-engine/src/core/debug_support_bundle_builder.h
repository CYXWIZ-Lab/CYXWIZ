#pragma once

#include "cyxwiz/backend_placement_observation.h"

#include "crash_run_recorder.h"
#include "debug_run_store.h"
#include "runtime_log_export.h"
#include "training_trace_collector.h"

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugSupportBundleInput {
    std::string request_id;
    std::string reason;
    DebugRunStoreRecord debug_run;
    CrashRunSummary crash_run;
    TrainingTraceSummary training_trace;
    std::vector<BackendPlacementObservation> placement_observations;
    std::map<std::string, std::string> environment;
    std::vector<std::string> recent_logs;
    std::optional<RuntimeLogExportSnapshot> runtime_log_slice;
    bool allow_hq_upload = false;
};

class DebugSupportBundleBuilder {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.support_bundle.v1";

    nlohmann::json Build(const DebugSupportBundleInput& input) const;

    static nlohmann::json RedactJson(const nlohmann::json& value);
    static std::string RedactString(const std::string& value);

private:
    static nlohmann::json DebugRunToJson(const DebugRunStoreRecord& record);
    static nlohmann::json CrashRunToJson(const CrashRunSummary& summary);
    static nlohmann::json TrainingTraceToJson(
        const TrainingTraceSummary& summary);
    static nlohmann::json PlacementObservationsToJson(
        const std::vector<BackendPlacementObservation>& observations);
    static nlohmann::json PlacementObservationToJson(
        const BackendPlacementObservation& observation);
};

} // namespace cyxwiz
