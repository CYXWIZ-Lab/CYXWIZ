#include "cyxwiz/backend_placement_observation.h"

#include "arrayfire_backend_utils.h"
#include "cyxwiz/cyxwiz.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <ctime>
#include <tuple>
#include <unordered_map>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

std::mutex g_observation_mutex;
std::unordered_map<std::string, BackendPlacementObservation> g_observations;

constexpr int kPlacementObservationCacheSchemaVersion = 1;

std::string BuildObservationKey(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature) {
    return op_type + "|" + backend + "|" + device + "|" + dtype + "|" +
           shape_signature;
}

void AppendShape(std::ostringstream& out, const std::vector<size_t>& shape) {
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << "x";
        }
        out << shape[i];
    }
    out << "]";
}

std::string CurrentTimestampUtc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

std::string CyxWizBackendVersionString() {
    std::ostringstream out;
    out << CYXWIZ_VERSION_MAJOR << "."
        << CYXWIZ_VERSION_MINOR << "."
        << CYXWIZ_VERSION_PATCH;
    return out.str();
}

void SetError(std::string* error_message, const std::string& message) {
    if (error_message != nullptr) {
        *error_message = message;
    }
}

nlohmann::json ObservationToJson(
    const BackendPlacementObservation& observation) {
    return nlohmann::json{
        {"op_type", observation.op_type},
        {"backend", observation.backend},
        {"device", observation.device},
        {"dtype", observation.dtype},
        {"shape_signature", observation.shape_signature},
        {"reason_code", observation.reason_code},
        {"source", observation.source},
        {"detail", observation.detail},
        {"timestamp", observation.timestamp},
        {"probe_outcome", observation.probe_outcome},
        {"probe_scope", observation.probe_scope},
    };
}

bool ObservationFromJson(
    const nlohmann::json& entry,
    BackendPlacementObservation& observation) {
    if (!entry.is_object()) {
        return false;
    }
    const auto string_or_empty = [&entry](const char* key) {
        const auto it = entry.find(key);
        if (it == entry.end() || !it->is_string()) {
            return std::string();
        }
        return it->get<std::string>();
    };
    observation.op_type = string_or_empty("op_type");
    observation.backend = string_or_empty("backend");
    observation.device = string_or_empty("device");
    observation.dtype = string_or_empty("dtype");
    observation.shape_signature = string_or_empty("shape_signature");
    observation.reason_code = string_or_empty("reason_code");
    observation.source = string_or_empty("source");
    observation.detail = string_or_empty("detail");
    observation.timestamp = string_or_empty("timestamp");
    observation.probe_outcome = string_or_empty("probe_outcome");
    observation.probe_scope = string_or_empty("probe_scope");
    if (observation.source.empty()) {
        observation.source = BackendPlacementObservationSource::RuntimeFallback;
    }
    if (observation.timestamp.empty()) {
        observation.timestamp = CurrentTimestampUtc();
    }
    return !observation.op_type.empty() &&
           !observation.backend.empty() &&
           !observation.device.empty() &&
           !observation.dtype.empty() &&
           !observation.shape_signature.empty() &&
           !observation.reason_code.empty();
}

} // namespace

std::string CurrentBackendPlacementDeviceSignature() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const int device_id = af::getDevice();
        char name[64] = {};
        char platform[64] = {};
        char toolkit[64] = {};
        char compute[64] = {};
        af::deviceInfo(name, platform, toolkit, compute);
        std::ostringstream out;
        out << "af_device=" << device_id
            << ";name=" << name
            << ";platform=" << platform
            << ";toolkit=" << toolkit
            << ";compute=" << compute;
        return out.str();
    } catch (...) {
        return "af_device=unknown";
    }
#else
    return "af_device=unavailable";
#endif
}

const char* BackendPlacementProbeOutcomeName(
    BackendPlacementProbeOutcome outcome) {
    switch (outcome) {
        case BackendPlacementProbeOutcome::Safe: return "safe";
        case BackendPlacementProbeOutcome::Unsafe: return "unsafe";
        case BackendPlacementProbeOutcome::Timeout: return "timeout";
        case BackendPlacementProbeOutcome::Unsupported: return "unsupported";
        case BackendPlacementProbeOutcome::Inconclusive: return "inconclusive";
    }
    return "inconclusive";
}

std::string BuildDensePlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    size_t out_features) {
    const size_t in_features = input_shape.empty() ? 0 : input_shape.back();
    std::ostringstream out;
    out << "in_features=" << in_features
        << ";out_features=" << out_features;
    return out.str();
}

std::string BuildEmbeddingPlacementShapeSignature(
    size_t num_embeddings,
    size_t embedding_dim,
    const std::vector<size_t>& input_shape,
    const std::string& index_dtype) {
    std::ostringstream out;
    out << "num_embeddings=" << num_embeddings
        << ";embedding_dim=" << embedding_dim
        << ";input_rank=" << input_shape.size()
        << ";index_dtype=" << index_dtype
        << ";input=";
    AppendShape(out, input_shape);
    return out.str();
}

std::string BuildActivationPlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    const std::string& dtype) {
    std::ostringstream out;
    out << "input=";
    AppendShape(out, input_shape);
    out << ";dtype=" << dtype;
    return out.str();
}

std::string BuildLinearPlacementShapeSignature(
    const std::vector<size_t>& lhs_shape,
    const std::vector<size_t>& rhs_shape,
    const std::vector<size_t>& output_shape,
    const std::string& dtype,
    bool use_bias) {
    std::ostringstream out;
    out << "lhs=";
    AppendShape(out, lhs_shape);
    out << ";rhs=";
    AppendShape(out, rhs_shape);
    out << ";output=";
    AppendShape(out, output_shape);
    out << ";dtype=" << dtype
        << ";bias=" << (use_bias ? "true" : "false");
    return out.str();
}

std::string BuildLossPlacementShapeSignature(
    const std::vector<size_t>& prediction_shape,
    const std::vector<size_t>& target_shape,
    const std::string& reduction,
    const std::string& dtype) {
    std::ostringstream out;
    out << "prediction=";
    AppendShape(out, prediction_shape);
    out << ";target=";
    AppendShape(out, target_shape);
    out << ";reduction=" << reduction
        << ";dtype=" << dtype;
    return out.str();
}

std::string BuildTensorOpPlacementShapeSignature(
    const std::vector<std::vector<size_t>>& input_shapes,
    const std::vector<size_t>& output_shape,
    const std::string& dtype,
    const std::string& attributes) {
    std::ostringstream out;
    out << "inputs=[";
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        AppendShape(out, input_shapes[i]);
    }
    out << "];output=";
    AppendShape(out, output_shape);
    out << ";dtype=" << dtype;
    if (!attributes.empty()) {
        out << ";" << attributes;
    }
    return out.str();
}

std::string BuildTensorLayerPlacementShapeSignature(
    const std::vector<size_t>& input_shape) {
    std::ostringstream out;
    out << "input=";
    AppendShape(out, input_shape);
    return out.str();
}

void RecordBackendPlacementObservation(
    const BackendPlacementObservation& observation) {
    BackendPlacementObservation stored = observation;
    if (stored.timestamp.empty()) {
        stored.timestamp = CurrentTimestampUtc();
    }
    const std::string key = BuildObservationKey(
        stored.op_type,
        stored.backend,
        stored.device,
        stored.dtype,
        stored.shape_signature);
    std::lock_guard<std::mutex> lock(g_observation_mutex);
    g_observations[key] = stored;
}

void RecordBackendPlacementObservationForActiveDevice(
    const std::string& op_type,
    const std::string& backend,
    const std::string& dtype,
    const std::string& shape_signature,
    const std::string& reason_code,
    const std::string& source,
    const std::string& detail) {
    BackendPlacementObservation observation;
    observation.op_type = op_type;
    observation.backend = backend;
    observation.device = CurrentBackendPlacementDeviceSignature();
    observation.dtype = dtype;
    observation.shape_signature = shape_signature;
    observation.reason_code = reason_code;
    observation.source = source.empty()
        ? BackendPlacementObservationSource::RuntimeFallback
        : source;
    observation.detail = detail;
    observation.timestamp = CurrentTimestampUtc();
    RecordBackendPlacementObservation(observation);
}

bool TryGetBackendPlacementObservation(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature,
    BackendPlacementObservation& observation) {
    const std::string key = BuildObservationKey(
        op_type, backend, device, dtype, shape_signature);
    std::lock_guard<std::mutex> lock(g_observation_mutex);
    const auto it = g_observations.find(key);
    if (it == g_observations.end()) {
        return false;
    }
    observation = it->second;
    return true;
}

bool TryGetBackendPlacementObservationForActiveDevice(
    const std::string& op_type,
    const std::string& backend,
    const std::string& dtype,
    const std::string& shape_signature,
    BackendPlacementObservation& observation) {
    return TryGetBackendPlacementObservation(
        op_type,
        backend,
        CurrentBackendPlacementDeviceSignature(),
        dtype,
        shape_signature,
        observation);
}

std::vector<size_t> StripBatchDimensionForPlacementSignature(
    const std::vector<size_t>& runtime_shape) {
    if (runtime_shape.size() < 2) {
        return runtime_shape;
    }
    return std::vector<size_t>(runtime_shape.begin() + 1,
                               runtime_shape.end());
}

std::string BuildBackendPlacementObservationKey(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature) {
    return BuildObservationKey(op_type, backend, device, dtype,
                               shape_signature);
}

std::vector<BackendPlacementObservation>
SnapshotBackendPlacementObservations() {
    std::vector<BackendPlacementObservation> snapshot;
    {
        std::lock_guard<std::mutex> lock(g_observation_mutex);
        snapshot.reserve(g_observations.size());
        for (const auto& kv : g_observations) {
            snapshot.push_back(kv.second);
        }
    }
    std::sort(
        snapshot.begin(),
        snapshot.end(),
        [](const BackendPlacementObservation& lhs,
           const BackendPlacementObservation& rhs) {
            return std::tie(lhs.timestamp,
                            lhs.op_type,
                            lhs.backend,
                            lhs.device,
                            lhs.dtype,
                            lhs.shape_signature,
                            lhs.reason_code,
                            lhs.source) <
                   std::tie(rhs.timestamp,
                            rhs.op_type,
                            rhs.backend,
                            rhs.device,
                            rhs.dtype,
                            rhs.shape_signature,
                            rhs.reason_code,
                            rhs.source);
        });
    return snapshot;
}

bool SaveBackendPlacementObservationCache(
    const std::string& path,
    std::string* error_message) {
    try {
        nlohmann::json observations = nlohmann::json::array();
        {
            std::lock_guard<std::mutex> lock(g_observation_mutex);
            for (const auto& kv : g_observations) {
                observations.push_back(ObservationToJson(kv.second));
            }
        }

        nlohmann::json root = {
            {"schema_version", kPlacementObservationCacheSchemaVersion},
            {"cyxwiz_backend_version", CyxWizBackendVersionString()},
            {"arrayfire_backend", CurrentArrayFireBackendName()},
            {"saved_at", CurrentTimestampUtc()},
            {"observations", observations},
        };

        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) {
            SetError(error_message, "failed to open placement cache for writing: " + path);
            return false;
        }
        file << root.dump(2);
        file << "\n";
        return true;
    } catch (const std::exception& e) {
        SetError(error_message, e.what());
        return false;
    } catch (...) {
        SetError(error_message, "unknown error while saving placement cache");
        return false;
    }
}

bool LoadBackendPlacementObservationCache(
    const std::string& path,
    std::string* error_message) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            SetError(error_message, "failed to open placement cache for reading: " + path);
            return false;
        }
        nlohmann::json root;
        file >> root;
        if (!root.is_object()) {
            SetError(error_message, "placement cache root must be a JSON object");
            return false;
        }
        const int schema_version =
            root.value("schema_version", 0);
        if (schema_version != kPlacementObservationCacheSchemaVersion) {
            SetError(
                error_message,
                "unsupported placement cache schema_version: " +
                    std::to_string(schema_version));
            return false;
        }
        const auto observations_it = root.find("observations");
        if (observations_it == root.end() || !observations_it->is_array()) {
            SetError(error_message, "placement cache observations must be an array");
            return false;
        }

        std::vector<BackendPlacementObservation> loaded;
        for (const auto& entry : *observations_it) {
            BackendPlacementObservation observation;
            if (ObservationFromJson(entry, observation)) {
                loaded.push_back(std::move(observation));
            }
        }
        for (const BackendPlacementObservation& observation : loaded) {
            RecordBackendPlacementObservation(observation);
        }
        return true;
    } catch (const std::exception& e) {
        SetError(error_message, e.what());
        return false;
    } catch (...) {
        SetError(error_message, "unknown error while loading placement cache");
        return false;
    }
}

void ClearBackendPlacementObservationCacheForTesting() {
    std::lock_guard<std::mutex> lock(g_observation_mutex);
    g_observations.clear();
}

} // namespace cyxwiz
