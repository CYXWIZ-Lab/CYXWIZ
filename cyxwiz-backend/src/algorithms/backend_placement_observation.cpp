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
std::mutex g_probe_attempt_mutex;
std::unordered_map<std::string, bool> g_probe_attempts;

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

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool IsBoundedRecurrentProbeShape(
    const RecurrentCudaPlacementRequest& request) {
    return request.batch_size > 0 &&
           request.seq_len > 0 &&
           request.input_size > 0 &&
           request.hidden_size > 0 &&
           request.batch_size <= 256 &&
           request.seq_len <= 256 &&
           request.input_size <= 4096 &&
           request.hidden_size <= 512 &&
           request.num_layers <= 4;
}

bool MarkRecurrentProbeAttemptedOnce(
    const RecurrentCudaPlacementRequest& request) {
    std::string probe_signature = BuildRecurrentCudaPlacementShapeSignature(
        request);
    probe_signature += request.deep_preflight
        ? ";probe_mode=deep"
        : ";probe_mode=single";
    const std::string key = BuildObservationKey(
        RecurrentKindName(request.kind),
        "cuda",
        CurrentBackendPlacementDeviceSignature(),
        "float32",
        probe_signature);
    std::lock_guard<std::mutex> lock(g_probe_attempt_mutex);
    return g_probe_attempts.emplace(key, true).second;
}

size_t RecurrentProbeStepCount(const RecurrentCudaPlacementRequest& request) {
    return request.deep_preflight ? request.seq_len : 1;
}

void RunLstmProbe(const RecurrentCudaPlacementRequest& request,
                  size_t step_count) {
    const unsigned batch = static_cast<unsigned>(request.batch_size);
    const unsigned input = static_cast<unsigned>(request.input_size);
    const unsigned hidden = static_cast<unsigned>(request.hidden_size);
    const unsigned gates = 4 * hidden;

    af::array h = af::randu(af::dim4(batch, hidden), af::dtype::f32);
    af::array c = af::randu(af::dim4(batch, hidden), af::dtype::f32);
    af::array w_ih = af::randu(af::dim4(gates, input), af::dtype::f32);
    af::array w_hh = af::randu(af::dim4(gates, hidden), af::dtype::f32);
    af::array b_ih = af::constant(0.0f, af::dim4(gates));
    af::array b_hh = af::constant(0.0f, af::dim4(gates));

    for (size_t step = 0; step < step_count; ++step) {
        af::array x_t = af::randu(af::dim4(batch, input), af::dtype::f32);
        af::array gates_t =
            af::matmul(x_t, af::transpose(w_ih)) +
            af::tile(af::transpose(b_ih), batch) +
            af::matmul(h, af::transpose(w_hh)) +
            af::tile(af::transpose(b_hh), batch);
        af::array i_gate =
            af::sigmoid(gates_t(af::span, af::seq(0, hidden - 1)));
        af::array f_gate =
            af::sigmoid(gates_t(af::span, af::seq(hidden, 2 * hidden - 1)));
        af::array g_gate =
            af::tanh(gates_t(af::span, af::seq(2 * hidden, 3 * hidden - 1)));
        af::array o_gate =
            af::sigmoid(gates_t(af::span, af::seq(3 * hidden, gates - 1)));
        c = f_gate * c + i_gate * g_gate;
        h = o_gate * af::tanh(c);
        h.eval();
    }
    af::sync();
}

void RunGruProbe(const RecurrentCudaPlacementRequest& request,
                 size_t step_count) {
    const unsigned batch = static_cast<unsigned>(request.batch_size);
    const unsigned input = static_cast<unsigned>(request.input_size);
    const unsigned hidden = static_cast<unsigned>(request.hidden_size);
    const unsigned gates = 3 * hidden;

    af::array h = af::randu(af::dim4(batch, hidden), af::dtype::f32);
    af::array w_ih = af::randu(af::dim4(gates, input), af::dtype::f32);
    af::array w_hh = af::randu(af::dim4(gates, hidden), af::dtype::f32);
    af::array b_ih = af::constant(0.0f, af::dim4(gates));
    af::array b_hh = af::constant(0.0f, af::dim4(gates));

    for (size_t step = 0; step < step_count; ++step) {
        af::array x_t = af::randu(af::dim4(batch, input), af::dtype::f32);
        af::array gates_x =
            af::matmul(x_t, af::transpose(w_ih)) +
            af::tile(af::transpose(b_ih), batch);
        af::array gates_h =
            af::matmul(h, af::transpose(w_hh)) +
            af::tile(af::transpose(b_hh), batch);
        af::array x_z = gates_x(af::span, af::seq(0, hidden - 1));
        af::array x_r = gates_x(af::span, af::seq(hidden, 2 * hidden - 1));
        af::array x_n = gates_x(af::span, af::seq(2 * hidden, gates - 1));
        af::array h_z = gates_h(af::span, af::seq(0, hidden - 1));
        af::array h_r = gates_h(af::span, af::seq(hidden, 2 * hidden - 1));
        af::array h_n = gates_h(af::span, af::seq(2 * hidden, gates - 1));

        af::array z_gate = af::sigmoid(x_z + h_z);
        af::array r_gate = af::sigmoid(x_r + h_r);
        af::array n_gate = af::tanh(x_n + r_gate * h_n);
        af::array one = af::constant(1.0f, z_gate.dims(), af::dtype::f32);
        h = (one - z_gate) * n_gate + z_gate * h;
        h.eval();
    }
    af::sync();
}
#endif

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

std::string BuildRecurrentCudaPlacementShapeSignature(
    const RecurrentCudaPlacementRequest& request) {
    std::ostringstream out;
    out << "kind=" << RecurrentKindName(request.kind)
        << ";batch=" << request.batch_size
        << ";seq=" << request.seq_len
        << ";input=" << request.input_size
        << ";hidden=" << request.hidden_size
        << ";layers=" << request.num_layers
        << ";bidirectional=" << (request.bidirectional ? "true" : "false")
        << ";return_sequences="
        << (request.return_sequences ? "true" : "false");
    return out.str();
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
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& output_shape) {
    std::ostringstream out;
    out << "input=";
    AppendShape(out, input_shape);
    out << ";output=";
    AppendShape(out, output_shape);
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

void RecordRecurrentCudaPlacementObservation(
    const RecurrentCudaPlacementRequest& request,
    const std::string& reason_code,
    const std::string& source,
    const std::string& detail) {
    RecordBackendPlacementObservationForActiveDevice(
        RecurrentKindName(request.kind),
        "cuda",
        "float32",
        BuildRecurrentCudaPlacementShapeSignature(request),
        reason_code,
        source,
        detail);
}

void RecordRecurrentCudaPreflightProbeFailure(
    const RecurrentCudaPlacementRequest& request,
    const std::string& reason_code,
    const std::string& detail) {
    BackendPlacementObservation observation;
    observation.op_type = RecurrentKindName(request.kind);
    observation.backend = "cuda";
    observation.device = CurrentBackendPlacementDeviceSignature();
    observation.dtype = "float32";
    observation.shape_signature =
        BuildRecurrentCudaPlacementShapeSignature(request);
    observation.reason_code = reason_code;
    observation.source = BackendPlacementObservationSource::PreflightProbe;
    observation.detail = detail;
    observation.timestamp = CurrentTimestampUtc();
    observation.probe_outcome =
        reason_code == BackendPlacementObservationReason::BackendCompileTimeout
            ? BackendPlacementProbeOutcomeName(BackendPlacementProbeOutcome::Timeout)
            : BackendPlacementProbeOutcomeName(BackendPlacementProbeOutcome::Unsafe);
    observation.probe_scope = request.deep_preflight
        ? BackendPlacementProbeScope::DeepPreflight
        : BackendPlacementProbeScope::NormalCompile;
    RecordBackendPlacementObservation(observation);
}

bool TryRunRecurrentCudaPreflightProbe(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& failure_observation) {
    const BackendPlacementProbeResult result =
        RunRecurrentCudaPreflightProbe(request);
    if ((result.outcome == BackendPlacementProbeOutcome::Unsafe ||
         result.outcome == BackendPlacementProbeOutcome::Timeout) &&
        result.has_observation) {
        failure_observation = result.observation;
        return true;
    }
    return false;
}

BackendPlacementProbeResult RunRecurrentCudaPreflightProbe(
    const RecurrentCudaPlacementRequest& request) {
    BackendPlacementProbeResult result;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (request.kind != RecurrentLayerKind::LSTM &&
        request.kind != RecurrentLayerKind::GRU) {
        result.outcome = BackendPlacementProbeOutcome::Unsupported;
        result.reason_code =
            BackendPlacementObservationReason::UnsupportedShape;
        result.detail =
            "Recurrent CUDA preflight probe is currently implemented only for "
            "LSTM and GRU requests.";
        return result;
    }
    if (request.bidirectional) {
        result.outcome = BackendPlacementProbeOutcome::Unsupported;
        result.reason_code =
            BackendPlacementObservationReason::UnsupportedShape;
        result.detail =
            "Recurrent CUDA preflight probe is currently implemented only for "
            "single-direction LSTM and GRU requests.";
        return result;
    }
    if (!IsBoundedRecurrentProbeShape(request)) {
        result.outcome = BackendPlacementProbeOutcome::Unsupported;
        result.reason_code =
            BackendPlacementObservationReason::UnsupportedShape;
        result.detail =
            "Recurrent CUDA preflight probe request is outside the bounded "
            "normal-compile probe shape budget.";
        return result;
    }
    if (request.preflight_timeout_ms == 0) {
        const std::string detail =
            "Recurrent CUDA preflight probe skipped because timeout budget "
            "is 0 ms.";
        RecordRecurrentCudaPreflightProbeFailure(
            request,
            BackendPlacementObservationReason::BackendCompileTimeout,
            detail);
        result.outcome = BackendPlacementProbeOutcome::Timeout;
        result.reason_code =
            BackendPlacementObservationReason::BackendCompileTimeout;
        result.detail = detail;
        result.has_observation = TryGetRecurrentCudaPlacementObservation(
            request,
            result.observation);
        return result;
    }

    try {
        const auto probe_start = std::chrono::steady_clock::now();
        if (af::getActiveBackend() != AF_BACKEND_CUDA) {
            result.outcome = BackendPlacementProbeOutcome::Unsupported;
            result.reason_code =
                BackendPlacementObservationReason::UnsupportedShape;
            result.detail =
                "Recurrent CUDA preflight probe requires the ArrayFire CUDA "
                "backend to be active.";
            return result;
        }
        if (!MarkRecurrentProbeAttemptedOnce(request)) {
            result.outcome = BackendPlacementProbeOutcome::Inconclusive;
            result.detail =
                "Recurrent CUDA preflight probe was already attempted for "
                "this backend/device/dtype/shape/probe mode in this process.";
            return result;
        }
        const size_t step_count = RecurrentProbeStepCount(request);
        if (request.kind == RecurrentLayerKind::GRU) {
            RunGruProbe(request, step_count);
        } else {
            RunLstmProbe(request, step_count);
        }
        const auto probe_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - probe_start)
                .count();
        if (static_cast<size_t>(probe_elapsed) >
            request.preflight_timeout_ms) {
            std::string detail =
                "ArrayFire CUDA recurrent preflight probe exceeded timeout "
                "budget. elapsed_ms=";
            detail += std::to_string(probe_elapsed);
            detail += ";timeout_ms=";
            detail += std::to_string(request.preflight_timeout_ms);
            detail += ";probe_mode=";
            detail += request.deep_preflight ? "deep" : "single";
            detail += ";steps=";
            detail += std::to_string(step_count);
            detail += ".";
            RecordRecurrentCudaPreflightProbeFailure(
                request,
                BackendPlacementObservationReason::BackendCompileTimeout,
                detail);
            result.outcome = BackendPlacementProbeOutcome::Timeout;
            result.reason_code =
                BackendPlacementObservationReason::BackendCompileTimeout;
            result.detail = detail;
            result.has_observation = TryGetRecurrentCudaPlacementObservation(
                request,
                result.observation);
            return result;
        }
        result.outcome = BackendPlacementProbeOutcome::Safe;
        result.detail = "ArrayFire CUDA ";
        result.detail += RecurrentKindName(request.kind);
        result.detail +=
            " recurrent preflight probe completed for the exact "
            "backend/device/dtype/shape. This is not a full training safety "
            "proof. elapsed_ms=";
        result.detail += std::to_string(probe_elapsed);
        result.detail += ";timeout_ms=";
        result.detail += std::to_string(request.preflight_timeout_ms);
        result.detail += ";probe_mode=";
        result.detail += request.deep_preflight ? "deep" : "single";
        result.detail += ";steps=";
        result.detail += std::to_string(step_count);
        result.detail += ".";
        return result;
    } catch (const af::exception& e) {
        const BackendFallbackReason reason =
            ClassifyArrayFireBackendFallbackReason(e.what());
        const std::string reason_code = BackendFallbackReasonName(reason);
        std::string detail =
            "ArrayFire CUDA recurrent preflight probe failed for exact "
            "backend/device/dtype/shape. Error: ";
        detail += e.what();
        RecordRecurrentCudaPreflightProbeFailure(
            request,
            reason_code,
            detail);
        result.outcome = BackendPlacementProbeOutcome::Unsafe;
        result.reason_code = reason_code;
        result.detail = detail;
        result.has_observation = TryGetRecurrentCudaPlacementObservation(
            request,
            result.observation);
        return result;
    } catch (const std::exception& e) {
        std::string detail =
            "ArrayFire CUDA recurrent preflight probe failed for exact "
            "backend/device/dtype/shape. Error: ";
        detail += e.what();
        RecordRecurrentCudaPreflightProbeFailure(
            request,
            BackendPlacementObservationReason::GpuBackendException,
            detail);
        result.outcome = BackendPlacementProbeOutcome::Unsafe;
        result.reason_code =
            BackendPlacementObservationReason::GpuBackendException;
        result.detail = detail;
        result.has_observation = TryGetRecurrentCudaPlacementObservation(
            request,
            result.observation);
        return result;
    } catch (...) {
        const std::string detail =
            "ArrayFire CUDA recurrent preflight probe failed for exact "
            "backend/device/dtype/shape with an unknown exception.";
        RecordRecurrentCudaPreflightProbeFailure(
            request,
            BackendPlacementObservationReason::GpuBackendException,
            detail);
        result.outcome = BackendPlacementProbeOutcome::Unsafe;
        result.reason_code =
            BackendPlacementObservationReason::GpuBackendException;
        result.detail = detail;
        result.has_observation = TryGetRecurrentCudaPlacementObservation(
            request,
            result.observation);
        return result;
    }
#else
    (void)request;
    result.outcome = BackendPlacementProbeOutcome::Unsupported;
    result.reason_code = BackendPlacementObservationReason::UnsupportedShape;
    result.detail =
        "Recurrent CUDA preflight probe requires an ArrayFire-enabled build.";
    return result;
#endif
}

bool TryGetRecurrentCudaPlacementObservation(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& observation) {
    {
        std::lock_guard<std::mutex> lock(g_observation_mutex);
        if (g_observations.empty()) {
            return false;
        }
    }
    return TryGetBackendPlacementObservation(
        RecurrentKindName(request.kind),
        "cuda",
        CurrentBackendPlacementDeviceSignature(),
        "float32",
        BuildRecurrentCudaPlacementShapeSignature(request),
        observation);
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
    {
        std::lock_guard<std::mutex> lock(g_observation_mutex);
        g_observations.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_probe_attempt_mutex);
        g_probe_attempts.clear();
    }
}

} // namespace cyxwiz
