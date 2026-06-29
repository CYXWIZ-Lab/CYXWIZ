#include "cyxwiz/backend_placement_observation.h"

#include "arrayfire_backend_utils.h"

#include <mutex>
#include <sstream>
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

std::string BuildObservationKey(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature) {
    return op_type + "|" + backend + "|" + device + "|" + dtype + "|" +
           shape_signature;
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
    const std::string key = BuildObservationKey(
        RecurrentKindName(request.kind),
        "cuda",
        CurrentBackendPlacementDeviceSignature(),
        "float32",
        BuildRecurrentCudaPlacementShapeSignature(request));
    std::lock_guard<std::mutex> lock(g_probe_attempt_mutex);
    return g_probe_attempts.emplace(key, true).second;
}

void RunLstmSingleStepProbe(const RecurrentCudaPlacementRequest& request) {
    const unsigned batch = static_cast<unsigned>(request.batch_size);
    const unsigned input = static_cast<unsigned>(request.input_size);
    const unsigned hidden = static_cast<unsigned>(request.hidden_size);
    const unsigned gates = 4 * hidden;

    af::array x_t = af::randu(af::dim4(batch, input), af::dtype::f32);
    af::array h = af::randu(af::dim4(batch, hidden), af::dtype::f32);
    af::array c = af::randu(af::dim4(batch, hidden), af::dtype::f32);
    af::array w_ih = af::randu(af::dim4(gates, input), af::dtype::f32);
    af::array w_hh = af::randu(af::dim4(gates, hidden), af::dtype::f32);
    af::array b_ih = af::constant(0.0f, af::dim4(gates));
    af::array b_hh = af::constant(0.0f, af::dim4(gates));

    af::array gates_t =
        af::matmul(x_t, af::transpose(w_ih)) +
        af::tile(af::transpose(b_ih), batch) +
        af::matmul(h, af::transpose(w_hh)) +
        af::tile(af::transpose(b_hh), batch);
    af::array i_gate = af::sigmoid(gates_t(af::span, af::seq(0, hidden - 1)));
    af::array f_gate =
        af::sigmoid(gates_t(af::span, af::seq(hidden, 2 * hidden - 1)));
    af::array g_gate =
        af::tanh(gates_t(af::span, af::seq(2 * hidden, 3 * hidden - 1)));
    af::array o_gate =
        af::sigmoid(gates_t(af::span, af::seq(3 * hidden, gates - 1)));
    af::array c_next = f_gate * c + i_gate * g_gate;
    af::array h_next = o_gate * af::tanh(c_next);
    h_next.eval();
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

std::string BuildTensorLayerPlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& output_shape) {
    std::ostringstream out;
    out << "input=[";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i > 0) {
            out << "x";
        }
        out << input_shape[i];
    }
    out << "];output=[";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (i > 0) {
            out << "x";
        }
        out << output_shape[i];
    }
    out << "]";
    return out.str();
}

void RecordBackendPlacementObservation(
    const BackendPlacementObservation& observation) {
    const std::string key = BuildObservationKey(
        observation.op_type,
        observation.backend,
        observation.device,
        observation.dtype,
        observation.shape_signature);
    std::lock_guard<std::mutex> lock(g_observation_mutex);
    g_observations[key] = observation;
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
    RecordRecurrentCudaPlacementObservation(
        request,
        reason_code,
        BackendPlacementObservationSource::PreflightProbe,
        detail);
}

bool TryRunRecurrentCudaPreflightProbe(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& failure_observation) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (request.kind != RecurrentLayerKind::LSTM ||
        request.bidirectional ||
        !IsBoundedRecurrentProbeShape(request)) {
        return false;
    }

    try {
        if (af::getActiveBackend() != AF_BACKEND_CUDA) {
            return false;
        }
        if (!MarkRecurrentProbeAttemptedOnce(request)) {
            return false;
        }
        RunLstmSingleStepProbe(request);
        return false;
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
        return TryGetRecurrentCudaPlacementObservation(
            request,
            failure_observation);
    } catch (const std::exception& e) {
        std::string detail =
            "ArrayFire CUDA recurrent preflight probe failed for exact "
            "backend/device/dtype/shape. Error: ";
        detail += e.what();
        RecordRecurrentCudaPreflightProbeFailure(
            request,
            BackendPlacementObservationReason::GpuBackendException,
            detail);
        return TryGetRecurrentCudaPlacementObservation(
            request,
            failure_observation);
    } catch (...) {
        RecordRecurrentCudaPreflightProbeFailure(
            request,
            BackendPlacementObservationReason::GpuBackendException,
            "ArrayFire CUDA recurrent preflight probe failed for exact "
            "backend/device/dtype/shape with an unknown exception.");
        return TryGetRecurrentCudaPlacementObservation(
            request,
            failure_observation);
    }
#else
    (void)request;
    (void)failure_observation;
    return false;
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
