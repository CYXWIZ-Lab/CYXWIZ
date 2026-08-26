#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <arrayfire.h>

#include <nlohmann/json.hpp>

#include <cyxwiz/device.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>

#include "core/route_qualification_snapshot.h"
#include "route_probe_dropout_contract.h"
#include "route_probe_flatten_contract.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <tlhelp32.h>
#include <windows.h>
#endif

namespace {

constexpr std::string_view kRequiredOperations[] = {
#define CYXWIZ_ROUTE_OPERATION(name) #name,
#include "core/arrayfire_route_operations.def"
#undef CYXWIZ_ROUTE_OPERATION
};

constexpr int kUnavailableExitCode = 77;
constexpr std::string_view kDenseComputeBenchmark =
    "dense_compute_benchmark";

struct ProbeOptions {
  af::Backend backend = AF_BACKEND_ONEAPI;
  cyxwiz::DeviceType device_type = cyxwiz::DeviceType::ONEAPI;
  std::string backend_name = "oneapi";
  int device_id = 0;
  std::string operation;
  bool enumerate_backend = false;
};

std::string active_backend_name = "oneapi";
int active_device_id = 0;
std::string active_operation = "unknown";
std::string active_probe_stage = "argument_parse";

void ConfigureIsolatedProbeFailureMode() {
#ifdef _WIN32
  const UINT previous_mode = SetErrorMode(SEM_FAILCRITICALERRORS |
                                          SEM_NOGPFAULTERRORBOX);
  SetErrorMode(previous_mode | SEM_FAILCRITICALERRORS |
               SEM_NOGPFAULTERRORBOX);
#endif
}

void Stage(const std::string &operation, const char *stage) {
  active_operation = operation;
  active_probe_stage = stage;
  std::cout << "probe_event schema=1 backend=" << active_backend_name
            << " device_id=" << active_device_id
            << " operation=" << operation << " stage=" << stage
            << std::endl;
}

ProbeOptions ParseOptions(int argc, char **argv) {
  ProbeOptions options;
  if (argc == 2) {
    options.operation = argv[1];
    return options;
  }
  const auto select_backend = [&](const std::string &value) {
    options.backend_name = value;
    if (value == "cpu") {
      options.backend = AF_BACKEND_CPU;
      options.device_type = cyxwiz::DeviceType::CPU;
    } else if (value == "cuda") {
      options.backend = AF_BACKEND_CUDA;
      options.device_type = cyxwiz::DeviceType::CUDA;
    } else if (value == "opencl") {
      options.backend = AF_BACKEND_OPENCL;
      options.device_type = cyxwiz::DeviceType::OPENCL;
    } else if (value == "oneapi") {
      options.backend = AF_BACKEND_ONEAPI;
      options.device_type = cyxwiz::DeviceType::ONEAPI;
    } else {
      throw std::invalid_argument("unknown backend: " + value);
    }
  };
  if (argc == 3 && std::string(argv[1]) == "--enumerate-backend") {
    select_backend(argv[2]);
    options.operation = "route_inventory";
    options.enumerate_backend = true;
    return options;
  }
  if (argc != 7) {
    throw std::invalid_argument(
        "usage: test_oneapi_operation_probe <operation> OR "
        "--backend cpu|cuda|opencl|oneapi --device <id> --operation <name> OR "
        "--enumerate-backend cpu|cuda|opencl|oneapi");
  }

  for (int index = 1; index < argc; index += 2) {
    const std::string key = argv[index];
    const std::string value = argv[index + 1];
    if (key == "--backend") {
      select_backend(value);
    } else if (key == "--device") {
      options.device_id = std::stoi(value);
      if (options.device_id < 0) {
        throw std::invalid_argument("device id must be non-negative");
      }
    } else if (key == "--operation") {
      options.operation = value;
    } else {
      throw std::invalid_argument("unknown option: " + key);
    }
  }
  if (options.operation.empty()) {
    throw std::invalid_argument("operation must not be empty");
  }
  if (std::find(kRequiredOperations, std::end(kRequiredOperations),
                options.operation) == std::end(kRequiredOperations)) {
    if (options.operation != kDenseComputeBenchmark) {
      throw std::invalid_argument(
          "operation is not in the released qualification manifest: " +
          options.operation);
    }
  }
  return options;
}

int EnumerateBackendRoutes(const ProbeOptions &options) {
  Stage(options.operation, "backend_load_begin");
  af::setBackend(options.backend);
  Stage(options.operation, "backend_load_complete");
  const int count = af::getDeviceCount();
  if (count < 0 || count > 64) {
    throw std::runtime_error("backend returned an invalid device count");
  }
  nlohmann::json routes = nlohmann::json::array();
  for (int device_id = 0; device_id < count; ++device_id) {
    const auto info =
        cyxwiz::Device(options.device_type, device_id).GetInfo();
    const auto optional_text = [](bool known, const std::string &value) {
      return known ? nlohmann::json(value) : nlohmann::json(nullptr);
    };
    routes.push_back({
        {"device_id", device_id},
        {"name", optional_text(
                     info.name_known && !info.name_is_fallback, info.name)},
        {"kind", cyxwiz::DeviceKindName(info.kind)},
        {"identity_confidence",
         cyxwiz::DeviceIdentityConfidenceName(info.identity_confidence)},
        {"provider", optional_text(info.provider_known, info.provider)},
        {"driver_version",
         optional_text(info.driver_version_known, info.driver_version)},
        {"physical_fingerprint",
         optional_text(
             info.physical_fingerprint_known, info.physical_fingerprint)},
        {"metadata_status",
         cyxwiz::DeviceMetadataStatusName(info.metadata_status)},
        {"metadata_error_code", info.metadata_error_code},
        {"metadata_message",
         info.metadata_message.empty()
             ? nlohmann::json(nullptr)
             : nlohmann::json(info.metadata_message)}});
  }
  Stage(options.operation, "enumeration_complete");
  std::cout << "route_inventory_json="
            << nlohmann::json({
                   {"schema_version", 1},
                   {"backend", options.backend_name},
                   {"routes", std::move(routes)}})
                   .dump()
            << std::endl;
  return 0;
}

#ifdef _WIN32
#ifdef UNICODE
std::string Narrow(const wchar_t *value) {
  const int size =
      WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
  if (size <= 1)
    return {};
  std::string result(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr,
                      nullptr);
  result.pop_back();
  return result;
}
#else
std::string Narrow(const char *value) { return value; }
#endif

bool IsRelevantRuntimeModule(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  static const char *needles[] = {
      "af.dll", "afcpu", "afoneapi", "sycl", "ur_", "opencl",
      "intelocl", "igdrcl", "ze_loader", "mkl_rt", "mkl_sycl", "igc"};
  for (const char *needle : needles) {
    if (name.find(needle) != std::string::npos)
      return true;
  }
  return false;
}

void PrintLoadedRuntimeModules(const char *point) {
  HANDLE snapshot = CreateToolhelp32Snapshot(
      TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
  if (snapshot == INVALID_HANDLE_VALUE)
    return;

  MODULEENTRY32 entry{};
  entry.dwSize = sizeof(entry);
  if (Module32First(snapshot, &entry)) {
    do {
      const std::string module_name = Narrow(entry.szModule);
      if (IsRelevantRuntimeModule(module_name)) {
        std::cout << "runtime_module point=" << point << " name=" << module_name
                  << " path='" << Narrow(entry.szExePath) << "'" << std::endl;
      }
    } while (Module32Next(snapshot, &entry));
  }
  CloseHandle(snapshot);
}
#else
void PrintLoadedRuntimeModules(const char *) {}
#endif

af::array InputVector() {
  const float values[] = {-2.0f, -0.5f, 0.5f, 2.0f};
  return af::array(4, values);
}

void EvaluateAndRead(const std::string &operation, const af::array &value) {
  Stage(operation, "eval_begin");
  value.eval();
  Stage(operation, "eval_complete");
  af::sync();
  Stage(operation, "sync_complete");

  std::vector<float> host(static_cast<size_t>(value.elements()));
  value.host(host.data());
  Stage(operation, "host_read_complete");
  for (float item : host) {
    if (!std::isfinite(item)) {
      throw std::runtime_error("operation produced a non-finite value");
    }
  }
}

void RunDirectBceForward(const std::string &operation) {
  const float logits_data[] = {0.0f, 0.0f};
  const float target_data[] = {1.0f, 0.0f};
  af::array logits(2, logits_data);
  af::array target(2, target_data);
  const float pos_weight = 4.0f;

  af::array log_weight = 1.0f + (pos_weight - 1.0f) * target;
  af::array softplus_negative =
      af::max(-logits, 0.0f) + af::log(1.0f + af::exp(-af::abs(logits)));
  af::array loss = (1.0f - target) * logits + log_weight * softplus_negative;
  loss.eval();
  Stage(operation, "expression_eval_complete");
  EvaluateAndRead(operation, af::mean(loss));
}

void RunDirectBceBackward(const std::string &operation) {
  const float logits_data[] = {0.0f, 0.0f};
  const float target_data[] = {1.0f, 0.0f};
  af::array logits(2, logits_data);
  af::array target(2, target_data);
  const float pos_weight = 4.0f;

  af::array log_weight = 1.0f + (pos_weight - 1.0f) * target;
  af::array grad = (1.0f - target) + log_weight * (af::sigmoid(logits) - 1.0f);
  EvaluateAndRead(operation, grad / static_cast<float>(logits.elements()));
}

void RunDenseComputeBenchmark(const std::string &operation) {
  constexpr int kDimension = 512;
  constexpr int kWarmupIterations = 2;
  constexpr int kSamples = 5;
  constexpr int kIterationsPerSample = 3;

  Stage(operation, "input_create_begin");
  af::array activations =
      af::constant(0.001f, kDimension, kDimension, f32);
  af::array weights =
      af::constant(0.002f, kDimension, kDimension, f32);
  activations.eval();
  weights.eval();
  af::sync();
  Stage(operation, "input_create_complete");

  const auto iteration = [&] {
    af::array forward = af::sigmoid(af::matmul(activations, weights));
    af::array backward =
        af::matmul(forward, weights, AF_MAT_NONE, AF_MAT_TRANS);
    af::array reduced = af::sum(backward);
    backward.eval();
    reduced.eval();
    af::sync();
  };

  Stage(operation, "warmup_begin");
  for (int index = 0; index < kWarmupIterations; ++index) iteration();
  Stage(operation, "warmup_complete");

  std::array<double, kSamples> milliseconds{};
  Stage(operation, "measurement_begin");
  for (int sample = 0; sample < kSamples; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < kIterationsPerSample; ++index) iteration();
    const auto stopped = std::chrono::steady_clock::now();
    milliseconds[static_cast<size_t>(sample)] =
        std::chrono::duration<double, std::milli>(stopped - started).count() /
        static_cast<double>(kIterationsPerSample);
  }
  Stage(operation, "measurement_complete");
  std::sort(milliseconds.begin(), milliseconds.end());
  const double median_ms = milliseconds[milliseconds.size() / 2];
  if (!(median_ms > 0.0) || !std::isfinite(median_ms)) {
    throw std::runtime_error("dense compute benchmark produced invalid timing");
  }

  std::cout << std::fixed << std::setprecision(6)
            << "benchmark_result schema=1 backend=" << active_backend_name
            << " device_id=" << active_device_id
            << " benchmark_id=" << cyxwiz::kRoutePerformanceBenchmarkId
            << " samples=" << kSamples
            << " iterations_per_sample=" << kIterationsPerSample
            << " matrix_dimension=" << kDimension
            << " median_iteration_ms=" << median_ms << std::endl;
}

void RunOperation(const ProbeOptions &options) {
  const std::string &operation = options.operation;
  if (operation == kDenseComputeBenchmark) {
    RunDenseComputeBenchmark(operation);
    return;
  }
  if (operation == "route_metadata") {
    const auto info =
        cyxwiz::Device(options.device_type, options.device_id).GetInfo();
    std::cout << "route_metadata backend=" << options.backend_name
              << " device_id=" << options.device_id
              << " name='" << info.name << "'"
              << " name_known=" << (!info.name_is_fallback ? "true" : "false")
              << " kind=" << cyxwiz::DeviceKindName(info.kind)
              << " identity_confidence="
              << cyxwiz::DeviceIdentityConfidenceName(
                     info.identity_confidence)
              << " provider='"
              << (info.provider_known ? info.provider : "unknown") << "'"
              << " driver='"
              << (info.driver_version_known ? info.driver_version : "unknown")
              << "' fingerprint="
              << (info.physical_fingerprint_known
                      ? info.physical_fingerprint
                      : "unknown")
              << " metadata="
              << cyxwiz::DeviceMetadataStatusName(info.metadata_status)
              << std::endl;
    return;
  }
  if (operation == "device_info") {
    char name[256] = {};
    char platform[256] = {};
    char toolkit[256] = {};
    char compute[256] = {};
    try {
      af::deviceInfo(name, platform, toolkit, compute);
      std::cout << "device_info name='" << name << "' platform='" << platform
                << "' toolkit='" << toolkit << "' compute='" << compute << "'"
                << std::endl;
    } catch (af::exception &error) {
      std::cout << "device_info status=unsupported_or_failed error="
                << static_cast<int>(error.err()) << std::endl;
    }
    return;
  }
  if (operation == "constant") {
    EvaluateAndRead(operation, af::constant(1.0f, 4, f32) + 2.0f);
    return;
  }
  if (operation == "randu") {
    Stage(operation, "create_begin");
    af::array value = af::randu(2, 2, f32);
    Stage(operation, "create_complete");
    EvaluateAndRead(operation, value);
    return;
  }
  if (operation == "randu_scaled") {
    Stage(operation, "create_begin");
    af::array value = (af::randu(2, 2, f32) * 2.0f - 1.0f) * 0.5f;
    Stage(operation, "create_complete");
    EvaluateAndRead(operation, value);
    return;
  }
  if (operation == "abs") {
    EvaluateAndRead(operation, af::abs(InputVector()));
    return;
  }
  if (operation == "exp") {
    EvaluateAndRead(operation, af::exp(InputVector()));
    return;
  }
  if (operation == "log") {
    EvaluateAndRead(operation, af::log(af::abs(InputVector()) + 1.0f));
    return;
  }
  if (operation == "maximum") {
    EvaluateAndRead(operation, af::max(InputVector(), 0.0f));
    return;
  }
  if (operation == "sum") {
    Stage(operation, "expression_begin");
    EvaluateAndRead(operation, af::sum(InputVector()));
    return;
  }
  if (operation == "mean") {
    Stage(operation, "expression_begin");
    EvaluateAndRead(operation, af::mean(InputVector()));
    return;
  }
  if (operation == "sigmoid") {
    EvaluateAndRead(operation, af::sigmoid(InputVector()));
    return;
  }
  if (operation == "matmul") {
    Stage(operation, "input_create_begin");
    af::array left = af::constant(2.0f, 2, 2, f32);
    af::array right = af::constant(1.0f, 2, 2, f32);
    Stage(operation, "input_create_complete");
    EvaluateAndRead(operation, af::matmul(left, right));
    return;
  }
  if (operation == "identity") {
    Stage(operation, "create_begin");
    af::array value = af::identity(2, 2, f32);
    Stage(operation, "create_complete");
    EvaluateAndRead(operation, value);
    return;
  }
  if (operation == "transpose") {
    Stage(operation, "input_create_begin");
    af::array input = af::constant(1.0f, 2, 2, f32);
    Stage(operation, "input_create_complete");
    EvaluateAndRead(operation, af::transpose(input));
    return;
  }
  if (operation == "bce_forward_expression") {
    RunDirectBceForward(operation);
    return;
  }
  if (operation == "bce_backward_expression") {
    RunDirectBceBackward(operation);
    return;
  }
  if (operation == "tensor_row_major") {
    Stage(operation, "tensor_create_begin");
    cyxwiz::Tensor tensor =
        cyxwiz::Tensor::FromArrayRowMajor2D(af::constant(1.0f, 2, 2, f32));
    Stage(operation, "tensor_create_complete");
    Stage(operation, "tensor_read_begin");
    const float *data = tensor.ReadData<float>();
    if (data == nullptr || data[0] != 1.0f) {
      throw std::runtime_error("Tensor row-major conversion failed");
    }
    Stage(operation, "tensor_read_complete");
    return;
  }
  if (operation == "cyxwiz_bce_forward" || operation == "cyxwiz_bce_backward") {
    const float logits_data[] = {0.0f, 0.0f};
    const float target_data[] = {1.0f, 0.0f};
    cyxwiz::Tensor logits({2}, logits_data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, target_data, cyxwiz::DataType::Float32);
    cyxwiz::BCEWithLogitsLoss loss(cyxwiz::Reduction::Mean, 4.0f);
    Stage(operation, "loss_call_begin");
    cyxwiz::Tensor result = operation == "cyxwiz_bce_forward"
                                ? loss.Forward(logits, targets)
                                : loss.Backward(logits, targets);
    Stage(operation, "loss_call_complete");
    Stage(operation, "tensor_read_begin");
    const float *data = result.ReadData<float>();
    if (data == nullptr || !std::isfinite(data[0])) {
      throw std::runtime_error("CyxWiz BCE produced invalid output");
    }
    Stage(operation, "tensor_read_complete");
    return;
  }
  if (operation == "linear_init") {
    Stage(operation, "constructor_begin");
    cyxwiz::LinearLayer layer(2, 2, true);
    Stage(operation, "constructor_complete");
    return;
  }
  if (operation == "cyxwiz_flatten_forward_backward") {
    cyxwiz::route_probe::RunFlattenForwardBackwardContract(operation, &Stage);
    return;
  }
  if (operation == "cyxwiz_dropout_forward_backward") {
    cyxwiz::route_probe::RunDropoutForwardBackwardContract(operation, &Stage);
    return;
  }

  throw std::invalid_argument("unknown operation: " + operation);
}

} // namespace

int main(int argc, char **argv) {
  ConfigureIsolatedProbeFailureMode();
  try {
    const ProbeOptions options = ParseOptions(argc, argv);
    active_backend_name = options.backend_name;
    active_device_id = options.device_id;
    active_operation = options.operation;
    if (options.enumerate_backend) {
      return EnumerateBackendRoutes(options);
    }
    Stage(options.operation, "backend_load_begin");
    af::setBackend(options.backend);
    Stage(options.operation, "backend_load_complete");
    Stage(options.operation, "enumeration_begin");
    const int count = af::getDeviceCount();
    std::cout << "probe_event schema=1 backend=" << options.backend_name
              << " device_id=" << options.device_id
              << " operation=" << options.operation
              << " stage=enumeration_complete count=" << count << std::endl;
    if (count <= options.device_id) {
      std::cout << "probe_result schema=1 backend=" << options.backend_name
                << " device_id=" << options.device_id
                << " operation=" << options.operation
                << " status=unavailable reason=device_not_enumerated count="
                << count << std::endl;
      return kUnavailableExitCode;
    }
    Stage(options.operation, "activation_begin");
    af::setDevice(options.device_id);
    int major = 0;
    int minor = 0;
    int patch = 0;
    const af_err version_error = af_get_version(&major, &minor, &patch);
    if (version_error != AF_SUCCESS) {
      major = 0;
      minor = 0;
      patch = 0;
    }
    std::cout << "probe_event schema=1 backend=" << options.backend_name
              << " device_id=" << options.device_id
              << " operation=" << options.operation
              << " stage=activation_complete count=" << count
              << " effective_backend="
              << static_cast<int>(af::getActiveBackend())
              << " effective_device=" << af::getDevice()
              << " arrayfire_version=" << major << '.' << minor << '.'
              << patch << std::endl;
    const bool inspect_runtime = options.operation == "constant";
    if (inspect_runtime)
      PrintLoadedRuntimeModules("selected");

    RunOperation(options);
    if (inspect_runtime)
      PrintLoadedRuntimeModules("completed");
    std::cout << "probe_result schema=1 backend=" << options.backend_name
              << " device_id=" << options.device_id
              << " operation=" << options.operation
              << " status=pass effective_backend="
              << static_cast<int>(af::getActiveBackend())
              << " effective_device=" << af::getDevice() << std::endl;
    return 0;
  } catch (af::exception &error) {
    std::cerr << "probe_result schema=1 backend=" << active_backend_name
              << " device_id=" << active_device_id
              << " operation=" << active_operation
              << " status=arrayfire_error failure_stage="
              << active_probe_stage << " code="
              << static_cast<int>(error.err()) << " message='" << error.what()
              << "'" << std::endl;
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "probe_result schema=1 backend=" << active_backend_name
              << " device_id=" << active_device_id
              << " operation=" << active_operation
              << " status=error failure_stage=" << active_probe_stage
              << " message='" << error.what() << "'"
              << std::endl;
    return argc == 2 || argc == 7 ? 3 : 64;
  }
}
