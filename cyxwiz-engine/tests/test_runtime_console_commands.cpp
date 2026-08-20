#include "../src/core/runtime_console_commands.h"
#include "../src/core/route_qualification_snapshot.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool ContainsLine(const cyxwiz::RuntimeConsoleCommandResult& result,
                  const std::string& text) {
    for (const auto& line : result.lines) {
        if (line.text.find(text) != std::string::npos) return true;
    }
    return false;
}

class FakeTruthProvider final : public cyxwiz::RuntimeTruthQueryProvider {
public:
    cyxwiz::RuntimeTrainingTruth current_training;
    cyxwiz::RuntimeTrainingTruth last_training;
    cyxwiz::RuntimeDeviceTruth device;
    cyxwiz::RuntimeBackendPackTruth backend_packs;
    cyxwiz::RuntimeRunTruth current_run;
    cyxwiz::RuntimeRunTruth run;
    mutable bool last_inventory_requested = false;

    cyxwiz::RuntimeTrainingTruth GetCurrentTraining() const override {
        return current_training;
    }

    cyxwiz::RuntimeTrainingTruth GetLastTraining() const override {
        return last_training;
    }

    cyxwiz::RuntimeDeviceTruth GetDeviceTruth(
        bool include_inventory) override {
        last_inventory_requested = include_inventory;
        auto truth = device;
        if (!include_inventory) truth.available_devices.clear();
        return truth;
    }

    cyxwiz::RuntimeBackendPackTruth GetBackendPackTruth() const override {
        return backend_packs;
    }

    cyxwiz::RuntimeRunTruth GetCurrentRun() const override {
        return current_run;
    }

    cyxwiz::RuntimeRunTruth GetRun(std::string_view) const override {
        return run;
    }
};

cyxwiz::RuntimeLogEvent MakeEvent(
    std::string category, cyxwiz::RuntimeLogLevel level,
    std::string message) {
    cyxwiz::RuntimeLogEvent event;
    event.timestamp_utc = std::chrono::system_clock::time_point{
        std::chrono::milliseconds{1704067200000}};
    event.category = std::move(category);
    event.level = level;
    event.source = "command_test";
    event.event_name = "fixture";
    event.message = std::move(message);
    return event;
}

void Populate(cyxwiz::RuntimeLogStore& store) {
    Check(store.Append(MakeEvent("system", cyxwiz::RuntimeLogLevel::Info,
                                 "startup")),
          "fixture append should succeed");

    auto training_warning = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Warning, "batch warning");
    training_warning.run_id = "run-1";
    training_warning.issue_codes = {"CW-G-0501"};
    Check(store.Append(std::move(training_warning)),
          "fixture append should succeed");

    auto device_error = MakeEvent(
        "device", cyxwiz::RuntimeLogLevel::Error, "native CPU fallback");
    device_error.run_id = "run-1";
    device_error.primary_error_code = "CW-G-0501";
    device_error.backend = "arrayfire_cuda";
    device_error.device_id = 0;
    Check(store.Append(std::move(device_error)),
          "fixture append should succeed");

    auto training_info = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Info, "batch complete");
    training_info.run_id = "run-2";
    Check(store.Append(std::move(training_info)),
          "fixture append should succeed");

    auto compiler_error = MakeEvent(
        "compiler", cyxwiz::RuntimeLogLevel::Error, "missing training path");
    compiler_error.primary_error_code = "CW-C-0101";
    compiler_error.node_id = 17;
    Check(store.Append(std::move(compiler_error)),
          "fixture append should succeed");

    auto data_warning = MakeEvent(
        "data", cyxwiz::RuntimeLogLevel::Warning, "column mismatch");
    data_warning.primary_error_code = "CW-D-0301";
    Check(store.Append(std::move(data_warning)),
          "fixture append should succeed");
}

void TestRegistryAndCompatibility(cyxwiz::RuntimeConsoleCommandService& service,
                                  const cyxwiz::RuntimeLogStore& store) {
    const auto help = service.Execute("help");
    Check(help.success && ContainsLine(help, "show logs|errors|code|codes") &&
              ContainsLine(help, "help <command>"),
          "help should be generated from registered command metadata");

    const auto show_help = service.Execute("help SHOW");
    Check(show_help.success && ContainsLine(show_help, "show logs where") &&
              ContainsLine(show_help, "show errors last") &&
              ContainsLine(show_help, "show codes family") &&
              ContainsLine(show_help,
                           "category=training and level>=warn"),
          "help show should provide forms, behavior, and examples");

    const auto filter_help = service.Execute("help filter");
    Check(filter_help.success && ContainsLine(filter_help, "filter save") &&
              ContainsLine(filter_help, "combined with subsequent") &&
              ContainsLine(filter_help, "error_code matches CW-G-*") &&
              ContainsLine(filter_help, "not is evaluated first") &&
              ContainsLine(filter_help, "Quote values containing spaces") &&
              ContainsLine(filter_help, "event or event_name"),
          "help filter should explain session behavior and examples");

    Check(!service.Execute("help missing").success &&
              !service.Execute("help show extra").success,
          "unknown help topics and extra arguments should be explicit errors");

    const auto clear = service.Execute("clear");
    Check(clear.success && clear.action == cyxwiz::RuntimeConsoleAction::Clear,
          "clear should return a UI action without mutating the store");

    const auto test = service.Execute("test");
    Check(test.success && test.lines.size() == 4 &&
              test.lines[1].level ==
                  cyxwiz::RuntimeConsoleOutputLevel::Warning &&
              test.lines[2].level ==
                  cyxwiz::RuntimeConsoleOutputLevel::Error,
          "test should preserve the existing four severity outputs");

    const auto pip = service.Execute("pip install \"some package\"");
    Check(pip.success &&
              pip.action == cyxwiz::RuntimeConsoleAction::ExecutePip &&
              pip.action_arguments.size() == 2 &&
              pip.action_arguments[0] == "install" &&
              pip.action_arguments[1] == "some package",
          "pip should return decoded arguments without a shell command");
    Check(service.Execute("pip3 list").action ==
              cyxwiz::RuntimeConsoleAction::ExecutePip,
          "pip3 compatibility should remain available");
    const auto wheel = service.Execute(
        "pip install \"C:\\models\\some wheel.whl\"");
    Check(wheel.success && wheel.action_arguments.size() == 2 &&
              wheel.action_arguments[1] == "C:\\models\\some wheel.whl",
          "quoted Windows paths should preserve literal backslashes");
    Check(!service.Execute("pipeline").success,
          "unrelated pip-prefixed commands should not execute a process");
    Check(store.GetStats().size == 6,
          "compatibility command results must not enter the runtime store");
}

void TestLogQueries(cyxwiz::RuntimeConsoleCommandService& service,
                    const cyxwiz::RuntimeLogStore& store) {
    const auto before = store.GetStats();
    const auto last = service.Execute("show logs last 2");
    Check(last.success && ContainsLine(last, "matched=6 showing=2") &&
              ContainsLine(last, "#5 ") && ContainsLine(last, "#6 ") &&
              !ContainsLine(last, "#4 "),
          "last should return the newest bounded events in sequence order");
    Check(ContainsLine(last, "2024-01-01T00:00:00.000Z") &&
              ContainsLine(last, "category=compiler") &&
              ContainsLine(last, "code=CW-C-0101") &&
              ContainsLine(last, "node=17"),
          "formatted rows should expose deterministic structured fields");

    const auto errors = service.Execute("show logs errors");
    Check(errors.success && ContainsLine(errors, "#3 ") &&
              ContainsLine(errors, "#5 ") && !ContainsLine(errors, "#2 "),
          "error view should include error and critical levels only");

    const auto alias = service.Execute("show errors last 1");
    Check(alias.success && ContainsLine(alias, "#5 ") &&
              ContainsLine(alias, "omitted 1 older"),
          "show errors should remain a bounded documented alias");

    const auto run_code = service.Execute(
        "show logs where run_id=run-1 and error_code=CW-G-0501");
    Check(run_code.success && ContainsLine(run_code, "#2 ") &&
              ContainsLine(run_code, "#3 ") &&
              !ContainsLine(run_code, "#5 "),
          "where should share run-plus-code filter semantics");

    const auto grep = service.Execute(
        "show logs grep \"native CPU fallback\"");
    Check(grep.success && ContainsLine(grep, "#3 ") &&
              !ContainsLine(grep, "#2 "),
          "grep should support quoted text without a second search engine");

    Check(!service.Execute("show logs last 0").success &&
              !service.Execute("show logs last 1001").success &&
              !service.Execute("show logs where unknown=value").success,
          "invalid limits and filters should return usage or parse errors");
    const auto after = store.GetStats();
    Check(after.size == before.size &&
              after.newest_sequence == before.newest_sequence,
          "query output must not recursively append to the event store");
}

void TestDiagnosticCommands(cyxwiz::RuntimeConsoleCommandService& service) {
    const auto exact = service.Execute("show code cw-c-0101");
    Check(exact.success && ContainsLine(exact, "Family: compiler") &&
              ContainsLine(exact, "Compiler.MissingTrainingPathNode") &&
              ContainsLine(exact, "#5 "),
          "show code should combine catalog and recent event truth");

    const auto unknown = service.Execute("show code CW-C-9999");
    Check(unknown.success && ContainsLine(unknown, "unregistered"),
          "canonical legacy codes should not invent symbolic descriptions");

    const auto family = service.Execute("show codes family cw-g-*");
    Check(family.success && ContainsLine(family, "CW-G-0501") &&
              ContainsLine(family, "Gpu.KernelExecutionFailed") &&
              !ContainsLine(family, "CW-P-0501"),
          "family lookup should remain distinct from native CPU codes");

    const auto recent = service.Execute("show codes last 2");
    Check(recent.success && ContainsLine(recent, "#5 ") &&
              ContainsLine(recent, "#6 ") && !ContainsLine(recent, "#3 "),
          "recent-code view should exclude uncoded events and retain newest");
}

void TestSessionFilters(cyxwiz::RuntimeConsoleCommandService& service) {
    const auto set = service.Execute("filter set category=training");
    Check(set.success && service.ActiveFilterExpression() ==
                             "category=training",
          "filter set should validate and activate the expression");
    const auto training = service.Execute("show logs last 10");
    Check(ContainsLine(training, "#2 ") && ContainsLine(training, "#4 ") &&
              !ContainsLine(training, "#3 "),
          "active filters should constrain subsequent log commands");

    Check(service.Execute(
              "filter save compiler_errors category=compiler and level>=error")
              .success,
          "valid named filters should be stored for the session");
    Check(service.Execute("filter use compiler_errors").success,
          "saved filters should be reusable");
    const auto compiler = service.Execute("show logs last 10");
    Check(ContainsLine(compiler, "#5 ") && !ContainsLine(compiler, "#2 "),
          "using a saved filter should replace the active filter");

    Check(!service.Execute("filter set unknown=value").success &&
              !service.Execute("filter save bad/name category=data").success &&
              !service.Execute("filter use missing").success,
          "invalid filters, names, and saved references should be explicit");
    Check(service.Execute("filter clear").success &&
              service.ActiveFilterExpression().empty(),
          "filter clear should remove only session filter state");
}

void TestBoundedHistory() {
    cyxwiz::RuntimeLogStore store(2);
    cyxwiz::RuntimeConsoleCommandService service(store);
    for (size_t index = 0;
         index < cyxwiz::RuntimeConsoleCommandService::kCommandHistoryCapacity +
                     5;
         ++index) {
        service.Execute("unknown-" + std::to_string(index));
    }
    Check(service.CommandHistorySize() ==
              cyxwiz::RuntimeConsoleCommandService::kCommandHistoryCapacity,
          "command history should remain at its fixed session capacity");
    Check(service.PreviousCommand() == "unknown-104" &&
              service.PreviousCommand() == "unknown-103" &&
              service.NextCommand() == "unknown-104" &&
              service.NextCommand() == "",
          "history navigation should move backward, forward, then clear input");

    service.Execute("help");
    service.Execute("help");
    Check(service.PreviousCommand() == "help",
          "consecutive duplicate commands should occupy one history slot");
}

void TestRuntimeTruthCommands() {
    cyxwiz::RuntimeLogStore store(8);
    FakeTruthProvider provider;
    provider.current_training.available = true;
    provider.current_training.active = true;
    provider.current_training.source = "current_training_trace";
    provider.current_training.run_id = "train-42";
    provider.current_training.status = "running";
    provider.current_training.latest_stage = "optimizer";
    provider.current_training.epoch = 3;
    provider.current_training.total_epochs = 10;
    provider.current_training.batch = 7;
    provider.current_training.total_batches = 20;
    provider.current_training.loss = 0.25f;
    provider.current_training.accuracy = 0.75f;
    provider.current_training.requested_backend = "arrayfire_cuda";
    provider.current_training.requested_device_id = 1;
    provider.current_training.effective_backend = "arrayfire_opencl";
    provider.current_training.effective_device_id = 0;
    provider.current_training.effective_device_name = "Intel UHD 630";
    provider.current_training.execution_context_id = "ctx-42";
    provider.current_training.physical_fingerprint = "uuid:opencl-42";
    provider.current_training.identity_confidence = "stable_hardware";
    provider.current_training.requested_qualification_evidence_available = true;
    provider.current_training.requested_route_qualified = false;
    provider.current_training.requested_qualification_matrix_id =
        "console-matrix";
    provider.current_training.requested_qualification_message =
        "CUDA route rejected";
    provider.current_training.effective_qualification_evidence_available = true;
    provider.current_training.effective_route_qualified = true;
    provider.current_training.effective_qualification_matrix_id =
        "console-matrix";
    provider.current_training.effective_qualification_message =
        "OpenCL route certified";
    provider.current_training.activation_succeeded = true;
    provider.current_training.execution_validated = true;
    provider.current_training.preflight_stage = "complete";
    provider.current_training.fallback_policy =
        "allow_native_cpu_fallback";
    provider.current_training.native_cpu_fallback_count = 1;
    provider.current_training.host_sync_count = 2;
    provider.current_training.host_sync_bytes = 16;
    provider.current_training.host_sync_summary = "bounded_scalar=2";
    provider.current_training.placement_fingerprint = "placement-42";
    provider.current_training.placement_entry_count = 8;
    provider.current_training.placement_summary = "8 ArrayFire stages";
    provider.current_training.residency_verdict = "mixed_with_fallback";
    provider.current_training.residency_reason = "declared_lstm_fallback";

    cyxwiz::RuntimeTrainingEventTruth fallback;
    fallback.timestamp = "2026-08-11 10:00:00";
    fallback.stage = "forward";
    fallback.status = "warning";
    fallback.native_cpu_fallback = true;
    fallback.fallback_operation = "lstm_forward";
    fallback.fallback_reason = "kernel_overflow";
    fallback.fallback_policy = "allow_native_cpu_fallback";
    provider.current_training.recent_events.push_back(fallback);

    cyxwiz::RuntimeTrainingEventTruth host_sync;
    host_sync.timestamp = "2026-08-11 10:00:01";
    host_sync.stage = "metrics";
    host_sync.host_sync_bytes = 8;
    host_sync.host_sync_category = "bounded_scalar";
    host_sync.host_sync_reason = "metric_reporting";
    provider.current_training.recent_events.push_back(host_sync);

    cyxwiz::RuntimeTrainingEventTruth materialization;
    materialization.stage = "materialization";
    materialization.task_id = 9;
    materialization.task_stage = "encode";
    materialization.task_progress = 0.5f;
    materialization.node_id = 17;
    provider.current_training.materialization_events.push_back(materialization);

    provider.last_training = provider.current_training;
    provider.last_training.active = false;
    provider.last_training.source = "last_training_trace";
    provider.last_training.status = "completed";

    provider.device.active_available = true;
    provider.device.active_source = "active_training_trace";
    provider.device.active_run_id = "train-42";
    provider.device.active_backend = "arrayfire_opencl";
    provider.device.active_device_id = 0;
    provider.device.active_device_name = "Intel UHD 630";
    provider.device.active_execution_validated = true;
    provider.device.active_preflight_stage = "complete";
    provider.device.queued_available = true;
    provider.device.queued_backend = "arrayfire_cpu";
    provider.device.queued_device_id = 0;
    provider.device.next_run_policy = "forbid_native_cpu_fallback";
    provider.device.next_run_policy_source = "queued";
    provider.device.inventory_source = "cached_discovery";
    provider.device.inventory_status = "available";
    provider.device.available_devices = {
        {"arrayfire_cpu", 0, "CPU", 0},
        {"arrayfire_opencl", 0, "Intel UHD 630", 2147483648ULL},
        {"arrayfire_oneapi", 0, "oneAPI device 0", 0},
    };
    auto& cpu_device = provider.device.available_devices[0];
    cpu_device.device_kind = "cpu";
    cpu_device.identity_confidence = "provider_reported";
    auto& oneapi_device = provider.device.available_devices.back();
    oneapi_device.backend_available = true;
    oneapi_device.device_selectable = true;
    oneapi_device.metadata_status = "unsupported";
    oneapi_device.metadata_error_code = 301;
    oneapi_device.name_is_fallback = true;
    oneapi_device.device_kind = "unknown";
    oneapi_device.identity_confidence = "backend_local";
    auto& opencl_device = provider.device.available_devices[1];
    opencl_device.device_kind = "gpu";
    opencl_device.identity_confidence = "stable_hardware";
    opencl_device.provider = "Intel";
    opencl_device.provider_known = true;
    opencl_device.driver_version = "31.0";
    opencl_device.driver_version_known = true;
    opencl_device.physical_fingerprint = "uuid:opencl-42";
    opencl_device.physical_fingerprint_known = true;

    cyxwiz::RouteQualificationSnapshot qualification;
    qualification.matrix_id = "console-matrix";
    const auto add_qualification =
        [&](cyxwiz::DeviceType type, int device_id,
            std::string fingerprint, int crash_count) {
            cyxwiz::RouteQualificationRecord record;
            record.type = type;
            record.device_id = device_id;
            record.physical_fingerprint = std::move(fingerprint);
            record.operation_count =
                cyxwiz::kRouteQualificationOperationCount;
            record.pass_count =
                cyxwiz::kRouteQualificationOperationCount - crash_count;
            record.crash_count = crash_count;
            record.certified = crash_count == 0;
            qualification.routes.push_back(std::move(record));
        };
    add_qualification(cyxwiz::DeviceType::CPU, 0, {}, 0);
    add_qualification(cyxwiz::DeviceType::OPENCL, 0,
                      "uuid:opencl-42", 0);
    add_qualification(cyxwiz::DeviceType::ONEAPI, 0, {}, 1);
    auto& oneapi_qualification = qualification.routes.back();
    oneapi_qualification.display_name = "Intel(R) Iris(R) Xe Graphics";
    oneapi_qualification.device_kind = cyxwiz::DeviceKind::GPU;
    oneapi_qualification.device_kind_known = true;
    oneapi_qualification.identity_source = "intel_ur_selector_opencl_gpu";
    cyxwiz::InstallRouteQualificationSnapshot(std::move(qualification));

    cyxwiz::RuntimeConsoleCommandService service(store, &provider);
    const auto current = service.Execute("show training current");
    Check(current.success && ContainsLine(current, "run=train-42") &&
              ContainsLine(current, "epoch=3/10 batch=7/20") &&
              ContainsLine(current, "requested=arrayfire_cuda:1") &&
              ContainsLine(current, "effective=arrayfire_opencl:0") &&
              ContainsLine(current, "preflight=complete") &&
              ContainsLine(current, "execution=validated") &&
              ContainsLine(current,
                           "Qualification: requested=failed evidence='Legacy route qualification evidence' effective=passed evidence='Legacy route qualification evidence'") &&
              ContainsLine(current,
                           "Identity: confidence=stable_hardware fingerprint=uuid:opencl-42") &&
              ContainsLine(current, "native_cpu_fallback=1"),
          "current training should preserve progress and requested/effective truth");

    Check(ContainsLine(service.Execute("show training fallback"),
                       "fallback_operation=lstm_forward") &&
              ContainsLine(service.Execute("show training host-sync"),
                           "host_sync_category=bounded_scalar") &&
              ContainsLine(service.Execute("show training placement"),
                           "fingerprint=placement-42") &&
              ContainsLine(service.Execute("show training materialization"),
                           "task=9 task_stage=encode"),
          "specialized training commands should expose retained trace evidence");

    provider.current_training = {};
    const auto no_current = service.Execute("show training current");
    const auto last = service.Execute("show training last");
    Check(no_current.success && ContainsLine(no_current, "No active training") &&
              last.success && ContainsLine(last, "source=last_training_trace") &&
              ContainsLine(last, "status=completed"),
          "current and last training sources should never be conflated");
    const auto last_materialization =
        service.Execute("show materialization last");
    Check(last_materialization.success &&
              ContainsLine(last_materialization,
                           "source=last_training_trace") &&
              ContainsLine(last_materialization, "task=9 task_stage=encode"),
          "materialization last should reuse retained training evidence");

    const auto active = service.Execute("show device active");
    Check(active.success &&
              ContainsLine(active, "source=active_training_trace") &&
              ContainsLine(active, "backend=arrayfire_opencl") &&
              ContainsLine(active, "preflight=complete") &&
              ContainsLine(active, "execution=validated") &&
              ContainsLine(active, "Active run: train-42") &&
              !provider.last_inventory_requested,
          "active device should use run-bound truth without discovery");
    const auto queued = service.Execute("show device queued");
    Check(queued.success && ContainsLine(queued, "backend=arrayfire_cpu") &&
              ContainsLine(queued, "source=queued") &&
              ContainsLine(queued, "forbid_native_cpu_fallback"),
          "queued device and next-run policy should remain separately labeled");
    const auto backends = service.Execute("show device backends");
    const auto available = service.Execute("show device available");
    const auto oneapi = service.Execute("show device oneapi");
    Check(provider.last_inventory_requested &&
              ContainsLine(backends, "source=cached_discovery") &&
              ContainsLine(backends, "arrayfire_cpu") &&
              ContainsLine(backends, "arrayfire_opencl"),
          "backend inventory should label its source and available backends");
    Check(ContainsLine(available, "provider='Intel'"),
          "available-device inventory should expose provider");
    Check(ContainsLine(available, "driver='31.0'"),
          "available-device inventory should expose driver");
    Check(ContainsLine(available, "identity=stable_hardware"),
          "available-device inventory should expose identity confidence");
    Check(ContainsLine(available, "fingerprint=uuid:opencl-42"),
          "available-device inventory should expose physical fingerprint");
    Check(ContainsLine(oneapi, "backend=arrayfire_oneapi") &&
              ContainsLine(oneapi, "name='Intel(R) Iris(R) Xe Graphics'") &&
              ContainsLine(oneapi,
                           "name_source=intel_ur_selector_opencl_gpu") &&
              ContainsLine(oneapi, "selectable=true") &&
              ContainsLine(oneapi, "kind=gpu") &&
              ContainsLine(oneapi, "identity=backend_local") &&
              ContainsLine(oneapi, "metadata=unsupported") &&
              ContainsLine(oneapi, "matrix_status=failed") &&
              ContainsLine(oneapi, "authorization=matrix_rejected") &&
              ContainsLine(oneapi, "memory=unknown") &&
              ContainsLine(oneapi, "arrayfire_error=301") &&
              !ContainsLine(oneapi, "backend=arrayfire_opencl"),
          "oneAPI inventory should preserve unknown identity and filter exactly");

    const auto qualification_result =
        service.Execute("show device qualification");
    const auto route = service.Execute("show device route opencl:0");
    const auto recommendations =
        service.Execute("show device recommendations oneapi:0");
    Check(qualification_result.success &&
              ContainsLine(qualification_result,
                           "evidence='Legacy route qualification evidence'") &&
              !ContainsLine(qualification_result, "console-matrix") &&
              ContainsLine(qualification_result,
                           "route=arrayfire_cpu:0 name='") &&
              ContainsLine(qualification_result, "matrix_status=passed") &&
              ContainsLine(qualification_result, "crash=1"),
          "qualification command should expose exact retained matrix evidence");
    Check(route.success &&
              ContainsLine(route, "Route: arrayfire_opencl:0") &&
              ContainsLine(route, "matrix_status=passed") &&
              ContainsLine(route, "authorization=ready") &&
              ContainsLine(route, "fingerprint=uuid:opencl-42"),
          "route command should combine inventory identity and qualification");
    Check(recommendations.success &&
              ContainsLine(recommendations, "failed=arrayfire_oneapi:0") &&
              ContainsLine(recommendations, "arrayfire_cpu_recovery") &&
              ContainsLine(recommendations,
                           "Stable physical identity does not prove"),
          "recommendations should use shared identity and certification policy");
    Check(!service.Execute("show device route cuda:not-a-number").success &&
              !service.Execute("show device recommendations cuda:9").success,
          "device route commands should reject malformed or absent routes");
    cyxwiz::ClearRouteQualificationSnapshot();

    provider.current_run.available = true;
    provider.current_run.source = "current_training_trace";
    provider.current_run.run_id = "train-42";
    provider.current_run.training_run_id = "train-42";
    provider.current_run.status = "running";
    provider.current_run.effective_backend = "arrayfire_opencl";
    provider.current_run.effective_device_name = "Intel UHD 630";
    const auto current_run = service.Execute("show run current");
    Check(current_run.success && ContainsLine(current_run, "run=train-42") &&
              ContainsLine(current_run, "status=running") &&
              ContainsLine(current_run, "effective=arrayfire_opencl:0"),
          "show run current should use the active run adapter snapshot");

    provider.run = provider.current_run;
    provider.run.source = "debug_run_store_training_link";
    provider.run.debug_run_id = "local-debug-42";
    provider.run.issue_count = 2;
    provider.run.native_cpu_fallback_count = 1;
    provider.run.training_evidence_available = true;
    provider.run.training = provider.last_training;
    provider.run.training.run_id = "train-42";
    provider.run.training.host_sync_count = 2;
    provider.run.training.host_sync_bytes = 16;
    provider.run.training.recent_events = {fallback, host_sync};
    provider.run.issues = {
        {"debug_run_issue", "error", "CW-T-0501", 17, "Dense",
         "Training failed"},
        {"debug_run_issue", "warning", "CW-G-0501", 18, "LSTM",
         "Kernel fallback"},
    };
    provider.run.events = {
        {"2026-08-11 10:00:02", "studio_event", "Debugger.Run", "ok",
         17, "Completed debugger capture"},
    };

    auto live_run_event = MakeEvent(
        "training", cyxwiz::RuntimeLogLevel::Error, "live training error");
    live_run_event.run_id = "train-42";
    live_run_event.primary_error_code = "CW-T-0501";
    Check(store.Append(std::move(live_run_event)),
          "run command fixture should append");

    Check(ContainsLine(service.Execute("show run train-42 summary"),
                       "debug_run=local-debug-42") &&
              ContainsLine(service.Execute("show run train-42 events"),
                           "source=studio_event") &&
              ContainsLine(service.Execute("show run train-42 fallback"),
                           "fallback_operation=lstm_forward") &&
              ContainsLine(service.Execute("show run train-42 host-sync"),
                           "host_sync_category=bounded_scalar"),
          "run summary and evidence commands should use adapter truth");

    const auto exact_run_code =
        service.Execute("show run train-42 code cw-t-0501");
    const auto family_run_code =
        service.Execute("show run train-42 codes cw-g-*");
    Check(exact_run_code.success &&
              ContainsLine(exact_run_code, "live training error") &&
              ContainsLine(exact_run_code, "Training failed") &&
              !ContainsLine(exact_run_code, "Kernel fallback") &&
              family_run_code.success &&
              ContainsLine(family_run_code, "Kernel fallback") &&
              !ContainsLine(family_run_code, "Training failed"),
          "run code queries should combine live and persisted evidence with exact filtering");
}

void TestBackendSupportCommands() {
    cyxwiz::RuntimeLogStore store(4);
    FakeTruthProvider provider;
    provider.backend_packs.packaged_runtime = true;
    provider.backend_packs.runtime_status = "ready";
    provider.backend_packs.runtime_set_id = "runtime-v1";
    provider.backend_packs.runtime_generation = 4;
    provider.backend_packs.base_pack_id = "base-v1";
    provider.backend_packs.catalog_status = "verified";
    provider.backend_packs.catalog_id = "catalog-v1";
    provider.backend_packs.catalog_source = "local_signed_cache";
    provider.backend_packs.network_policy = "read_only_no_network";
    provider.backend_packs.proxy_policy = "direct_https_or_explicit_offline";

    const auto add_pack = [&](std::string backend, std::string pack_id,
                              bool installed, bool active,
                              std::string layout, std::string support,
                              bool evidence, bool authorized) {
        cyxwiz::RuntimeBackendPackEntryTruth pack;
        pack.backend = std::move(backend);
        pack.pack_id = pack_id;
        pack.installed_pack_id = installed ? pack_id : std::string{};
        pack.package_version = "1.0";
        pack.state = !installed
            ? "not_installed"
            : (active ? "installed_active" : "installed_inactive");
        pack.layout_status = std::move(layout);
        pack.catalog_support = std::move(support);
        pack.download_size_bytes = 1024;
        pack.provider_requirements = {"vendor_runtime"};
        pack.installed = installed;
        pack.active = active;
        pack.delivery_metadata_available = true;
        pack.qualification_evidence_available = evidence;
        pack.training_authorized = authorized;
        provider.backend_packs.packs.push_back(std::move(pack));
    };
    add_pack("cpu", "base-v1", true, true, "valid", "supported",
             true, true);
    add_pack("cuda", "cuda-v1", true, true, "valid", "supported",
             true, false);
    add_pack("opencl", "opencl-v1", true, true, "valid", "supported",
             true, false);
    add_pack("oneapi", "oneapi-v1", true, true, "valid", "supported",
             true, false);
    add_pack("cuda", "cuda-next", false, false, "not_installed",
             "supported", false, false);
    add_pack("opencl", "opencl-corrupt", true, false, "invalid",
             "supported", false, false);
    add_pack("oneapi", "oneapi-diagnostic", true, false, "not_checked",
             "blocked", false, false);

    const auto add_device = [&](std::string backend, int id,
                                std::string name) {
        cyxwiz::RuntimeDeviceEntryTruth device;
        device.backend = std::move(backend);
        device.device_id = id;
        device.name = std::move(name);
        device.backend_available = true;
        device.device_selectable = true;
        device.device_kind = id == 0 ? "gpu" : "accelerator";
        device.identity_confidence = "provider_reported";
        device.provider = "Fixture Provider";
        device.driver_version = "1.2.3";
        device.provider_known = true;
        device.driver_version_known = true;
        provider.device.available_devices.push_back(std::move(device));
    };
    add_device("arrayfire_cpu", 0, "CPU");
    add_device("arrayfire_cuda", 0, "CUDA missing provider");
    add_device("arrayfire_cuda", 1, "CUDA incompatible device");
    add_device("arrayfire_opencl", 0, "OpenCL failed matrix");
    add_device("arrayfire_oneapi", 0, "oneAPI failed training probe");
    provider.device.inventory_source = "cached_discovery";
    provider.device.inventory_status = "available";

    cyxwiz::RouteQualificationSnapshot qualification;
    qualification.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    qualification.runtime_set_id = "runtime-v1";
    qualification.runtime_generation = 4;
    qualification.base_pack_id = "base-v1";
    qualification.compute_contract_id = cyxwiz::kCyxWizComputeContractId;
    qualification.operation_manifest_id =
        cyxwiz::kRouteQualificationOperationManifestId;
    const auto add_route = [&](cyxwiz::DeviceType type, int id,
                               std::string pack_id,
                               cyxwiz::RouteFailureStage stage,
                               cyxwiz::RouteFailureCategory category) {
        cyxwiz::RouteQualificationRecord route;
        route.type = type;
        route.device_id = id;
        route.pack_id = std::move(pack_id);
        route.operation_count = cyxwiz::kRouteQualificationOperationCount;
        if (category == cyxwiz::RouteFailureCategory::None) {
            route.pass_count = route.operation_count;
            route.certified = true;
        } else {
            route.pass_count = route.operation_count - 1;
            route.failure_count = 1;
            route.failure.stage = stage;
            route.failure.category = category;
            route.failure.operation = "sum";
            route.failure.observed_fact = "bounded fixture failure";
            route.failure.bounded_interpretation = "fixture interpretation";
            route.failure.recommended_action = "fixture action";
        }
        qualification.routes.push_back(std::move(route));
    };
    add_route(cyxwiz::DeviceType::CPU, 0, "base-v1",
              cyxwiz::RouteFailureStage::None,
              cyxwiz::RouteFailureCategory::None);
    add_route(cyxwiz::DeviceType::CUDA, 0, "cuda-v1",
              cyxwiz::RouteFailureStage::BackendLoad,
              cyxwiz::RouteFailureCategory::ProviderMissing);
    add_route(cyxwiz::DeviceType::CUDA, 1, "cuda-v1",
              cyxwiz::RouteFailureStage::Identity,
              cyxwiz::RouteFailureCategory::IdentityMismatch);
    add_route(cyxwiz::DeviceType::OPENCL, 0, "opencl-v1",
              cyxwiz::RouteFailureStage::Operation,
              cyxwiz::RouteFailureCategory::OperationFailed);
    add_route(cyxwiz::DeviceType::ONEAPI, 0, "oneapi-v1",
              cyxwiz::RouteFailureStage::StrictTraining,
              cyxwiz::RouteFailureCategory::NativeFallback);
    cyxwiz::InstallRouteQualificationSnapshot(std::move(qualification));

    cyxwiz::RuntimeConsoleCommandService service(store, &provider);
    const auto packs = service.Execute("show backend packs");
    Check(packs.success &&
              ContainsLine(packs, "network=read_only_no_network") &&
              ContainsLine(packs, "compatibility=missing_pack") &&
              ContainsLine(packs, "compatibility=corrupt_pack") &&
              ContainsLine(packs, "compatibility=policy_block"),
          "pack command should distinguish package and catalog policy states without network access");

    const auto compatibility =
        service.Execute("show backend compatibility");
    Check(compatibility.success &&
              ContainsLine(compatibility, "compatibility=ready") &&
              ContainsLine(compatibility, "compatibility=missing_provider") &&
              ContainsLine(compatibility, "compatibility=incompatible_device") &&
              ContainsLine(compatibility, "compatibility=failed_matrix") &&
              ContainsLine(compatibility, "compatibility=failed_training_probe"),
          "compatibility command should preserve distinct package, provider, device, matrix, and training failures");

    const auto support =
        service.Execute("show backend support-bundle 2");
    Check(support.success &&
              ContainsLine(support, "schema=cyxwiz.backend.support.v1") &&
              ContainsLine(support, "upload=not_requested") &&
              ContainsLine(support, "shown=2") &&
              ContainsLine(support, "limit=2") &&
              ContainsLine(support, cyxwiz::kRouteQualificationMatrixId) &&
              !ContainsLine(support, "ticket89"),
          "support output should be bounded, shareable, local-only, and use production contract labels");
    const auto full_support =
        service.Execute("show backend support-bundle 100");
    Check(full_support.success &&
              ContainsLine(full_support, "route=arrayfire_cuda:0") &&
              ContainsLine(full_support, "failure_category=provider_missing") &&
              ContainsLine(full_support, "passed=") &&
              ContainsLine(full_support, "timeout=0 crash=0") &&
              ContainsLine(full_support, "benchmark_median_ms="),
          "support output should retain bounded install, probe, failure, and benchmark facts");
    Check(!service.Execute("show backend support-bundle 0").success &&
              !service.Execute("show backend support-bundle 101").success &&
              !service.Execute("show backend packs extra").success,
          "backend command limits and forms should fail closed");
    cyxwiz::ClearRouteQualificationSnapshot();
}

} // namespace

int main() {
    cyxwiz::RuntimeLogStore store(16);
    Populate(store);
    cyxwiz::RuntimeConsoleCommandService service(store);
    TestRegistryAndCompatibility(service, store);
    TestLogQueries(service, store);
    TestDiagnosticCommands(service);
    TestSessionFilters(service);
    TestBoundedHistory();
    TestRuntimeTruthCommands();
    TestBackendSupportCommands();
    std::cout << "Runtime Console command contracts passed\n";
    return 0;
}
