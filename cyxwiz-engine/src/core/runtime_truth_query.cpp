#include "runtime_truth_query.h"

#include "backend_pack_catalog_adapter.h"
#include "debug_run_store.h"
#include "execution_device_context.h"
#include "execution_device_preferences.h"
#include "route_qualification_snapshot.h"
#include "training_manager.h"
#include "training_trace_collector.h"

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_platform.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/device.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <utility>

namespace cyxwiz {
namespace {

RuntimeTrainingEventTruth MakeEventTruth(const TrainingTraceEvent& event) {
    RuntimeTrainingEventTruth truth;
    truth.timestamp = event.timestamp;
    truth.stage = event.stage;
    truth.status = event.status;
    truth.message = event.message;
    truth.epoch = event.epoch;
    truth.batch = event.batch;
    truth.total_batches = event.total_batches;
    truth.native_cpu_fallback = event.native_cpu_fallback;
    truth.fallback_operation = event.fallback_operation;
    truth.fallback_reason = event.fallback_reason;
    truth.fallback_policy = event.fallback_policy;
    truth.host_sync_bytes = event.arrayfire_host_sync_bytes;
    truth.host_sync_reason = event.arrayfire_host_sync_reason;
    truth.host_sync_category = event.arrayfire_host_sync_category;
    truth.host_sync_operation = event.arrayfire_host_sync_operation;
    truth.task_id = event.task_id;
    truth.task_name = event.task_name;
    truth.task_stage = event.task_stage;
    truth.task_progress = event.task_progress;
    truth.node_id = event.node_id;
    truth.node_name = event.node_name;
    return truth;
}

RuntimeTrainingTruth MakeTrainingTruth(
    const TrainingTraceSummary& trace, bool active,
    const TrainingMetrics* metrics) {
    RuntimeTrainingTruth truth;
    truth.available = trace.available || active;
    truth.active = active;
    truth.source = active ? "current_training_trace" : "last_training_trace";
    truth.run_id = trace.run_id;
    truth.status = trace.status;
    truth.latest_stage = trace.latest_stage;
    truth.latest_timestamp = trace.latest_timestamp;
    truth.epoch = trace.latest_epoch;
    truth.total_epochs = 0;
    truth.batch = trace.latest_batch;
    truth.total_batches = trace.latest_total_batches;
    truth.loss = trace.latest_loss;
    truth.accuracy = trace.latest_accuracy;
    if (metrics) {
        truth.epoch = metrics->current_epoch;
        truth.total_epochs = metrics->total_epochs;
        truth.batch = metrics->current_batch;
        truth.total_batches = metrics->total_batches;
        truth.loss = metrics->train_loss;
        truth.accuracy = metrics->train_accuracy;
        if (truth.status.empty() || truth.status == "idle") {
            truth.status = metrics->status_message;
        }
    }
    truth.requested_backend = trace.requested_backend;
    truth.requested_device_id = trace.requested_device_id;
    truth.effective_backend = trace.effective_backend;
    truth.effective_device_id = trace.effective_device_id;
    truth.effective_device_name = trace.effective_device_name;
    truth.execution_context_id = trace.execution_context_id;
    truth.physical_fingerprint = trace.physical_fingerprint;
    truth.identity_confidence = trace.identity_confidence;
    truth.requested_qualification_evidence_available =
        trace.requested_qualification_evidence_available;
    truth.requested_route_qualified = trace.requested_route_qualified;
    truth.requested_qualification_matrix_id =
        trace.requested_qualification_matrix_id;
    truth.requested_qualification_message =
        trace.requested_qualification_message;
    truth.effective_qualification_evidence_available =
        trace.effective_qualification_evidence_available;
    truth.effective_route_qualified = trace.effective_route_qualified;
    truth.effective_qualification_matrix_id =
        trace.effective_qualification_matrix_id;
    truth.effective_qualification_message =
        trace.effective_qualification_message;
    truth.activation_succeeded = trace.activation_succeeded;
    truth.execution_validated = trace.execution_validated;
    truth.selection_fallback_applied =
        trace.selection_fallback_applied;
    truth.preflight_stage = trace.preflight_stage;
    truth.preflight_error_code = trace.preflight_error_code;
    truth.fallback_policy = trace.fallback_policy;
    truth.native_cpu_fallback_count = trace.native_cpu_fallback_count;
    truth.transfer_event_count = trace.transfer_event_count;
    truth.transfer_known_bytes = trace.transfer_known_bytes;
    truth.synchronization_event_count = trace.synchronization_event_count;
    truth.synchronization_known_bytes = trace.synchronization_known_bytes;
    truth.host_sync_count = trace.arrayfire_host_sync_count;
    truth.host_sync_bytes = trace.arrayfire_host_sync_bytes;
    truth.host_sync_summary = trace.arrayfire_host_sync_summary;
    truth.placement_fingerprint = trace.placement_fingerprint;
    truth.placement_entry_count = trace.placement_entry_count;
    truth.placement_summary = trace.placement_summary;
    truth.residency_verdict = trace.residency_verdict;
    truth.residency_reason = trace.residency_reason;
    truth.recent_events.reserve(trace.recent_events.size());
    for (const auto& event : trace.recent_events) {
        truth.recent_events.push_back(MakeEventTruth(event));
    }
    truth.materialization_events.reserve(trace.materialization_events.size());
    for (const auto& event : trace.materialization_events) {
        truth.materialization_events.push_back(MakeEventTruth(event));
    }
    return truth;
}

void ApplyTrainingTruth(RuntimeRunTruth& run,
                        const RuntimeTrainingTruth& training) {
    run.training_evidence_available = training.available;
    run.training = training;
    if (!training.available) return;
    run.training_run_id = training.run_id;
    run.status = training.status;
    run.requested_backend = training.requested_backend;
    run.requested_device_id = training.requested_device_id;
    run.effective_backend = training.effective_backend;
    run.effective_device_id = training.effective_device_id;
    run.effective_device_name = training.effective_device_name;
    run.execution_context_id = training.execution_context_id;
    run.execution_validated = training.execution_validated;
    run.preflight_stage = training.preflight_stage;
    run.placement_fingerprint = training.placement_fingerprint;
    run.residency_verdict = training.residency_verdict;
    run.native_cpu_fallback_count = training.native_cpu_fallback_count;
    run.transfer_event_count = training.transfer_event_count;
    run.transfer_known_bytes = training.transfer_known_bytes;
    run.synchronization_event_count = training.synchronization_event_count;
    run.synchronization_known_bytes =
        training.synchronization_known_bytes;
    run.events.reserve(run.events.size() + training.recent_events.size());
    for (const auto& event : training.recent_events) {
        RuntimeRunEventTruth converted;
        converted.timestamp = event.timestamp;
        converted.source = "training_trace";
        converted.stage = event.stage;
        converted.status = event.status;
        converted.node_id = event.node_id;
        converted.message = event.message;
        run.events.push_back(std::move(converted));
    }
}

RuntimeRunTruth MakeTrainingRunTruth(const RuntimeTrainingTruth& training) {
    RuntimeRunTruth run;
    if (!training.available) return run;
    run.available = true;
    run.source = training.source;
    run.run_id = training.run_id;
    run.summary = training.active ? "Active training run"
                                  : "Latest retained training run";
    ApplyTrainingTruth(run, training);
    return run;
}

const char* RunIssueLevelName(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error: return "error";
        case IssueLevel::Warning: return "warning";
        case IssueLevel::Info: return "info";
    }
    return "unknown";
}

void AddRunIssue(RuntimeRunTruth& run, const ValidationIssue& issue,
                 std::string source) {
    RuntimeRunIssueTruth converted;
    converted.source = std::move(source);
    converted.level = RunIssueLevelName(issue.level);
    converted.code = issue.error_code;
    converted.node_id = issue.node_id;
    converted.node_name = issue.node_name;
    converted.message = issue.message;
    run.issues.push_back(std::move(converted));
}

RuntimeRunTruth MakeDebugRunTruth(const DebugRunStoreRecord& record,
                                  std::string_view requested_run_id) {
    RuntimeRunTruth run;
    run.available = true;
    run.source = requested_run_id == record.summary.run_id
        ? "debug_run_store"
        : "debug_run_store_training_link";
    run.run_id = std::string(requested_run_id);
    run.debug_run_id = record.summary.run_id;
    run.training_run_id = record.summary.execution.training_run_id;
    run.timestamp = record.summary.timestamp;
    run.status = record.summary.execution.status.empty()
        ? (record.summary.success ? "completed" : "failed")
        : record.summary.execution.status;
    run.summary = record.summary.summary;
    run.success = record.summary.success;
    run.graph_hash = record.summary.graph_hash;
    run.issue_count = record.summary.issue_count;
    run.trace_count = record.summary.trace_count;
    run.event_count = record.summary.event_count;
    run.recommendation_count = record.summary.recommendation_count;
    const auto& execution = record.summary.execution;
    run.requested_backend = execution.requested_backend;
    run.requested_device_id = execution.requested_device_id;
    run.effective_backend = execution.effective_backend;
    run.effective_device_id = execution.effective_device_id;
    run.effective_device_name = execution.effective_device_name;
    run.execution_context_id = execution.execution_context_id;
    run.placement_fingerprint = execution.placement_fingerprint;
    run.residency_verdict = execution.residency_verdict;
    run.native_cpu_fallback_count = execution.native_cpu_fallback_count;
    run.transfer_event_count = execution.transfer_event_count;
    run.transfer_known_bytes = execution.transfer_known_bytes;
    run.synchronization_event_count = execution.synchronization_event_count;
    run.synchronization_known_bytes =
        execution.synchronization_known_bytes;

    for (const auto& issue : record.issues) {
        AddRunIssue(run, issue, "debug_run_issue");
    }
    for (const auto& trace : record.traces) {
        RuntimeRunEventTruth event;
        event.source = "debug_trace";
        event.stage = trace.phase;
        event.status = trace.status;
        event.node_id = trace.node_id;
        event.message = std::string(DebugTraceRoleName(trace.role));
        run.events.push_back(std::move(event));
        for (const auto& issue : trace.issues) {
            AddRunIssue(run, issue, "debug_trace_issue");
        }
    }
    for (const auto& stored : record.studio_events) {
        RuntimeRunEventTruth event;
        event.timestamp = stored.timestamp;
        event.source = "studio_event";
        event.stage = stored.action;
        event.status = stored.status;
        event.node_id = stored.selected_node_id;
        event.message = stored.message;
        run.events.push_back(std::move(event));
    }
    return run;
}

bool IsSafeRunId(std::string_view run_id) {
    return !run_id.empty() && run_id.size() <= 160 &&
           std::all_of(run_id.begin(), run_id.end(), [](unsigned char value) {
               return std::isalnum(value) != 0 || value == '-' || value == '_';
           });
}

std::filesystem::path ActiveRuntimeRoot() {
    const char* value = std::getenv("CYXWIZ_ACTIVE_RUNTIME_ROOT");
    return value && *value ? std::filesystem::path(value)
                           : std::filesystem::path{};
}

std::string CurrentUtc() {
    const auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char buffer[32]{};
    if (std::strftime(
            buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) {
        return {};
    }
    return buffer;
}

const char* CatalogSupportId(BackendPackCatalogSupport support) {
    switch (support) {
        case BackendPackCatalogSupport::Supported: return "supported";
        case BackendPackCatalogSupport::Diagnostic: return "diagnostic";
        case BackendPackCatalogSupport::Blocked: return "blocked";
        case BackendPackCatalogSupport::Revoked: return "revoked";
        case BackendPackCatalogSupport::Unavailable: return "unavailable";
    }
    return "unavailable";
}

const char* PackBackendId(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::ONEAPI: return "oneapi";
        default: return "unknown";
    }
}

bool QualificationMatchesRuntime(
    const RouteQualificationSnapshot& snapshot,
    const runtime::ActiveRuntimeState& state) {
    if (snapshot.matrix_id != kRouteQualificationMatrixId ||
        snapshot.compute_contract_id != kCyxWizComputeContractId ||
        snapshot.operation_manifest_id !=
            kRouteQualificationOperationManifestId ||
        snapshot.runtime_set_id != state.runtime_set_id ||
        snapshot.runtime_generation != state.generation ||
        snapshot.base_pack_id != state.base_pack_id) {
        return false;
    }
    for (const auto& route : snapshot.routes) {
        if (route.type == DeviceType::CPU) {
            if (route.pack_id != state.base_pack_id) return false;
            continue;
        }
        const auto active_pack = std::find_if(
            state.packs.begin(), state.packs.end(), [&](const auto& pack) {
                return pack.backend == PackBackendId(route.type);
            });
        if (active_pack == state.packs.end() ||
            route.pack_id != active_pack->pack_id) {
            return false;
        }
    }
    return true;
}

RuntimeBackendPackEntryTruth MakePackTruth(
    const BackendPackManagerRecord& record,
    bool runtime_layout_valid,
    const std::optional<RouteQualificationSnapshot>& qualification,
    bool qualification_matches_runtime) {
    RuntimeBackendPackEntryTruth truth;
    truth.backend = record.backend;
    truth.pack_id = record.pack_id;
    truth.installed_pack_id = record.installed_pack_id;
    truth.package_version = record.package_version;
    truth.catalog_support = CatalogSupportId(record.catalog_support);
    truth.download_size_bytes = record.download_size_bytes;
    truth.provider_requirements = record.provider_requirements;
    truth.installed = record.installed;
    truth.active = record.active;
    truth.delivery_metadata_available = record.delivery_metadata_available;
    truth.state = !record.installed
        ? "not_installed"
        : (record.active ? "installed_active" : "installed_inactive");
    truth.layout_status = !record.installed
        ? "not_installed"
        : (record.active
               ? (runtime_layout_valid ? "valid" : "invalid")
               : "not_checked");

    const std::string installed_id = record.installed_pack_id.empty()
        ? record.pack_id
        : record.installed_pack_id;
    if (!record.installed || !qualification ||
        !qualification_matches_runtime ||
        installed_id.empty()) {
        return truth;
    }
    for (const auto& route : qualification->routes) {
        if (record.backend != PackBackendId(route.type) ||
            route.pack_id != installed_id) {
            continue;
        }
        truth.qualification_evidence_available = true;
        truth.training_authorized =
            truth.training_authorized || route.certified;
    }
    return truth;
}

RuntimeBackendPackTruth ReadBackendPackTruth() {
    RuntimeBackendPackTruth truth;
    truth.catalog_source = "local_signed_cache";
    truth.network_policy = "read_only_no_network";
#ifdef _WIN32
    truth.proxy_policy = "winhttp_system_automatic";
#else
    truth.proxy_policy = "direct_https_or_explicit_offline";
#endif

    const auto runtime_root = ActiveRuntimeRoot();
    if (runtime_root.empty()) {
        truth.runtime_status = "development_runtime";
        truth.catalog_status = "not_packaged";
        return truth;
    }
    truth.packaged_runtime = true;
    if (!runtime_root.is_absolute()) {
        truth.runtime_status = "invalid_root";
        truth.catalog_status = "local_cache_unavailable";
        return truth;
    }

    runtime::ActiveRuntimeState active;
    std::string state_error;
    if (!runtime::LoadActiveRuntimeState(
            runtime_root / "active-runtime.json", active, state_error)) {
        truth.runtime_status = "invalid_state";
        truth.catalog_status = "local_cache_unavailable";
        return truth;
    }
    truth.runtime_set_id = active.runtime_set_id;
    truth.runtime_generation = active.generation;
    truth.base_pack_id = active.base_pack_id;

    runtime::ActiveRuntime resolved;
    std::string layout_error;
    const bool runtime_layout_valid = runtime::ResolveRuntimeState(
        runtime_root, active, resolved, layout_error);
    truth.runtime_status = runtime_layout_valid ? "ready" : "invalid_layout";

    runtime::VerifiedBackendPackCatalogSnapshot catalog;
    std::string catalog_error;
    auto trust = runtime::BackendPackTrustStore::Load(
        runtime_root / "trust" / "trusted-keys.json", catalog_error);
    bool catalog_available = false;
    if (trust) {
        runtime::BackendPackLifecycleService lifecycle(
            runtime_root,
            runtime::BackendPackMetadataVerifier(
                std::move(*trust), GetVersionString(),
                std::string(runtime::CurrentBackendPackPlatformId()),
                std::string(runtime::CurrentBackendPackArchitectureId())));
        catalog_available = lifecycle.ReadCatalogSnapshot(
            CurrentUtc(), catalog, catalog_error);
    }
    truth.catalog_status = catalog_available ? "verified"
                                             : "local_cache_unavailable";
    if (catalog_available) truth.catalog_id = catalog.catalog.catalog_id;

    const auto qualification = GetRouteQualificationSnapshot();
    const bool qualification_matches_runtime =
        qualification && QualificationMatchesRuntime(*qualification, active);

    BackendPackManagerRecord base;
    base.backend = "cpu";
    base.pack_id = active.base_pack_id;
    base.installed_pack_id = active.base_pack_id;
    base.catalog_support = BackendPackCatalogSupport::Supported;
    base.installed = true;
    base.active = true;
    base.delivery_metadata_available = true;
    truth.packs.push_back(MakePackTruth(
        base, runtime_layout_valid, qualification,
        qualification_matches_runtime));

    std::vector<BackendPackManagerRecord> records;
    if (catalog_available) {
        records = BuildBackendPackCatalogRecords(catalog, active);
        for (auto& record : records) {
            if (record.installed && !record.active &&
                record.pack_id != record.installed_pack_id) {
                record.installed = false;
                record.installed_pack_id.clear();
            }
        }
        for (const auto& installed : active.packs) {
            const bool exact_record = std::any_of(
                records.begin(), records.end(), [&](const auto& record) {
                    return record.pack_id == installed.pack_id &&
                           record.backend == installed.backend &&
                           record.installed && record.active;
                });
            if (exact_record) continue;
            BackendPackManagerRecord record;
            record.backend = installed.backend;
            record.pack_id = installed.pack_id;
            record.installed_pack_id = installed.pack_id;
            record.installed = true;
            record.active = true;
            records.push_back(std::move(record));
        }
    } else {
        records.reserve(active.packs.size());
        for (const auto& installed : active.packs) {
            BackendPackManagerRecord record;
            record.backend = installed.backend;
            record.pack_id = installed.pack_id;
            record.installed_pack_id = installed.pack_id;
            record.installed = true;
            record.active = true;
            records.push_back(std::move(record));
        }
    }
    truth.packs.reserve(truth.packs.size() + records.size());
    for (const auto& record : records) {
        truth.packs.push_back(MakePackTruth(
            record, runtime_layout_valid, qualification,
            qualification_matches_runtime));
    }
    return truth;
}

class EngineRuntimeTruthProvider final : public RuntimeTruthQueryProvider {
public:
    RuntimeTrainingTruth GetCurrentTraining() const override {
        auto& manager = TrainingManager::Instance();
        const bool active = manager.IsTrainingActive();
        if (!active) {
            RuntimeTrainingTruth truth;
            truth.source = "training_manager";
            return truth;
        }
        const auto metrics = manager.GetCurrentMetrics();
        return MakeTrainingTruth(
            TrainingTraceCollector::Instance().Snapshot(), true, &metrics);
    }

    RuntimeTrainingTruth GetLastTraining() const override {
        auto& manager = TrainingManager::Instance();
        const bool active = manager.IsTrainingActive();
        if (active) {
            const auto metrics = manager.GetCurrentMetrics();
            return MakeTrainingTruth(
                TrainingTraceCollector::Instance().Snapshot(), true, &metrics);
        }
        return MakeTrainingTruth(
            TrainingTraceCollector::LatestTrace(), false, nullptr);
    }

    RuntimeDeviceTruth GetDeviceTruth(bool include_inventory) override {
        RuntimeDeviceTruth truth;
        const auto training = GetCurrentTraining();
        if (training.active && !training.effective_backend.empty()) {
            truth.active_available = true;
            truth.active_source = "active_training_trace";
            truth.active_run_id = training.run_id;
            truth.active_backend = training.effective_backend;
            truth.active_device_id = training.effective_device_id;
            truth.active_device_name = training.effective_device_name;
            truth.active_execution_validated =
                training.execution_validated;
            truth.active_preflight_stage = training.preflight_stage;
        } else {
            try {
                if (auto* process = Device::GetCurrentDevice()) {
                    truth.active_available = true;
                    truth.active_source = "process_runtime";
                    truth.active_backend =
                        ExecutionDeviceSelectionBackendName(process->GetType());
                    truth.active_device_id = process->GetDeviceId();
                    truth.active_device_name = process->GetInfo().name;
                }
            } catch (const std::exception&) {
                truth.active_available = false;
                truth.active_source = "process_runtime_query_failed";
            }
        }

        if (const auto pending = GetPendingExecutionDeviceSelection()) {
            truth.queued_available = true;
            truth.queued_backend =
                ExecutionDeviceSelectionBackendName(pending->type);
            truth.queued_device_id = pending->device_id;
        }
        const auto policy = GetNextRunExecutionPolicy();
        truth.next_run_policy = policy.has_value()
            ? (*policy == ArrayFireFallbackPolicy::ForbidNativeCpuFallback
                   ? "forbid_native_cpu_fallback"
                   : "allow_native_cpu_fallback")
            : "allow_native_cpu_fallback";
        truth.next_run_policy_source = policy.has_value() ? "queued"
                                                          : "default";

        if (!include_inventory) return truth;

        std::lock_guard<std::mutex> lock(inventory_mutex_);
        if (!inventory_initialized_) {
            if (HasActiveExecutionDeviceContext() || training.active) {
                truth.inventory_source = "not_queried";
                truth.inventory_status =
                    "deferred_while_execution_context_is_active";
                return truth;
            }
            try {
                for (const auto& device : Device::GetAvailableDevices()) {
                    RuntimeDeviceEntryTruth entry;
                    entry.backend =
                        ExecutionDeviceSelectionBackendName(device.type);
                    entry.device_id = device.device_id;
                    entry.name = device.name;
                    entry.memory_total =
                        static_cast<uint64_t>(device.memory_total);
                    entry.backend_available = device.backend_available;
                    entry.device_selectable = device.device_selectable;
                    entry.execution_validated = device.execution_validated;
                    entry.device_kind = DeviceKindName(device.kind);
                    entry.identity_confidence =
                        DeviceIdentityConfidenceName(
                            device.identity_confidence);
                    entry.provider = device.provider;
                    entry.driver_version = device.driver_version;
                    entry.hardware_vendor_id = device.hardware_vendor_id;
                    entry.hardware_device_id = device.hardware_device_id;
                    entry.pci_domain = device.pci_domain;
                    entry.pci_bus = device.pci_bus;
                    entry.pci_device = device.pci_device;
                    entry.pci_function = device.pci_function;
                    entry.hardware_uuid = device.hardware_uuid;
                    entry.hardware_luid = device.hardware_luid;
                    entry.physical_fingerprint =
                        device.physical_fingerprint;
                    entry.provider_known = device.provider_known;
                    entry.driver_version_known =
                        device.driver_version_known;
                    entry.hardware_vendor_id_known =
                        device.hardware_vendor_id_known;
                    entry.hardware_device_id_known =
                        device.hardware_device_id_known;
                    entry.pci_location_known = device.pci_location_known;
                    entry.hardware_uuid_known = device.hardware_uuid_known;
                    entry.hardware_luid_known = device.hardware_luid_known;
                    entry.physical_fingerprint_known =
                        device.physical_fingerprint_known;
                    entry.metadata_status =
                        DeviceMetadataStatusName(device.metadata_status);
                    entry.metadata_error_code = device.metadata_error_code;
                    entry.metadata_message = device.metadata_message;
                    entry.name_known = device.name_known;
                    entry.name_is_fallback = device.name_is_fallback;
                    entry.memory_total_known = device.memory_total_known;
                    inventory_.push_back(std::move(entry));
                }
                inventory_initialized_ = true;
                truth.inventory_source = "fresh_discovery";
            } catch (const std::exception& error) {
                truth.inventory_source = "discovery_failed";
                truth.inventory_status = error.what();
                return truth;
            }
        } else {
            truth.inventory_source = "cached_discovery";
        }
        truth.available_devices = inventory_;
        truth.inventory_status = "available";
        return truth;
    }

    RuntimeBackendPackTruth GetBackendPackTruth() const override {
        return ReadBackendPackTruth();
    }

    RuntimeRunTruth GetCurrentRun() const override {
        return MakeTrainingRunTruth(GetCurrentTraining());
    }

    RuntimeRunTruth GetRun(std::string_view run_id) const override {
        const auto current = GetCurrentTraining();
        if (current.available && current.run_id == run_id) {
            return MakeTrainingRunTruth(current);
        }
        const auto latest = GetLastTraining();
        if (latest.available && latest.run_id == run_id) {
            return MakeTrainingRunTruth(latest);
        }
        if (!IsSafeRunId(run_id)) return {};

        if (const auto direct = DebugRunStore::Load(std::string(run_id))) {
            auto truth = MakeDebugRunTruth(*direct, run_id);
            if (latest.available &&
                latest.run_id == truth.training_run_id) {
                ApplyTrainingTruth(truth, latest);
            }
            return truth;
        }
        for (const auto& summary : DebugRunStore::ListRecent(1000)) {
            if (summary.execution.training_run_id != run_id) continue;
            if (const auto linked = DebugRunStore::Load(summary.run_id)) {
                auto truth = MakeDebugRunTruth(*linked, run_id);
                if (latest.available && latest.run_id == run_id) {
                    ApplyTrainingTruth(truth, latest);
                }
                return truth;
            }
        }
        return {};
    }

private:
    std::mutex inventory_mutex_;
    bool inventory_initialized_ = false;
    std::vector<RuntimeDeviceEntryTruth> inventory_;
};

} // namespace

std::unique_ptr<RuntimeTruthQueryProvider> CreateEngineRuntimeTruthProvider() {
    return std::make_unique<EngineRuntimeTruthProvider>();
}

} // namespace cyxwiz
