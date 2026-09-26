#include "node_doctor.h"

#include "node_data_dir.h"

#include <grpcpp/grpcpp.h>

#include "core/compute_runtime_config.h"
#include "core/compute_runtime_paths.h"
#include "core/execution_device_preferences.h"

#include <cyxwiz/cyxwiz.h>
#include <spdlog/fmt/fmt.h>

#include <chrono>
#include <fstream>

namespace cyxwiz::servernode {
namespace {

std::string RouteName(const RouteQualificationRecord& route) {
    return ExecutionDeviceSelectionBackendName(route.type) + ":" + std::to_string(route.device_id);
}

std::string Gigabytes(std::uint64_t bytes) {
    return fmt::format("{:.1f} GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

}  // namespace

std::vector<DoctorCheck> EvaluateNodeReadiness(const DoctorFacts& facts) {
    std::vector<DoctorCheck> checks;
    const auto& routes = facts.capability.routes;
    size_t certified = 0;
    for (const auto& entry : routes) certified += entry.route.certified ? 1 : 0;

    if (!facts.route_evidence_loaded) {
        checks.push_back({"Verified routes", DoctorStatus::Fail,
                          facts.route_evidence_message +
                              " - verify this machine's devices (Engine Preferences > Devices, or the installer); "
                              "training jobs are refused until then"});
    } else if (certified == 0) {
        checks.push_back({"Verified routes", DoctorStatus::Fail,
                          "no route passed verification; training jobs are refused"});
    } else {
        std::string names;
        for (const auto& entry : routes) {
            if (!entry.route.certified) continue;
            names += (names.empty() ? "" : ", ") + RouteName(entry.route);
        }
        checks.push_back({"Verified routes", DoctorStatus::Ok, std::to_string(certified) + " (" + names + ")"});
    }

    if (certified > 0) {
        std::string failed;
        for (const auto& entry : routes) {
            if (entry.route.certified && entry.benchmark && entry.benchmark_current && !entry.benchmark->ok) {
                failed += (failed.empty() ? "" : "; ") + RouteName(entry.route) + ": " + entry.benchmark->error;
            }
        }
        if (facts.capability.compute_score <= 0.0) {
            checks.push_back({"Training benchmark", DoctorStatus::Warn,
                              "no current measurement; the node reports compute score 0 - run "
                              "cyxwiz-server-daemon --benchmark" +
                                  (failed.empty() ? std::string{} : " (failed: " + failed + ")")});
        } else {
            checks.push_back({"Training benchmark", failed.empty() ? DoctorStatus::Ok : DoctorStatus::Warn,
                              fmt::format("{:.0f} tokens/s on {}", facts.capability.compute_score,
                                          facts.capability.compute_score_route) +
                                  (failed.empty() ? std::string{} : "; failed on " + failed)});
        }
    }

    if (!facts.preferred_route) {
        checks.push_back({"Compute preference", DoctorStatus::Warn,
                          facts.preference_file_loaded
                              ? "no preferred route saved; jobs use the default device"
                              : "no machine compute preference; jobs use the default device"});
    } else {
        const auto& [backend, device_id] = *facts.preferred_route;
        bool verified = false;
        for (const auto& entry : routes) {
            verified = verified || (entry.route.certified &&
                                    ExecutionDeviceSelectionBackendName(entry.route.type) == backend &&
                                    entry.route.device_id == device_id);
        }
        const std::string name = backend + ":" + std::to_string(device_id);
        checks.push_back(verified ? DoctorCheck{"Compute preference", DoctorStatus::Ok, name}
                                  : DoctorCheck{"Compute preference", DoctorStatus::Fail,
                                                name + " is not a verified route; jobs would be refused"});
    }

    if (!facts.data_dir_writable) {
        checks.push_back({"Job data folder", DoctorStatus::Fail,
                          facts.data_dir.string() + " is not writable: " + facts.data_dir_error});
    } else if (facts.data_dir_free_bytes < kDoctorMinimumFreeBytes) {
        checks.push_back({"Job data folder", DoctorStatus::Warn,
                          facts.data_dir.string() + " has " + Gigabytes(facts.data_dir_free_bytes) +
                              " free; remote datasets may not fit (set CYXWIZ_NODE_DATA_DIR to a data drive)"});
    } else {
        checks.push_back({"Job data folder", DoctorStatus::Ok,
                          facts.data_dir.string() + " (" + Gigabytes(facts.data_dir_free_bytes) + " free)"});
    }

    checks.push_back(facts.central_server_reachable
                         ? DoctorCheck{"Central server", DoctorStatus::Ok, facts.central_server + " reachable"}
                         : DoctorCheck{"Central server", DoctorStatus::Warn,
                                       facts.central_server + " not reachable; direct Engine (P2P) jobs still work"});

    if (facts.tls_enabled && !facts.tls_files_present && !facts.tls_auto) {
        checks.push_back({"TLS", DoctorStatus::Fail, "enabled but the certificate or key file is missing"});
    } else {
        checks.push_back({"TLS", DoctorStatus::Ok,
                          facts.tls_enabled ? (facts.tls_files_present ? "certificate present" : "auto-generated")
                                            : "off"});
    }

    checks.push_back({"Environment", DoctorStatus::Ok,
                      "build " + facts.build + ", " + facts.capability.environment.os + ", fingerprint " +
                          facts.capability.environment.fingerprint.substr(0, 16)});
    return checks;
}

bool NodeIsReady(const std::vector<DoctorCheck>& checks) {
    for (const auto& check : checks) {
        if (check.status == DoctorStatus::Fail) return false;
    }
    return true;
}

DoctorFacts GatherDoctorFacts(const DoctorOptions& options) {
    DoctorFacts facts;
    facts.build = GetVersionString();

    const auto qualification = LoadAndInstallRouteQualificationSnapshot(GetRouteQualificationCachePath());
    facts.route_evidence_loaded = qualification.loaded;
    facts.route_evidence_message = qualification.message;
    facts.capability = DetectMachineCapability();

    const auto preference = LoadComputeRuntimeConfig(GetComputeRuntimeConfigPath());
    facts.preference_file_loaded = preference.loaded;
    if (preference.loaded && preference.config.preferred_route) {
        facts.preferred_route = std::make_pair(ExecutionDeviceSelectionBackendName(preference.config.preferred_route->type),
                                               preference.config.preferred_route->last_device_id);
    }

    facts.data_dir = NodeDataRoot();
    std::error_code ec;
    std::filesystem::create_directories(facts.data_dir, ec);
    const auto probe = facts.data_dir / ".doctor-write-probe";
    {
        std::ofstream out(probe, std::ios::binary | std::ios::trunc);
        facts.data_dir_writable = static_cast<bool>(out << "ok");
    }
    if (!facts.data_dir_writable) facts.data_dir_error = ec ? ec.message() : "cannot create a file";
    std::filesystem::remove(probe, ec);
    const auto space = std::filesystem::space(facts.data_dir, ec);
    if (!ec) facts.data_dir_free_bytes = space.available;

    facts.central_server = options.central_server;
    if (!options.central_server.empty()) {
        auto channel = grpc::CreateChannel(options.central_server, grpc::InsecureChannelCredentials());
        facts.central_server_reachable =
            channel->WaitForConnected(std::chrono::system_clock::now() + std::chrono::seconds(3));
    }

    facts.tls_enabled = options.tls_enabled;
    facts.tls_auto = options.tls_auto;
    facts.tls_files_present = !options.tls_cert_path.empty() && !options.tls_key_path.empty() &&
                              std::filesystem::exists(options.tls_cert_path, ec) &&
                              std::filesystem::exists(options.tls_key_path, ec);
    return facts;
}

}  // namespace cyxwiz::servernode
