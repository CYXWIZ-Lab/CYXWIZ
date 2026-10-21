#pragma once

// cyxwiz-server-daemon --doctor (TOFIX118 P3 S3): can this node take training
// jobs, and if not, why. Gathering the facts touches the machine; judging
// them is a pure function so the rules can be tested.

#include "core/machine_capability.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::servernode {

enum class DoctorStatus { Ok, Warn, Fail };

struct DoctorCheck {
    std::string name;
    DoctorStatus status = DoctorStatus::Ok;
    std::string detail;
};

struct DoctorFacts {
    std::string build;
    bool route_evidence_loaded = false;
    std::string route_evidence_message;
    MachineCapability capability;
    // Preferences > Devices: the saved preferred route, if any.
    bool preference_file_loaded = false;
    std::optional<std::pair<std::string, int>> preferred_route;  // backend, device id
    std::filesystem::path data_dir;
    bool data_dir_writable = false;
    std::string data_dir_error;
    std::uint64_t data_dir_free_bytes = 0;
    std::string central_server;
    bool central_server_reachable = false;
    bool tls_enabled = false;
    bool tls_auto = false;
    bool tls_files_present = false;
};

inline constexpr std::uint64_t kDoctorMinimumFreeBytes = 10ull * 1024 * 1024 * 1024;

std::vector<DoctorCheck> EvaluateNodeReadiness(const DoctorFacts& facts);
bool NodeIsReady(const std::vector<DoctorCheck>& checks);  // no Fail

struct DoctorOptions {
    std::string central_server;
    bool tls_enabled = false;
    bool tls_auto = false;
    std::string tls_cert_path;
    std::string tls_key_path;
};

// Loads the route evidence and preference, checks the data folder and
// probes the central server (3 s).
DoctorFacts GatherDoctorFacts(const DoctorOptions& options);

}  // namespace cyxwiz::servernode
