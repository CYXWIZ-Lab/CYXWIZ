#pragma once

// One presentation of compute routes for every verification screen (Engine
// Preferences > Compute devices, Installer > Verification). It groups exact
// routes under the physical device they run on, maps evidence to one shared
// status vocabulary, and keeps every fact the older views showed in Details.
// Pure data in, data out: no ImGui, no global state, no backend DLL calls.

#include "route_qualification_snapshot.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class ComputeRouteStatus {
    Verified,
    NotVerifiedYet,
    Failed,
    NotSupported,
    NeedsDriverUpdate,
    Verifying
};

const char* ComputeRouteStatusName(ComputeRouteStatus status);

// Engine-only context for a route; the installer leaves it unset.
struct ComputeRouteSelectionState {
    bool active_run = false;
    bool last_run = false;
    bool active = false;
    bool next_run = false;
    bool saved = false;
    bool selected = false;
    bool training_authorized = false;
    std::string training_authorization;   // status name
    std::string qualification_message;
    std::string authorization_message;
};

struct ComputeRouteInput {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    // Present when this process enumerated the route (Engine).
    std::optional<DeviceInfo> device;
    // Present when saved verification results cover the route.
    std::optional<RouteQualificationRecord> evidence;
    // Installed pack that supplies the route, e.g. "CUDA pack · 1.6 GiB".
    std::string pack_label;
    std::optional<ComputeRouteSelectionState> selection;
    bool verifying = false;
    bool verification_allowed = true;
    // False when the evidence names a pack that is not the active one.
    bool pack_active = true;
};

using ComputeDetail = std::pair<std::string, std::string>;

struct ComputeRouteView {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string route_label;   // "CUDA", "OpenCL", "oneAPI", "CPU"
    std::string pack_label;
    ComputeRouteStatus status = ComputeRouteStatus::NotVerifiedYet;
    std::string summary;
    std::vector<std::string> badges;
    std::vector<ComputeDetail> details;
    bool can_verify = false;
    std::string verify_label;
    bool selectable = false;
    bool recommended = false;
};

struct ComputeDeviceCard {
    std::string key;
    std::string title;
    std::string subtitle;
    std::string recommended_route;   // empty when none is verified
    std::string recommendation_reason;
    std::vector<ComputeRouteView> routes;
};

struct ComputeFastestRoute {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    double median_iteration_ms = 0.0;
};

std::vector<ComputeDeviceCard> BuildComputeDeviceCards(
    const std::vector<ComputeRouteInput>& routes,
    const std::optional<ComputeFastestRoute>& fastest = std::nullopt);

// Routes as the installer sees them: only saved evidence, no enumeration.
std::vector<ComputeRouteInput> ComputeRouteInputsFromEvidence(
    const RouteQualificationSnapshot& snapshot);

}  // namespace cyxwiz
