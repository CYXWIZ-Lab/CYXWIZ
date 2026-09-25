#include "compute_device_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

namespace cyxwiz {
namespace {

const char* RouteLabel(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::ONEAPI: return "oneAPI";
        case DeviceType::METAL: return "Metal";
        case DeviceType::VULKAN: return "Vulkan";
    }
    return "Unknown";
}

const char* KindLabel(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::CPU: return "CPU";
        case DeviceKind::GPU: return "GPU";
        case DeviceKind::Accelerator: return "Accelerator";
        case DeviceKind::Unknown: break;
    }
    return "Unknown";
}

const char* ConfidenceLabel(DeviceIdentityConfidence confidence) {
    switch (confidence) {
        case DeviceIdentityConfidence::StableHardware: return "Stable hardware";
        case DeviceIdentityConfidence::ProviderReported: return "Provider reported";
        case DeviceIdentityConfidence::BackendLocal: return "Backend local";
        case DeviceIdentityConfidence::Unknown: break;
    }
    return "Unknown";
}

const char* MetadataLabel(DeviceMetadataStatus status) {
    switch (status) {
        case DeviceMetadataStatus::Available: return "Available";
        case DeviceMetadataStatus::Unsupported: return "Not reported by this runtime";
        case DeviceMetadataStatus::Failed: return "Query failed";
        case DeviceMetadataStatus::NotQueried: break;
    }
    return "Not queried";
}

// ArrayFire reports names like "NVIDIA_GeForce_GTX_1050_Ti".
std::string CleanName(std::string name) {
    std::replace(name.begin(), name.end(), '_', ' ');
    const auto first = name.find_first_not_of(' ');
    if (first == std::string::npos) return {};
    const auto last = name.find_last_not_of(' ');
    return name.substr(first, last - first + 1);
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    return value;
}

std::string FormatMs(double ms) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), ms < 1.0 ? "%.2f ms" : "%.1f ms", ms);
    return buffer;
}

std::string FormatGb(size_t bytes) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f GB",
                  static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buffer;
}

std::string RouteName(const ComputeRouteInput& input) {
    if (input.device && input.device->name_known &&
        !input.device->name_is_fallback) {
        return CleanName(input.device->name);
    }
    if (input.evidence && !input.evidence->display_name.empty()) {
        return CleanName(input.evidence->display_name);
    }
    return {};
}

std::string RouteFingerprint(const ComputeRouteInput& input) {
    if (input.device && input.device->physical_fingerprint_known) {
        return Lower(input.device->physical_fingerprint);
    }
    if (input.evidence) return Lower(input.evidence->physical_fingerprint);
    return {};
}

DeviceKind RouteKind(const ComputeRouteInput& input) {
    if (input.type == DeviceType::CPU) return DeviceKind::CPU;
    if (input.device && input.device->kind != DeviceKind::Unknown) {
        return input.device->kind;
    }
    if (input.evidence && input.evidence->device_kind_known) {
        return input.evidence->device_kind;
    }
    return DeviceKind::Unknown;
}

int RouteOrder(DeviceType type, bool cpu_card) {
    if (cpu_card) {
        switch (type) {
            case DeviceType::CPU: return 0;
            case DeviceType::OPENCL: return 1;
            case DeviceType::ONEAPI: return 2;
            default: return 3;
        }
    }
    switch (type) {
        case DeviceType::CUDA: return 0;
        case DeviceType::OPENCL: return 1;
        case DeviceType::ONEAPI: return 2;
        case DeviceType::CPU: return 3;
        default: return 4;
    }
}

int PreferenceRank(DeviceType type) {
    switch (type) {
        case DeviceType::CUDA: return 0;
        case DeviceType::OPENCL: return 1;
        case DeviceType::CPU: return 2;
        case DeviceType::ONEAPI: return 3;
        default: return 4;
    }
}

void AddDetail(std::vector<ComputeDetail>& details, const char* key,
               const std::string& value) {
    if (!value.empty()) details.emplace_back(key, value);
}

std::string OperationsSummary(const RouteQualificationRecord& e) {
    std::ostringstream out;
    out << e.pass_count << " passed";
    if (e.failure_count > 0) out << ", " << e.failure_count << " failed";
    if (e.timeout_count > 0) out << ", " << e.timeout_count << " timed out";
    if (e.crash_count > 0) out << ", " << e.crash_count << " crashed";
    if (e.unavailable_count > 0) out << ", " << e.unavailable_count << " unavailable";
    if (e.not_run_count > 0) out << ", " << e.not_run_count << " not run";
    out << " of " << e.operation_count;
    return out.str();
}

void Classify(const ComputeRouteInput& input, ComputeRouteView& view) {
    if (input.verifying) {
        view.status = ComputeRouteStatus::Verifying;
        view.summary = "Running the verification check in an isolated process...";
        return;
    }
    if (!input.evidence) {
        view.status = ComputeRouteStatus::NotVerifiedYet;
        view.summary = input.device
            ? "Detected on this machine. Verify this route before training on it."
            : "No saved verification result for this route.";
        return;
    }
    const auto& e = *input.evidence;
    const std::string op = e.failure.operation.empty()
        ? std::string("an operation") : e.failure.operation;
    const std::string passed = std::to_string(e.pass_count) + " of " +
        std::to_string(e.operation_count) + " operations passed";
    if (e.certified) {
        view.status = ComputeRouteStatus::Verified;
        view.summary = "All " + std::to_string(e.operation_count) +
            " operations passed";
        if (e.benchmark_median_iteration_ms > 0.0) {
            view.summary += " · median " + FormatMs(e.benchmark_median_iteration_ms);
        }
        view.summary += ".";
        return;
    }
    if (e.crash_count > 0) {
        view.status = ComputeRouteStatus::NotSupported;
        view.summary = "Crashed during verification in " + op + "; " + passed +
            ". The Engine kept running and this route is disabled.";
        return;
    }
    if (e.failure.category == RouteFailureCategory::ProviderMissing ||
        e.failure.category == RouteFailureCategory::DependencyMissing) {
        view.status = ComputeRouteStatus::NeedsDriverUpdate;
        view.summary = (e.failure.observed_fact.empty()
                            ? std::string("The device driver or runtime is missing")
                            : e.failure.observed_fact) +
            ". Install or update the device driver, then verify again.";
        return;
    }
    view.status = ComputeRouteStatus::Failed;
    if (e.timeout_count > 0) {
        const int seconds = e.failure.timeout_ms / 1000;
        view.summary = "Timed out in " + op +
            (seconds > 0 ? " after " + std::to_string(seconds) + " s" : "") +
            "; " + passed + ".";
    } else {
        view.summary = "Failed in " + op + "; " + passed + ".";
    }
}

void FillDetails(const ComputeRouteInput& input, ComputeRouteView& view) {
    auto& d = view.details;
    AddDetail(d, "Backend", RouteLabel(input.type));
    AddDetail(d, "Backend device ID", std::to_string(input.device_id));
    AddDetail(d, "Pack", view.pack_label);
    if (input.evidence) AddDetail(d, "Pack ID", input.evidence->pack_id);
    if (input.device) {
        const auto& dev = *input.device;
        AddDetail(d, "Name", dev.name);
        std::string source = input.evidence && !input.evidence->identity_source.empty()
            ? input.evidence->identity_source
            : dev.name_is_fallback ? "Fallback label" : "Provider metadata";
        AddDetail(d, "Name source", source);
        AddDetail(d, "Device kind", KindLabel(dev.kind));
        AddDetail(d, "Identity", ConfidenceLabel(dev.identity_confidence));
        AddDetail(d, "Provider", dev.provider_known ? dev.provider : "Unknown");
        AddDetail(d, "Driver", dev.driver_version_known ? dev.driver_version : "Unknown");
        if (dev.pci_location_known) {
            char pci[32];
            std::snprintf(pci, sizeof(pci), "%04x:%02x:%02x.%x", dev.pci_domain,
                          dev.pci_bus, dev.pci_device, dev.pci_function);
            AddDetail(d, "PCI", pci);
        } else {
            AddDetail(d, "PCI", "Unknown");
        }
        AddDetail(d, "Physical identity", dev.physical_fingerprint_known
                                              ? dev.physical_fingerprint : "Unknown");
        AddDetail(d, "Metadata", MetadataLabel(dev.metadata_status));
        AddDetail(d, "Selectable", dev.device_selectable ? "Yes" : "No");
        if (dev.memory_total_known) {
            AddDetail(d, "Memory", dev.memory_available_known
                ? FormatGb(dev.memory_total) + " total, " +
                      FormatGb(dev.memory_available) + " available"
                : FormatGb(dev.memory_total) + " total, available unknown");
        } else {
            AddDetail(d, "Memory", "Unknown");
        }
    } else if (input.evidence) {
        AddDetail(d, "Name", input.evidence->display_name);
        AddDetail(d, "Physical identity", input.evidence->physical_fingerprint);
        AddDetail(d, "Provider", input.evidence->provider);
        AddDetail(d, "Driver", input.evidence->driver_version);
    }
    if (input.selection) {
        AddDetail(d, "Training authorization", input.selection->training_authorization);
        AddDetail(d, "Qualification evidence", input.selection->qualification_message);
        AddDetail(d, "Authorization policy", input.selection->authorization_message);
    }
    if (!input.evidence) return;
    const auto& e = *input.evidence;
    AddDetail(d, "Operations", OperationsSummary(e));
    AddDetail(d, "Runtime version", e.runtime_version);
    if (e.failure.category != RouteFailureCategory::None) {
        AddDetail(d, "Failure category", RouteFailureCategoryName(e.failure.category));
        AddDetail(d, "Failure stage", RouteFailureStageName(e.failure.stage));
    }
    AddDetail(d, "Failed operation", e.failure.operation);
    AddDetail(d, "Probe stage", e.failure.probe_stage);
    if (e.failure.error_code != 0) {
        char code[48];
        std::snprintf(code, sizeof(code), "%d (0x%08X)", e.failure.error_code,
                      static_cast<unsigned>(e.failure.error_code));
        AddDetail(d, "Error code", code);
    }
    if (e.failure.timeout_ms > 0) {
        AddDetail(d, "Timeout", std::to_string(e.failure.timeout_ms) + " ms");
    }
    AddDetail(d, "Observed", e.failure.observed_fact);
    AddDetail(d, "Interpretation", e.failure.bounded_interpretation);
    AddDetail(d, "Recommended action", e.failure.recommended_action);
    AddDetail(d, "Evidence", e.failure.evidence_id);
    if (!e.benchmark_id.empty()) {
        AddDetail(d, "Benchmark",
                  e.benchmark_id + " · median " +
                      FormatMs(e.benchmark_median_iteration_ms) + " · " +
                      std::to_string(e.benchmark_sample_count) + " samples × " +
                      std::to_string(e.benchmark_iterations_per_sample) +
                      " iterations");
    }
    AddDetail(d, "Benchmark note", e.benchmark_message);
}

void FillBadges(const ComputeRouteInput& input, ComputeRouteView& view) {
    if (input.selection) {
        const auto& s = *input.selection;
        if (s.active_run) view.badges.emplace_back("Active run");
        else if (s.last_run) view.badges.emplace_back("Last run");
        else if (s.active) view.badges.emplace_back("Active");
        if (s.next_run) view.badges.emplace_back("Next run");
        if (s.saved) view.badges.emplace_back("Saved");
        if (s.selected) view.badges.emplace_back("Selected");
        if (view.status == ComputeRouteStatus::Verified) {
            view.badges.emplace_back(s.training_authorized ? "Training ready"
                                                           : "Diagnostic only");
        }
    }
    if (!input.pack_active) view.badges.emplace_back("Pack not active");
    if (input.device) {
        if (input.device->name_is_fallback) view.badges.emplace_back("Fallback name");
        if (input.device->metadata_status == DeviceMetadataStatus::Unsupported ||
            input.device->metadata_status == DeviceMetadataStatus::Failed) {
            view.badges.emplace_back("Metadata limited");
        }
    }
}

struct CardBuild {
    ComputeDeviceCard card;
    std::vector<std::string> fingerprints;
    std::vector<std::string> names;
    bool cpu = false;
    DeviceKind kind = DeviceKind::Unknown;
    std::vector<ComputeRouteInput> inputs;
};

bool Contains(const std::vector<std::string>& values, const std::string& value) {
    return !value.empty() &&
        std::find(values.begin(), values.end(), value) != values.end();
}

void Recommend(CardBuild& build, const std::optional<ComputeFastestRoute>& fastest) {
    auto& card = build.card;
    const bool opencl_verified = std::any_of(
        card.routes.begin(), card.routes.end(), [](const ComputeRouteView& r) {
            return r.type == DeviceType::OPENCL &&
                r.status == ComputeRouteStatus::Verified;
        });
    const bool has_oneapi = std::any_of(
        card.routes.begin(), card.routes.end(),
        [](const ComputeRouteView& r) { return r.type == DeviceType::ONEAPI; });
    ComputeRouteView* best = nullptr;
    double best_ms = 0.0;
    for (size_t i = 0; i < card.routes.size(); ++i) {
        auto& route = card.routes[i];
        if (route.status != ComputeRouteStatus::Verified) continue;
        // On Intel GPUs OpenCL is preferred until ArrayFire exposes its SYCL
        // queue and the oneAPI provider can share buffers.
        if (route.type == DeviceType::ONEAPI && opencl_verified) continue;
        const auto& evidence = build.inputs[i].evidence;
        const double ms = evidence ? evidence->benchmark_median_iteration_ms : 0.0;
        if (!best) {
            best = &route;
            best_ms = ms;
            continue;
        }
        const bool both_measured = ms > 0.0 && best_ms > 0.0;
        if ((both_measured && ms < best_ms) ||
            (!both_measured && ms > 0.0 && best_ms <= 0.0) ||
            (!both_measured && (ms > 0.0) == (best_ms > 0.0) &&
             PreferenceRank(route.type) < PreferenceRank(best->type))) {
            best = &route;
            best_ms = ms;
        }
    }
    if (!best) {
        const bool any_unverified = std::any_of(
            card.routes.begin(), card.routes.end(), [](const ComputeRouteView& r) {
                return r.status == ComputeRouteStatus::NotVerifiedYet;
            });
        card.recommendation_reason = any_unverified
            ? "Not verified yet. Verify the routes on this device before training on it."
            : "No route on this device passed verification; training uses a verified route on another device.";
        return;
    }
    best->recommended = true;
    card.recommended_route = best->route_label;
    std::string reason = best->route_label + " is the verified route for training on this device";
    if (best_ms > 0.0) reason += " (median " + FormatMs(best_ms) + ")";
    reason += ".";
    if (best->type == DeviceType::OPENCL && has_oneapi) {
        reason += " OpenCL is preferred over oneAPI here until ArrayFire exposes its SYCL queue.";
    }
    std::vector<std::string> others;
    for (const auto& route : card.routes) {
        if (&route != best && route.status == ComputeRouteStatus::Verified) {
            others.push_back(route.route_label);
        }
    }
    if (!others.empty()) {
        reason += " Also verified: ";
        for (size_t i = 0; i < others.size(); ++i) {
            reason += (i ? ", " : "") + others[i];
        }
        reason += ".";
    }
    if (fastest && fastest->type == best->type &&
        fastest->device_id == best->device_id) {
        reason += " This is the fastest verified route on this machine.";
    }
    card.recommendation_reason = reason;
}

int KindRank(DeviceKind kind, bool cpu) {
    if (cpu) return 3;
    switch (kind) {
        case DeviceKind::GPU: return 0;
        case DeviceKind::Accelerator: return 1;
        default: return 2;
    }
}

}  // namespace

const char* ComputeRouteStatusName(ComputeRouteStatus status) {
    switch (status) {
        case ComputeRouteStatus::Verified: return "Verified";
        case ComputeRouteStatus::NotVerifiedYet: return "Not verified yet";
        case ComputeRouteStatus::Failed: return "Failed";
        case ComputeRouteStatus::NotSupported: return "Not supported on this device";
        case ComputeRouteStatus::NeedsDriverUpdate: return "Needs driver update";
        case ComputeRouteStatus::Verifying: return "Verifying...";
    }
    return "Not verified yet";
}

std::vector<ComputeDeviceCard> BuildComputeDeviceCards(
    const std::vector<ComputeRouteInput>& routes,
    const std::optional<ComputeFastestRoute>& fastest) {
    std::vector<CardBuild> builds;
    for (const auto& input : routes) {
        const std::string fingerprint = RouteFingerprint(input);
        const std::string name = RouteName(input);
        const std::string name_key = Lower(name);
        const DeviceKind kind = RouteKind(input);
        const bool cpu = kind == DeviceKind::CPU;
        CardBuild* target = nullptr;
        for (auto& build : builds) {
            if ((cpu && build.cpu) || Contains(build.fingerprints, fingerprint) ||
                (!cpu && !build.cpu && Contains(build.names, name_key))) {
                target = &build;
                break;
            }
        }
        if (!target) {
            builds.emplace_back();
            target = &builds.back();
            target->cpu = cpu;
            target->kind = kind;
        }
        if (!fingerprint.empty()) target->fingerprints.push_back(fingerprint);
        if (!name_key.empty()) target->names.push_back(name_key);
        if (target->kind == DeviceKind::Unknown) target->kind = kind;

        ComputeRouteView view;
        view.type = input.type;
        view.device_id = input.device_id;
        view.route_label = RouteLabel(input.type);
        view.pack_label = input.pack_label;
        Classify(input, view);
        FillBadges(input, view);
        FillDetails(input, view);
        view.can_verify = input.device.has_value() && input.verification_allowed &&
            !input.verifying;
        view.verify_label = view.status == ComputeRouteStatus::NotVerifiedYet
            ? "Verify" : "Verify again";
        view.selectable = input.device && input.device->device_selectable;
        target->card.routes.push_back(std::move(view));
        target->inputs.push_back(input);

        if (name.size() > target->card.title.size()) target->card.title = name;
    }

    std::vector<ComputeDeviceCard> cards;
    for (auto& build : builds) {
        auto& card = build.card;
        // Keep routes and their inputs aligned while ordering.
        std::vector<size_t> order(card.routes.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const auto& ra = card.routes[a];
            const auto& rb = card.routes[b];
            const int oa = RouteOrder(ra.type, build.cpu);
            const int ob = RouteOrder(rb.type, build.cpu);
            return oa != ob ? oa < ob : ra.device_id < rb.device_id;
        });
        std::vector<ComputeRouteView> routes_sorted;
        std::vector<ComputeRouteInput> inputs_sorted;
        for (size_t index : order) {
            routes_sorted.push_back(std::move(card.routes[index]));
            inputs_sorted.push_back(std::move(build.inputs[index]));
        }
        card.routes = std::move(routes_sorted);
        build.inputs = std::move(inputs_sorted);

        const auto& first = build.inputs.front();
        if (card.title.empty()) {
            card.title = build.cpu ? "Host CPU"
                : std::string(RouteLabel(first.type)) + " device " +
                      std::to_string(first.device_id);
        }
        card.key = build.cpu ? "cpu"
            : !build.fingerprints.empty() ? build.fingerprints.front()
            : "route:" + std::string(RouteLabel(first.type)) + ":" +
                  std::to_string(first.device_id);

        std::vector<std::string> parts;
        parts.emplace_back(build.cpu ? "Host processor" : KindLabel(build.kind));
        for (const auto& input : build.inputs) {
            if (input.device && input.device->provider_known &&
                !input.device->provider.empty()) {
                parts.push_back(input.device->provider);
                break;
            }
        }
        for (const auto& input : build.inputs) {
            if (input.device && input.device->memory_total_known &&
                input.device->memory_total > 0 && !build.cpu) {
                parts.push_back(FormatGb(input.device->memory_total));
                break;
            }
        }
        if (build.names.empty() && !build.cpu) {
            parts.emplace_back("details not reported by this runtime");
        }
        for (size_t i = 0; i < parts.size(); ++i) {
            card.subtitle += (i ? " · " : "") + parts[i];
        }
        Recommend(build, fastest);
    }
    std::stable_sort(builds.begin(), builds.end(),
                     [](const CardBuild& a, const CardBuild& b) {
                         const int ka = KindRank(a.kind, a.cpu);
                         const int kb = KindRank(b.kind, b.cpu);
                         return ka != kb ? ka < kb : a.card.title < b.card.title;
                     });
    for (auto& build : builds) cards.push_back(std::move(build.card));
    return cards;
}

std::vector<ComputeRouteInput> ComputeRouteInputsFromEvidence(
    const RouteQualificationSnapshot& snapshot) {
    std::vector<ComputeRouteInput> inputs;
    for (const auto& record : snapshot.routes) {
        ComputeRouteInput input;
        input.type = record.type;
        input.device_id = record.device_id;
        input.evidence = record;
        input.pack_label = record.type == DeviceType::CPU
            ? "Engine base" : std::string(RouteLabel(record.type)) + " pack";
        input.verification_allowed = false;
        inputs.push_back(std::move(input));
    }
    return inputs;
}

}  // namespace cyxwiz
