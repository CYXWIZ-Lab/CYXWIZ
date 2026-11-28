#pragma once

// Internal to the backend and its tests; not part of the plugin SDK.
//
// How the OpenCL provider finds, among its own GPUs, the device ArrayFire's
// OpenCL backend is on. Pure functions of what each side reports, so every
// case is tested with descriptors on any machine (test_opencl_provider_parity),
// and the provider's per-process memory of the answers (IdentifiedDevices).
//
// The provider lists GPUs in platform order through its own OpenCL loader;
// ArrayFire orders its devices itself, and its cl_device_id may come from
// another ICD loader or a layered driver, so pointer equality alone can miss
// the device it is on. In order:
//  1. ArrayFire says the device is not a GPU (afcl_get_device_type): declined;
//  2. the same cl_device_id;
//  3. the same PCI address or UUID (cl_khr_pci_bus_info, cl_khr_device_uuid),
//     when exactly one GPU has it;
//  4. the same name and vendor, when exactly one GPU has them and its PCI
//     address and UUID do not contradict ArrayFire's;
//  5. the provider's own index (ArrayFire's id as its position), only where
//     the device cannot be told otherwise, and then marked uncertain (the
//     provider warns once that the physical GPU is not certain):
//     - ArrayFire's cl_device_id could not be read, and ArrayFire says the
//       device is a GPU;
//     - identical twins: several GPUs have its name and vendor and no PCI
//       address or UUID tells them apart, and the position is one of them.
// Anything else is declined with why: a device that was read and matches no
// provider GPU, or one whose kind ArrayFire could not report
// (afcl_get_device_type missing) and that no name and vendor match.

#include <algorithm>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::opencl_match {

// An OpenCL device as one side reports it; a string is empty when unknown.
struct DeviceDescriptor {
    const void* id = nullptr;  // cl_device_id; null: ArrayFire's could not be read
    std::string name;          // CL_DEVICE_NAME
    std::string vendor;        // CL_DEVICE_VENDOR
    std::string pci;           // "domain:bus:device.function"
    std::string uuid;          // hex
};

// What afcl_get_device_type said; Unknown when it could not be asked.
enum class DeviceKind { Gpu, NotGpu, Unknown };

enum class MatchedBy { Identity, PciOrUuid, NameAndVendor, Index, NotGpu, None };

struct Match {
    int index = -1;  // into the provider's GPUs; -1: none of them
    MatchedBy by = MatchedBy::None;
    std::string why;        // when index is -1
    std::string uncertain;  // by Index: why the physical GPU is not certain
};

inline const char* MatchedByName(MatchedBy by) {
    switch (by) {
        case MatchedBy::Identity: return "identity";
        case MatchedBy::PciOrUuid: return "pci or uuid";
        case MatchedBy::NameAndVendor: return "name and vendor";
        case MatchedBy::Index: return "index";
        case MatchedBy::NotGpu: return "not a gpu";
        case MatchedBy::None: return "none";
    }
    return "none";
}

// af_id: ArrayFire's OpenCL device id; af: what ArrayFire's cl_device_id
// reports (id null when it could not be read); gpus: the provider's GPUs.
inline Match MatchArrayFireDevice(int af_id, DeviceKind kind, const DeviceDescriptor& af,
                                  const std::vector<DeviceDescriptor>& gpus) {
    Match match;
    const std::string which = "ArrayFire OpenCL device " + std::to_string(af_id);
    const std::string provider_gpus = "the provider's " + std::to_string(gpus.size()) + " GPU devices";
    if (kind == DeviceKind::NotGpu) {
        match.by = MatchedBy::NotGpu;
        match.why = which + " is not one of the provider's GPU devices (an OpenCL CPU or accelerator device)";
        return match;
    }
    const auto found = [&](int index, MatchedBy by, std::string uncertain = {}) {
        match.index = index;
        match.by = by;
        match.why.clear();
        match.uncertain = std::move(uncertain);
        return match;
    };
    const bool read = af.id != nullptr;
    if (read) {
        for (size_t i = 0; i < gpus.size(); ++i)
            if (gpus[i].id == af.id) return found(static_cast<int>(i), MatchedBy::Identity);
    }
    // The one GPU whose `field` equals ArrayFire's, or -1 (none, several, or unknown)
    const auto unique = [&](std::string DeviceDescriptor::*field) {
        if ((af.*field).empty()) return -1;
        int index = -1;
        for (size_t i = 0; i < gpus.size(); ++i) {
            if (gpus[i].*field != af.*field) continue;
            if (index >= 0) return -1;
            index = static_cast<int>(i);
        }
        return index;
    };
    if (const int i = unique(&DeviceDescriptor::pci); i >= 0) return found(i, MatchedBy::PciOrUuid);
    if (const int i = unique(&DeviceDescriptor::uuid); i >= 0) return found(i, MatchedBy::PciOrUuid);
    // A GPU whose PCI address or UUID both sides report, and differ, is another device
    const auto contradicts = [&](const DeviceDescriptor& gpu) {
        return (!af.pci.empty() && !gpu.pci.empty() && gpu.pci != af.pci) ||
               (!af.uuid.empty() && !gpu.uuid.empty() && gpu.uuid != af.uuid);
    };
    std::vector<int> named;  // the GPUs with ArrayFire's name and vendor that it may be
    if (!af.name.empty()) {
        for (size_t i = 0; i < gpus.size(); ++i) {
            const bool vendor = af.vendor.empty() || gpus[i].vendor.empty() || gpus[i].vendor == af.vendor;
            if (gpus[i].name == af.name && vendor && !contradicts(gpus[i])) named.push_back(static_cast<int>(i));
        }
    }
    if (named.size() == 1) return found(named.front(), MatchedBy::NameAndVendor);
    const bool in_range = af_id >= 0 && static_cast<size_t>(af_id) < gpus.size();
    if (!read) {
        if (kind == DeviceKind::Gpu && in_range)
            return found(af_id, MatchedBy::Index,
                         "ArrayFire's OpenCL device handle could not be read, so the GPU is taken by its position");
        match.why = kind == DeviceKind::Unknown
                        ? which + " cannot be identified: ArrayFire reported neither its device type nor its OpenCL "
                                  "device, so it may not be a GPU"
                        : which + " is past " + provider_gpus + ", and its OpenCL device handle could not be read";
        return match;
    }
    if (named.size() > 1) {
        if (in_range && std::find(named.begin(), named.end(), af_id) != named.end())
            return found(af_id, MatchedBy::Index,
                         std::to_string(named.size()) + " GPUs are named " + af.name +
                             " and no PCI address or UUID tells them apart");
        match.why = which + " (" + af.name + ") is one of " + std::to_string(named.size()) +
                    " identical GPUs, and its position is none of them";
        return match;
    }
    match.why = which + " (" + (af.name.empty() ? std::string("no name reported") : af.name) + ") matches none of " +
                provider_gpus + " (by identity, PCI address, UUID, or name and vendor)";
    return match;
}

// The warning for a match by position: which GPU, and why it is uncertain.
inline std::string UncertainMatchWarning(int af_id, const Match& match, const std::vector<DeviceDescriptor>& gpus) {
    const std::string name = match.index >= 0 && static_cast<size_t>(match.index) < gpus.size()
                                 ? gpus[static_cast<size_t>(match.index)].name
                                 : std::string();
    return "OpenCL provider: ArrayFire OpenCL device " + std::to_string(af_id) + " is taken to be the provider's GPU " +
           std::to_string(match.index) + (name.empty() ? std::string() : " (" + name + ")") + " by its position: " +
           match.uncertain + "; which physical GPU runs it is uncertain";
}

// The provider's answer for each ArrayFire OpenCL device, for the process:
// identified once, at the first call on the device, then a lookup. An
// uncertain answer is warned about once, by the call that keeps it.
class IdentifiedDevices {
public:
    std::optional<Match> Find(int af_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto known = known_.find(af_id);
        if (known == known_.end()) return std::nullopt;
        return known->second;
    }

    // Keeps `match` for af_id unless another call kept one first, and returns
    // the kept answer. `warn` gets UncertainMatchWarning when this call kept
    // an uncertain match.
    Match Keep(int af_id, Match match, const std::vector<DeviceDescriptor>& gpus,
               const std::function<void(const std::string&)>& warn) {
        bool kept_now = false;
        Match kept;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto [entry, inserted] = known_.emplace(af_id, std::move(match));
            kept_now = inserted;
            kept = entry->second;
        }
        if (kept_now && !kept.uncertain.empty() && warn) warn(UncertainMatchWarning(af_id, kept, gpus));
        return kept;
    }

private:
    mutable std::mutex mutex_;
    std::map<int, Match> known_;
};

}  // namespace cyxwiz::opencl_match
