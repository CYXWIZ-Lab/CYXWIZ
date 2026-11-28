#pragma once

// Internal to the backend and its tests; not part of the plugin SDK.
//
// How the OpenCL provider finds, among its own GPUs, the device ArrayFire's
// OpenCL backend is on. A pure function of what each side reports, so every
// case is tested with descriptors on any machine (test_opencl_provider_parity).
//
// The provider lists GPUs in platform order through its own OpenCL loader;
// ArrayFire orders its devices itself, and its cl_device_id may come from
// another ICD loader or a layered driver, so pointer equality alone can miss
// the device it is on. In order:
//  1. ArrayFire says the device is not a GPU (afcl_get_device_type): declined,
//     the only decline by kind;
//  2. the same cl_device_id;
//  3. the same PCI address or UUID (cl_khr_pci_bus_info, cl_khr_device_uuid),
//     when exactly one GPU has it;
//  4. the same name and vendor, when exactly one GPU has them;
//  5. the provider's own index (ArrayFire's id as its position), the old
//     behaviour; the provider warns once when it falls back to it.

#include <string>
#include <vector>

namespace cyxwiz::opencl_match {

// An OpenCL device as one side reports it; a string is empty when unknown.
struct DeviceDescriptor {
    const void* id = nullptr;  // cl_device_id
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
    std::string why;  // when index is -1
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
    if (kind == DeviceKind::NotGpu) {
        match.by = MatchedBy::NotGpu;
        match.why = which + " is not one of the provider's GPU devices (an OpenCL CPU or accelerator device)";
        return match;
    }
    const auto found = [&](int index, MatchedBy by) {
        match.index = index;
        match.by = by;
        match.why.clear();
        return match;
    };
    if (af.id != nullptr) {
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
    if (!af.name.empty()) {
        int index = -1, count = 0;
        for (size_t i = 0; i < gpus.size(); ++i) {
            const bool vendor = af.vendor.empty() || gpus[i].vendor.empty() || gpus[i].vendor == af.vendor;
            if (gpus[i].name == af.name && vendor) {
                index = static_cast<int>(i);
                ++count;
            }
        }
        if (count == 1) return found(index, MatchedBy::NameAndVendor);
    }
    if (af_id >= 0 && static_cast<size_t>(af_id) < gpus.size()) return found(af_id, MatchedBy::Index);
    match.why = which + " matches none of the provider's " + std::to_string(gpus.size()) +
                " GPU devices (by identity, PCI address, UUID, name or index)";
    return match;
}

}  // namespace cyxwiz::opencl_match
