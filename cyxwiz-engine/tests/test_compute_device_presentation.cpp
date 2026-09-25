#include "../src/core/compute_device_presentation.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

cyxwiz::RouteQualificationRecord Record(cyxwiz::DeviceType type, int id,
                                        std::string name, double median_ms) {
    cyxwiz::RouteQualificationRecord r;
    r.type = type;
    r.device_id = id;
    r.display_name = std::move(name);
    r.operation_count = cyxwiz::kRouteQualificationOperationCount;
    r.pass_count = r.operation_count;
    r.certified = true;
    if (median_ms > 0.0) {
        r.benchmark_id = cyxwiz::kRoutePerformanceBenchmarkId;
        r.benchmark_sample_count = 5;
        r.benchmark_iterations_per_sample = 3;
        r.benchmark_median_iteration_ms = median_ms;
    }
    return r;
}

cyxwiz::ComputeRouteInput Input(cyxwiz::RouteQualificationRecord record,
                                std::string pack) {
    cyxwiz::ComputeRouteInput input;
    input.type = record.type;
    input.device_id = record.device_id;
    input.evidence = std::move(record);
    input.pack_label = std::move(pack);
    return input;
}

const cyxwiz::ComputeDeviceCard* Find(
    const std::vector<cyxwiz::ComputeDeviceCard>& cards, const std::string& title) {
    for (const auto& card : cards) {
        if (card.title == title) return &card;
    }
    return nullptr;
}

bool HasDetail(const cyxwiz::ComputeRouteView& view, const std::string& key) {
    return std::any_of(view.details.begin(), view.details.end(),
                       [&](const cyxwiz::ComputeDetail& d) { return d.first == key; });
}

// The dev box results from 2026-09-25 (tofix119).
void TestGroupsRoutesByPhysicalDevice() {
    using cyxwiz::DeviceType;
    auto cpu = Record(DeviceType::CPU, 0, "Intel", 6.84);
    auto cuda = Record(DeviceType::CUDA, 0, "NVIDIA_GeForce_GTX_1050_Ti", 0.455);
    cuda.physical_fingerprint = "uuid:5cfa";
    auto ocl_gtx = Record(DeviceType::OPENCL, 0, "NVIDIA_GeForce_GTX_1050_Ti", 0.92);
    auto ocl_uhd = Record(DeviceType::OPENCL, 1, "Intel(R)_UHD_Graphics_630", 12.0);
    ocl_uhd.device_kind = cyxwiz::DeviceKind::GPU;
    ocl_uhd.device_kind_known = true;
    auto ocl_cpu = Record(DeviceType::OPENCL, 2,
                          "Intel(R)_Core(TM)_i7-8750H_CPU_@ 2.20GHz", 8.33);
    ocl_cpu.device_kind = cyxwiz::DeviceKind::CPU;
    ocl_cpu.device_kind_known = true;
    auto oneapi = Record(DeviceType::ONEAPI, 0, "", 0.0);
    oneapi.certified = false;
    oneapi.pass_count = 3;
    oneapi.crash_count = 1;
    oneapi.not_run_count = oneapi.operation_count - 4;
    oneapi.failure.category = cyxwiz::RouteFailureCategory::ChildProcessCrash;
    oneapi.failure.stage = cyxwiz::RouteFailureStage::Operation;
    oneapi.failure.operation = "randu";
    oneapi.failure.probe_stage = "create_begin";
    oneapi.failure.error_code = -1073741819;
    oneapi.failure.observed_fact = "Operation 'randu' terminated its isolated child process";

    const auto cards = cyxwiz::BuildComputeDeviceCards(
        {Input(cpu, "Engine base"), Input(cuda, "CUDA pack"),
         Input(ocl_gtx, "OpenCL pack"), Input(ocl_uhd, "OpenCL pack"),
         Input(ocl_cpu, "OpenCL pack"), Input(oneapi, "oneAPI pack")},
        cyxwiz::ComputeFastestRoute{DeviceType::CUDA, 0, 0.455});

    Check(cards.size() == 4, "GTX, UHD, unnamed oneAPI and CPU cards");
    const auto* gtx = Find(cards, "NVIDIA GeForce GTX 1050 Ti");
    Check(gtx && gtx->routes.size() == 2, "CUDA and OpenCL share the GTX card");
    Check(gtx->routes[0].route_label == "CUDA", "CUDA is listed first on the GTX");
    Check(gtx->recommended_route == "CUDA", "fastest verified route recommended");
    Check(gtx->recommendation_reason.find("fastest verified route on this machine") !=
              std::string::npos, "machine-wide fastest is called out");

    const auto* host = Find(cards, "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz");
    Check(host && host->routes.size() == 2, "native CPU and OpenCL CPU share a card");
    Check(host->routes[0].type == DeviceType::CPU, "native CPU route first");
    Check(host->recommended_route == "CPU", "faster native CPU route recommended");
    Check(cards.back().title == host->title, "CPU card is listed last");

    const auto* unnamed = Find(cards, "oneAPI device 0");
    Check(unnamed != nullptr, "unnamed oneAPI route keeps its own card");
    const auto& crash = unnamed->routes.front();
    Check(crash.status == cyxwiz::ComputeRouteStatus::NotSupported,
          "a crash means not supported on this device");
    Check(crash.summary.find("randu") != std::string::npos, "summary names the operation");
    Check(HasDetail(crash, "Error code") && HasDetail(crash, "Probe stage") &&
              HasDetail(crash, "Operations") && HasDetail(crash, "Observed"),
          "failure facts are kept in Details");
    Check(unnamed->recommended_route.empty(), "nothing recommended without a verified route");
}

void TestOpenClPreferredOverOneApiOnIntelGpu() {
    using cyxwiz::DeviceType;
    auto ocl = Record(DeviceType::OPENCL, 0, "Intel(R) Iris(R) Xe Graphics", 9.0);
    auto oneapi = Record(DeviceType::ONEAPI, 0, "Intel(R) Iris(R) Xe Graphics", 4.0);
    const auto cards = cyxwiz::BuildComputeDeviceCards(
        {Input(oneapi, "oneAPI pack"), Input(ocl, "OpenCL pack")});
    Check(cards.size() == 1, "same Intel GPU forms one card");
    Check(cards.front().recommended_route == "OpenCL",
          "OpenCL preferred on Intel GPUs even when oneAPI measured faster");
    Check(cards.front().recommendation_reason.find("SYCL") != std::string::npos,
          "reason explains the oneAPI deferral");
}

void TestEnumeratedUnverifiedRouteCanBeVerified() {
    cyxwiz::ComputeRouteInput input;
    input.type = cyxwiz::DeviceType::OPENCL;
    input.device_id = 1;
    cyxwiz::DeviceInfo device;
    device.type = input.type;
    device.device_id = 1;
    device.name = "Intel(R) UHD Graphics 630";
    device.name_known = true;
    device.kind = cyxwiz::DeviceKind::GPU;
    device.device_selectable = true;
    input.device = device;
    cyxwiz::ComputeRouteSelectionState selection;
    selection.saved = true;
    input.selection = selection;
    const auto cards = cyxwiz::BuildComputeDeviceCards({input});
    const auto& route = cards.front().routes.front();
    Check(route.status == cyxwiz::ComputeRouteStatus::NotVerifiedYet, "no evidence yet");
    Check(route.can_verify && route.verify_label == "Verify", "offers a single-route Verify");
    Check(route.selectable, "enumerated selectable route");
    Check(std::find(route.badges.begin(), route.badges.end(), "Saved") != route.badges.end(),
          "selection badges are kept");
    Check(HasDetail(route, "PCI") && HasDetail(route, "Memory") &&
              HasDetail(route, "Selectable"),
          "device identity facts are kept in Details");
}

void TestInstallerInputsAreReadOnly() {
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.routes.push_back(Record(cyxwiz::DeviceType::CUDA, 0, "GPU", 1.0));
    const auto inputs = cyxwiz::ComputeRouteInputsFromEvidence(snapshot);
    const auto cards = cyxwiz::BuildComputeDeviceCards(inputs);
    Check(!cards.front().routes.front().can_verify,
          "the installer shows results but verifies in the Engine");
    Check(cards.front().routes.front().pack_label == "CUDA pack", "pack label from evidence");
}

}  // namespace

int main() {
    TestGroupsRoutesByPhysicalDevice();
    TestOpenClPreferredOverOneApiOnIntelGpu();
    TestEnumeratedUnverifiedRouteCanBeVerified();
    TestInstallerInputsAreReadOnly();
    std::cout << "Compute device presentation tests passed\n";
    return 0;
}
