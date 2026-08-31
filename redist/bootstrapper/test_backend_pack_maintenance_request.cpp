#include "backend_pack_maintenance_request.h"
#include "backend_pack_platform.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

class Fixture {
public:
    Fixture() {
#ifdef _WIN32
        const auto process_id = ::GetCurrentProcessId();
#else
        const auto process_id = 1;
#endif
        root = std::filesystem::temp_directory_path() /
            ("cyxwiz-maintenance-request-" +
             std::to_string(process_id) + "-" +
             std::to_string(++sequence));
        Touch(
            root / "base" / "base-v1" /
            cyxwiz::runtime::CurrentEngineExecutableName());
        Touch(root / "packs" / "opencl" / "opencl-v1" /
              "runtime" /
              cyxwiz::runtime::CurrentArrayFireBackendPluginName("opencl"));
        Save(ActiveWithOpenCl(1), root / "active-runtime.json");
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    static cyxwiz::runtime::ActiveRuntimeState ActiveWithOpenCl(
        std::uint64_t generation) {
        cyxwiz::runtime::ActiveRuntimeState state;
        state.runtime_set_id = "set-v1";
        state.generation = generation;
        state.base_pack_id = "base-v1";
        state.packs.push_back({"opencl", "opencl-v1"});
        return state;
    }

    static cyxwiz::runtime::ActiveRuntimeState CpuOnly(
        std::uint64_t generation) {
        auto state = ActiveWithOpenCl(generation);
        state.packs.clear();
        return state;
    }

    void Save(
        const cyxwiz::runtime::ActiveRuntimeState& state,
        const std::filesystem::path& path) {
        std::string error;
        if (!cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
                path, state, error)) {
            throw std::runtime_error(error);
        }
    }

    cyxwiz::runtime::ActiveRuntimeState Active() const {
        cyxwiz::runtime::ActiveRuntimeState state;
        std::string error;
        if (!cyxwiz::runtime::LoadActiveRuntimeState(
                root / "active-runtime.json", state, error)) {
            throw std::runtime_error(error);
        }
        return state;
    }

    static void Touch(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).put('\0');
    }

    std::filesystem::path root;
    static inline int sequence = 0;
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

cyxwiz::runtime::BackendPackMaintenanceRequest RemoveRequest(
    std::uint64_t generation = 1) {
    cyxwiz::runtime::BackendPackMaintenanceRequest request;
    request.action =
        cyxwiz::runtime::BackendPackMaintenanceAction::Remove;
    request.runtime_set_id = "set-v1";
    request.runtime_generation = generation;
    request.backend = "opencl";
    request.pack_id = "opencl-v1";
    return request;
}

cyxwiz::runtime::BackendPackMaintenanceRequest RepairRequest() {
    auto request = RemoveRequest();
    request.action =
        cyxwiz::runtime::BackendPackMaintenanceAction::Repair;
    return request;
}

}  // namespace

int main() {
    int failures = 0;
    {
        Fixture fixture;
        std::string error;
        const bool queued =
            cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, RepairRequest(), error);
        bool called = false;
        const auto applied =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                fixture.root, Fixture::ActiveWithOpenCl(1),
                [&](const auto& request, std::string& message) {
                    called = request.action == cyxwiz::runtime::
                            BackendPackMaintenanceAction::Repair &&
                        request.backend == "opencl" &&
                        request.pack_id == "opencl-v1";
                    message = "signed repair helper completed";
                    return called;
                });
        failures += !Expect(
            queued && called && applied.status == cyxwiz::runtime::
                BackendPackMaintenanceApplyStatus::Applied &&
                fixture.Active().generation == 1 &&
                !std::filesystem::exists(
                    cyxwiz::runtime::BackendPackMaintenanceRequestPath(
                        fixture.root)),
            "queued repair must dispatch the exact pack only after exit");
    }
    {
        Fixture fixture;
        std::string error;
        const bool queued =
            cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, RepairRequest(), error);
        const auto failed =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                fixture.root, Fixture::ActiveWithOpenCl(1));
        failures += !Expect(
            queued && failed.status == cyxwiz::runtime::
                BackendPackMaintenanceApplyStatus::Failed &&
                std::filesystem::exists(
                    cyxwiz::runtime::BackendPackMaintenanceRequestPath(
                        fixture.root)),
            "repair must remain queued when no exit-safe helper is connected");
    }
    {
        Fixture fixture;
        std::string error;
        const auto request = RemoveRequest();
        const bool queued =
            cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, request, error);
        const bool duplicate =
            cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, request, error);
        const auto applied =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                fixture.root, Fixture::ActiveWithOpenCl(1));
        failures += !Expect(
            queued && !duplicate &&
                applied.status == cyxwiz::runtime::
                    BackendPackMaintenanceApplyStatus::Applied &&
                fixture.Active().generation == 2 &&
                fixture.Active().packs.empty() &&
                !std::filesystem::exists(
                    fixture.root / "packs" / "opencl" / "opencl-v1") &&
                !std::filesystem::exists(
                    cyxwiz::runtime::BackendPackMaintenanceRequestPath(
                        fixture.root)),
            "queued removal must apply only after the launched runtime exits");
    }
    {
        Fixture fixture;
        std::string error;
        const bool queued =
            cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, RemoveRequest(), error);
        const auto stale =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                fixture.root, Fixture::ActiveWithOpenCl(2));
        failures += !Expect(
            queued && stale.status == cyxwiz::runtime::
                BackendPackMaintenanceApplyStatus::StaleRequest &&
                std::filesystem::exists(
                    cyxwiz::runtime::BackendPackMaintenanceRequestPath(
                        fixture.root)) &&
                fixture.Active().generation == 1,
            "stale maintenance identity must not mutate or consume the request");
    }
    {
        Fixture fixture;
        fixture.Save(
            Fixture::CpuOnly(2), fixture.root / "active-runtime.json");
        fixture.Save(
            Fixture::ActiveWithOpenCl(1),
            fixture.root / "rollback" / "set-v1" /
                "previous-active-runtime.json");
        cyxwiz::runtime::BackendPackMaintenanceRequest request;
        request.action =
            cyxwiz::runtime::BackendPackMaintenanceAction::Rollback;
        request.runtime_set_id = "set-v1";
        request.runtime_generation = 2;
        std::string error;
        const bool queued =
            cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, request, error);
        const auto applied =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                fixture.root, Fixture::CpuOnly(2));
        const auto active = fixture.Active();
        failures += !Expect(
            queued && applied.status == cyxwiz::runtime::
                BackendPackMaintenanceApplyStatus::Applied &&
                active.generation == 3 && active.packs.size() == 1 &&
                active.packs.front().pack_id == "opencl-v1",
            "queued rollback must restore a validated complete runtime after exit");
    }
    {
        Fixture fixture;
        fixture.Save(
            Fixture::CpuOnly(2), fixture.root / "active-runtime.json");
        auto foreign = Fixture::ActiveWithOpenCl(1);
        foreign.runtime_set_id = "set-v2";
        fixture.Save(
            foreign,
            fixture.root / "rollback" / "set-v1" /
                "previous-active-runtime.json");
        cyxwiz::runtime::BackendPackMaintenanceRequest request;
        request.action =
            cyxwiz::runtime::BackendPackMaintenanceAction::Rollback;
        request.runtime_set_id = "set-v1";
        request.runtime_generation = 2;
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::QueueBackendPackMaintenanceRequest(
                fixture.root, request, error),
            "rollback request must reject state from another runtime set");
    }
    {
        Fixture fixture;
        const auto path =
            cyxwiz::runtime::BackendPackMaintenanceRequestPath(fixture.root);
        std::ofstream(path)
            << R"({"schema_version":1,"action":"remove","runtime_set_id":"set-v1","runtime_generation":1,"backend":"opencl","pack_id":"opencl-v1","unknown":true})";
        cyxwiz::runtime::BackendPackMaintenanceRequest request;
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::LoadBackendPackMaintenanceRequest(
                fixture.root, request, error),
            "maintenance request parser must reject unknown fields");
    }

    if (failures == 0) {
        std::cout << "Backend pack maintenance request tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
