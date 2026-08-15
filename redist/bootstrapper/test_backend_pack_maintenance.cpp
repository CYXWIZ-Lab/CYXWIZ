#include "backend_pack_installer.h"
#include "backend_pack_platform.h"
#include "backend_pack_remover.h"
#include "runtime_mutation_gate.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

constexpr const char* kZeroByteSha256 =
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d";

class Fixture {
public:
    Fixture() {
#ifdef _WIN32
        const auto process_id = ::GetCurrentProcessId();
#else
        const auto process_id = 1;
#endif
        root = std::filesystem::temp_directory_path() /
            ("cyxwiz-pack-maintenance-test-" +
             std::to_string(process_id) + "-" +
             std::to_string(++sequence));
        Touch(
            root / "base" / "base-v1" /
            cyxwiz::runtime::CurrentEngineExecutableName());
        source = root.parent_path() /
            (root.filename().string() + "-source");
        Touch(source / "runtime" / "afopencl.dll");
        Touch(source / "THIRD_PARTY_LICENSES" / "ArrayFire" /
              "LICENSE.txt");
        Active({});
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
        std::filesystem::remove_all(source, error);
    }

    cyxwiz::runtime::VerifiedBackendPackPayload Payload() const {
        cyxwiz::runtime::VerifiedBackendPackPayload payload;
        payload.runtime_set_id = "set-v1";
        payload.companion_base_id = "base-v1";
        payload.backend = "opencl";
        payload.pack_id = "opencl-v1";
        payload.source_directory = source;
        payload.components = {
            {"runtime/afopencl.dll", 1, kZeroByteSha256},
            {"THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt", 1,
             kZeroByteSha256}};
        return payload;
    }

    void Install() {
        cyxwiz::runtime::BackendPackInstaller installer(root);
        const auto result = installer.InstallOrUpdate(Payload(), 1024);
        if (result.status !=
            cyxwiz::runtime::BackendPackInstallStatus::InstalledAndActivated) {
            throw std::runtime_error(result.message);
        }
    }

    void Active(std::vector<cyxwiz::runtime::ActivePackState> packs) {
        cyxwiz::runtime::ActiveRuntimeState state;
        state.runtime_set_id = "set-v1";
        state.generation = 1;
        state.base_pack_id = "base-v1";
        state.packs = std::move(packs);
        std::string error;
        if (!cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
                root / "active-runtime.json", state, error)) {
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

    cyxwiz::runtime::ActiveRuntimeState Rollback() const {
        cyxwiz::runtime::ActiveRuntimeState state;
        std::string error;
        if (!cyxwiz::runtime::LoadActiveRuntimeState(
                root / "rollback" / "set-v1" /
                    "previous-active-runtime.json",
                state, error)) {
            throw std::runtime_error(error);
        }
        return state;
    }

    static void Touch(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).put('\0');
    }

    std::filesystem::path root;
    std::filesystem::path source;
    static inline int sequence = 0;
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

bool HasPack(
    const cyxwiz::runtime::ActiveRuntimeState& state,
    const std::string& backend,
    const std::string& pack_id) {
    return std::any_of(
        state.packs.begin(), state.packs.end(),
        [&](const auto& pack) {
            return pack.backend == backend && pack.pack_id == pack_id;
        });
}

}  // namespace

int main() {
    int failures = 0;
    {
        Fixture fixture;
        fixture.Install();
        cyxwiz::runtime::BackendPackRemover remover(fixture.root);
        const auto result = remover.Remove("opencl", "opencl-v1");
        const auto active = fixture.Active();
        const auto rollback = fixture.Rollback();
        failures += !Expect(
            result.status ==
                    cyxwiz::runtime::BackendPackRemovalStatus::Removed &&
                active.generation == 3 && active.packs.empty() &&
                rollback.generation == 3 && rollback.packs.empty() &&
                !std::filesystem::exists(
                    fixture.root / "packs" / "opencl" / "opencl-v1"),
            "active removal must deactivate and invalidate rollback before deleting the pack");
        failures += !Expect(
            remover.GetProgress().stage ==
                cyxwiz::runtime::BackendPackRemovalStage::Complete,
            "removal must expose an immutable terminal progress snapshot");
    }
    {
        Fixture fixture;
        const auto target =
            fixture.root / "packs" / "opencl" / "opencl-v1";
        Fixture::Touch(target / "runtime" / "afopencl.dll");
        auto rollback = fixture.Active();
        rollback.packs.push_back({"opencl", "opencl-v1"});
        std::string error;
        const bool saved = cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
            fixture.root / "rollback" / "set-v1" /
                "previous-active-runtime.json",
            rollback, error);
        cyxwiz::runtime::BackendPackRemover remover(fixture.root);
        const auto result = remover.Remove("opencl", "opencl-v1");
        failures += !Expect(
            saved && result.status ==
                         cyxwiz::runtime::BackendPackRemovalStatus::Removed &&
                fixture.Active().generation == 1 &&
                fixture.Rollback().packs.empty() &&
                !std::filesystem::exists(target),
            "inactive removal must invalidate a rollback-protected reference before deletion");
    }
    for (const auto checkpoint : {
             cyxwiz::runtime::BackendPackRemovalCheckpoint::AfterDeactivation,
             cyxwiz::runtime::BackendPackRemovalCheckpoint::AfterRollbackUpdate,
             cyxwiz::runtime::BackendPackRemovalCheckpoint::AfterQuarantine}) {
        Fixture fixture;
        fixture.Install();
        cyxwiz::runtime::BackendPackRemover remover(
            fixture.root, [] { return false; }, {},
            [checkpoint](auto current) { return current != checkpoint; });
        const auto result = remover.Remove("opencl", "opencl-v1");
        const auto active = fixture.Active();
        failures += !Expect(
            result.status ==
                    cyxwiz::runtime::BackendPackRemovalStatus::Interrupted &&
                active.generation == 3 && active.packs.empty(),
            "removal interruption must retain a complete runtime without the removed route active");
        if (checkpoint ==
            cyxwiz::runtime::BackendPackRemovalCheckpoint::AfterQuarantine) {
            const auto quarantine = result.quarantined_directory;
            failures += !Expect(
                !std::filesystem::exists(
                    fixture.root / "packs" / "opencl" / "opencl-v1") &&
                    !quarantine.empty() &&
                    std::filesystem::is_directory(quarantine),
                "post-quarantine interruption must report recoverable cleanup state");
            cyxwiz::runtime::BackendPackRemover recovery(fixture.root);
            const auto recovered =
                recovery.Remove("opencl", "opencl-v1");
            failures += !Expect(
                recovered.status == cyxwiz::runtime::
                    BackendPackRemovalStatus::AlreadyAbsent &&
                    !std::filesystem::exists(quarantine),
                "a repeated removal must safely finish orphaned quarantine cleanup");
        } else {
            failures += !Expect(
                std::filesystem::is_directory(
                    fixture.root / "packs" / "opencl" / "opencl-v1"),
                "pre-quarantine interruption must retain the complete inactive pack");
        }
    }
    {
        Fixture fixture;
        fixture.Install();
        cyxwiz::runtime::RuntimeExecutionLease execution;
        cyxwiz::runtime::BackendPackRemover remover(fixture.root);
        const auto result = remover.Remove("opencl", "opencl-v1");
        failures += !Expect(
            result.status == cyxwiz::runtime::BackendPackRemovalStatus::
                                 ExecutionActive &&
                HasPack(fixture.Active(), "opencl", "opencl-v1") &&
                std::filesystem::is_directory(
                    fixture.root / "packs" / "opencl" / "opencl-v1"),
            "an execution lease must block removal before state or files change");
    }
    {
        Fixture fixture;
        fixture.Install();
        cyxwiz::runtime::BackendPackRemover* active_remover = nullptr;
        cyxwiz::runtime::BackendPackRemover remover(
            fixture.root, [] { return false; },
            [&](const auto& progress) {
                if (active_remover &&
                    progress.stage == cyxwiz::runtime::
                        BackendPackRemovalStage::Deactivating) {
                    active_remover->Cancel();
                }
            });
        active_remover = &remover;
        const auto result = remover.Remove("opencl", "opencl-v1");
        failures += !Expect(
            result.status ==
                    cyxwiz::runtime::BackendPackRemovalStatus::Interrupted &&
                fixture.Active().packs.empty() &&
                std::filesystem::is_directory(
                    fixture.root / "packs" / "opencl" / "opencl-v1"),
            "cancellation must stop before quarantine and leave a complete inactive pack");
    }
    {
        Fixture fixture;
        cyxwiz::runtime::BackendPackRemover remover(fixture.root);
        const auto absent = remover.Remove("opencl", "opencl-v1");
        const auto invalid = remover.Remove("cpu", "base-v1");
        failures += !Expect(
            absent.status == cyxwiz::runtime::BackendPackRemovalStatus::
                                 AlreadyAbsent &&
                invalid.status == cyxwiz::runtime::BackendPackRemovalStatus::
                                  InvalidRequest &&
                fixture.Active().generation == 1,
            "removal must be idempotent for absent optional packs and reject the CPU base");
    }

    if (failures == 0) {
        std::cout << "backend pack maintenance contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
