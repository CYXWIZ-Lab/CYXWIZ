#include "backend_pack_installer.h"
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
            ("cyxwiz-pack-installer-test-" + std::to_string(process_id) +
             "-" + std::to_string(++sequence));
        Touch(root / "base" / "base-v1" / "cyxwiz-engine.exe");
        source = root.parent_path() /
            (root.filename().string() + "-source");
        Touch(source / "runtime" / "afopencl.dll");
        Touch(source / "THIRD_PARTY_LICENSES" / "ArrayFire" /
              "LICENSE.txt");

        cyxwiz::runtime::ActiveRuntimeState state;
        state.runtime_set_id = "set-v1";
        state.generation = 1;
        state.base_pack_id = "base-v1";
        std::string error;
        if (!cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
                root / "active-runtime.json", state, error)) {
            throw std::runtime_error(error);
        }
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

    cyxwiz::runtime::ActiveRuntimeState Active() const {
        cyxwiz::runtime::ActiveRuntimeState state;
        std::string error;
        if (!cyxwiz::runtime::LoadActiveRuntimeState(
                root / "active-runtime.json", state, error)) {
            throw std::runtime_error(error);
        }
        return state;
    }

    static void Touch(const std::filesystem::path& path, char value = '\0') {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).put(value);
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
        std::vector<cyxwiz::runtime::BackendPackInstallProgress> progress;
        cyxwiz::runtime::BackendPackInstaller installer(
            fixture.root, [] { return false; },
            [&](const auto& value) { progress.push_back(value); });
        const auto result = installer.InstallOrUpdate(fixture.Payload(), 1024);
        const auto active = fixture.Active();
        failures += !Expect(
            result.status ==
                cyxwiz::runtime::BackendPackInstallStatus::InstalledAndActivated &&
                active.generation == 2 &&
                HasPack(active, "opencl", "opencl-v1"),
            "a verified pack must stage, publish, and activate atomically");
        failures += !Expect(
            std::filesystem::is_regular_file(
                fixture.root / "packs" / "opencl" / "opencl-v1" /
                "runtime" / "afopencl.dll"),
            "activation must reference a complete versioned pack directory");
        failures += !Expect(
            !progress.empty() &&
                progress.back().stage ==
                    cyxwiz::runtime::BackendPackInstallStage::Complete &&
                progress.back().completed_bytes == 2,
            "progress snapshots must expose bounded byte and terminal truth");

        const auto repeated =
            installer.InstallOrUpdate(fixture.Payload(), 1024);
        failures += !Expect(
            repeated.status == cyxwiz::runtime::BackendPackInstallStatus::
                                   AlreadyInstalledAndActivated &&
                repeated.activation.has_value() &&
                repeated.activation->status ==
                    cyxwiz::runtime::BackendPackStateStatus::Completed,
            "an already installed pack must be reverified before activation");
    }
    {
        Fixture fixture;
        Fixture::Touch(
            fixture.source / "runtime" / "afopencl.dll", 'x');
        const auto before = fixture.Active();
        cyxwiz::runtime::BackendPackInstaller installer(fixture.root);
        const auto result = installer.InstallOrUpdate(fixture.Payload(), 1024);
        failures += !Expect(
            result.status ==
                cyxwiz::runtime::BackendPackInstallStatus::IntegrityFailure &&
                fixture.Active().generation == before.generation &&
                installer.GetProgress().stage ==
                    cyxwiz::runtime::BackendPackInstallStage::Failed &&
                !std::filesystem::exists(
                    fixture.root / "packs" / "opencl" / "opencl-v1"),
            "component corruption must be rejected before staging or activation");
    }
    {
        Fixture fixture;
        cyxwiz::runtime::BackendPackInstaller installer(fixture.root);
        const auto result = installer.InstallOrUpdate(fixture.Payload(), 1);
        failures += !Expect(
            result.status == cyxwiz::runtime::BackendPackInstallStatus::
                                 DiskBudgetExceeded &&
                fixture.Active().generation == 1,
            "the approved disk budget must be enforced before copying");
    }
    {
        Fixture fixture;
        Fixture::Touch(fixture.source / "unexpected.dll");
        cyxwiz::runtime::BackendPackInstaller installer(fixture.root);
        const auto result = installer.InstallOrUpdate(fixture.Payload(), 1024);
        failures += !Expect(
            result.status ==
                cyxwiz::runtime::BackendPackInstallStatus::IntegrityFailure &&
                fixture.Active().generation == 1,
            "unexpected payload files must fail the exact signed inventory");
    }
    for (const auto checkpoint : {
             cyxwiz::runtime::BackendPackInstallCheckpoint::AfterValidation,
             cyxwiz::runtime::BackendPackInstallCheckpoint::AfterCopy,
             cyxwiz::runtime::BackendPackInstallCheckpoint::BeforePackPublish,
             cyxwiz::runtime::BackendPackInstallCheckpoint::AfterPackPublish,
             cyxwiz::runtime::BackendPackInstallCheckpoint::BeforeActivation}) {
        Fixture fixture;
        cyxwiz::runtime::BackendPackInstaller installer(
            fixture.root, [] { return false; }, {},
            [checkpoint](auto current) { return current != checkpoint; });
        const auto result = installer.InstallOrUpdate(fixture.Payload(), 1024);
        const auto active = fixture.Active();
        failures += !Expect(
            (result.status ==
                 cyxwiz::runtime::BackendPackInstallStatus::Interrupted ||
             result.status == cyxwiz::runtime::BackendPackInstallStatus::
                                  InstalledUnqualified) &&
                active.generation == 1 && active.packs.empty(),
            "interruption at every pre-activation stage must preserve the old complete runtime");
        if (checkpoint ==
                cyxwiz::runtime::BackendPackInstallCheckpoint::AfterPackPublish ||
            checkpoint ==
                cyxwiz::runtime::BackendPackInstallCheckpoint::BeforeActivation) {
            failures += !Expect(
                std::filesystem::is_regular_file(
                    fixture.root / "packs" / "opencl" / "opencl-v1" /
                    "runtime" / "afopencl.dll"),
                "post-publication interruption may leave only a complete inactive pack");
        }
    }
    {
        Fixture fixture;
        cyxwiz::runtime::RuntimeExecutionLease execution;
        cyxwiz::runtime::BackendPackInstaller installer(fixture.root);
        const auto result = installer.InstallOrUpdate(fixture.Payload(), 1024);
        failures += !Expect(
            result.status ==
                cyxwiz::runtime::BackendPackInstallStatus::ExecutionActive &&
                fixture.Active().generation == 1,
            "an active execution context must block pack publication and activation");
    }

    if (failures == 0) {
        std::cout << "backend pack installer contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
