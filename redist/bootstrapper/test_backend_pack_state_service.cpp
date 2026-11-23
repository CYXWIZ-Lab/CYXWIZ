#include "backend_pack_state_service.h"
#include "backend_pack_platform.h"
#include "runtime_mutation_gate.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

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
            ("cyxwiz-pack-state-test-" + std::to_string(process_id) + "-" +
             std::to_string(++sequence));
        Touch(
            root / "base" / "base-v1" /
            cyxwiz::runtime::CurrentEngineExecutableName());
        Touch(
            root / "base" / "base-v2" /
            cyxwiz::runtime::CurrentEngineExecutableName());
        AddPack("opencl", "opencl-v1");
        AddPack("opencl", "opencl-v2");
        AddPack("cuda", "cuda-v1");

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
    }

    void AddPack(const std::string& backend, const std::string& pack_id) {
        Touch(root / "packs" / backend / pack_id / "runtime" /
              cyxwiz::runtime::CurrentArrayFireBackendPluginName(backend));
    }

    static void Touch(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).put('\0');
    }

    std::string ActiveText() const {
        std::ifstream stream(root / "active-runtime.json", std::ios::binary);
        return std::string(
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>());
    }

    std::filesystem::path root;
    static inline int sequence = 0;
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

const cyxwiz::runtime::ActivePackState* FindPack(
    const cyxwiz::runtime::ActiveRuntimeState& state,
    const std::string& backend) {
    const auto pack = std::find_if(
        state.packs.begin(), state.packs.end(),
        [&](const auto& candidate) {
            return candidate.backend == backend;
        });
    return pack == state.packs.end() ? nullptr : &*pack;
}

}  // namespace

int main() {
    int failures = 0;
    {
        Fixture fixture;
        std::filesystem::remove(fixture.root / "active-runtime.json");
        cyxwiz::runtime::BackendPackStateService service(fixture.root);
        const auto initialized = service.InitializeBase("set-v1", "base-v1");
        const auto repeated = service.InitializeBase("set-v1", "base-v1");
        failures += !Expect(
            initialized.status ==
                    cyxwiz::runtime::BackendPackStateStatus::Completed &&
                initialized.current.has_value() &&
                initialized.current->generation == 1 &&
                initialized.current->base_pack_id == "base-v1" &&
                repeated.status ==
                    cyxwiz::runtime::BackendPackStateStatus::InvalidRuntime,
            "fresh initialization must publish generation 1 exactly once");
    }
    {
        Fixture fixture;
        std::vector<cyxwiz::runtime::BackendPackStateStage> stages;
        cyxwiz::runtime::BackendPackStateService service(
            fixture.root, [] { return false; },
            [&](const auto& progress) { stages.push_back(progress.stage); });

        const auto activated =
            service.ActivateOptionalPack("opencl", "opencl-v1");
        if (activated.status !=
            cyxwiz::runtime::BackendPackStateStatus::Completed) {
            std::cerr << "activation status="
                      << cyxwiz::runtime::BackendPackStateStatusName(
                             activated.status)
                      << " message=" << activated.message << '\n';
        }
        failures += !Expect(
            activated.status ==
                cyxwiz::runtime::BackendPackStateStatus::Completed &&
                activated.current.has_value() &&
                activated.current->generation == 2 &&
                FindPack(*activated.current, "opencl") != nullptr &&
                FindPack(*activated.current, "opencl")->pack_id ==
                    "opencl-v1",
            "a complete installed pack must activate at the next generation");
        failures += !Expect(
            std::filesystem::is_regular_file(
                fixture.root / "rollback" / "set-v1" /
                "previous-active-runtime.json"),
            "activation must retain the previous complete state");
        failures += !Expect(
            !stages.empty() &&
                stages.back() ==
                    cyxwiz::runtime::BackendPackStateStage::Complete,
            "consumers must receive an immutable terminal progress snapshot");

        const auto updated =
            service.ActivateOptionalPack("opencl", "opencl-v2");
        if (updated.status !=
            cyxwiz::runtime::BackendPackStateStatus::Completed) {
            std::cerr << "update status="
                      << cyxwiz::runtime::BackendPackStateStatusName(
                             updated.status)
                      << " message=" << updated.message << '\n';
        }
        failures += !Expect(
            updated.status ==
                cyxwiz::runtime::BackendPackStateStatus::Completed &&
                FindPack(*updated.current, "opencl")->pack_id == "opencl-v2",
            "an update must switch only the selected backend pack");

        const auto rolled_back = service.Rollback();
        if (rolled_back.status !=
            cyxwiz::runtime::BackendPackStateStatus::Completed) {
            std::cerr << "rollback status="
                      << cyxwiz::runtime::BackendPackStateStatusName(
                             rolled_back.status)
                      << " message=" << rolled_back.message << '\n';
        }
        failures += !Expect(
            rolled_back.status ==
                cyxwiz::runtime::BackendPackStateStatus::Completed &&
                rolled_back.current->generation == 4 &&
                FindPack(*rolled_back.current, "opencl")->pack_id ==
                    "opencl-v1",
            "rollback must restore the previous complete state with a monotonic generation");

        const auto deactivated = service.DeactivateOptionalPack("opencl");
        failures += !Expect(
            deactivated.status ==
                cyxwiz::runtime::BackendPackStateStatus::Completed &&
                FindPack(*deactivated.current, "opencl") == nullptr,
            "deactivation must publish a complete CPU-only runtime");
        const auto restored = service.Rollback();
        failures += !Expect(
            restored.status ==
                cyxwiz::runtime::BackendPackStateStatus::Completed &&
                FindPack(*restored.current, "opencl") != nullptr,
            "rollback must restore a deactivated pack while its versioned files remain");
    }
    {
        Fixture fixture;
        cyxwiz::runtime::BackendPackStateService service(fixture.root);
        const auto optional =
            service.ActivateOptionalPack("opencl", "opencl-v1");
        const auto updated = service.UpdateBase("set-v2", "base-v2");
        failures += !Expect(
            optional.status ==
                    cyxwiz::runtime::BackendPackStateStatus::Completed &&
                updated.status ==
                    cyxwiz::runtime::BackendPackStateStatus::Completed &&
                updated.current.has_value() &&
                updated.current->runtime_set_id == "set-v2" &&
                updated.current->base_pack_id == "base-v2" &&
                updated.current->generation == 3 &&
                updated.current->packs.empty() &&
                std::filesystem::is_regular_file(
                    fixture.root / "rollback" / "set-v2" /
                    "previous-active-runtime.json"),
            "a base update must activate the new runtime set CPU-only and retain the old closure");

        const auto rolled_back = service.Rollback();
        failures += !Expect(
            rolled_back.status ==
                    cyxwiz::runtime::BackendPackStateStatus::Completed &&
                rolled_back.current.has_value() &&
                rolled_back.current->runtime_set_id == "set-v1" &&
                rolled_back.current->base_pack_id == "base-v1" &&
                rolled_back.current->generation == 4 &&
                FindPack(*rolled_back.current, "opencl") != nullptr &&
                FindPack(*rolled_back.current, "opencl")->pack_id ==
                    "opencl-v1",
            "rollback must restore the complete pre-update runtime across runtime-set identities");
    }
    {
        Fixture fixture;
        cyxwiz::runtime::BackendPackStateService service(fixture.root);
        const std::string before = fixture.ActiveText();
        const auto rejected =
            service.ActivateOptionalPack("opencl", "missing-v1");
        failures += !Expect(
            rejected.status ==
                cyxwiz::runtime::BackendPackStateStatus::InvalidRuntime &&
                fixture.ActiveText() == before,
            "a missing pack closure must leave active state byte-for-byte unchanged");

        const auto missing_base =
            service.UpdateBase("set-v2", "missing-base-v2");
        failures += !Expect(
            missing_base.status ==
                    cyxwiz::runtime::BackendPackStateStatus::InvalidRuntime &&
                fixture.ActiveText() == before,
            "a missing base update must leave active state byte-for-byte unchanged");

        cyxwiz::runtime::ActiveRuntimeState invalid;
        invalid.runtime_set_id = "set-v1";
        invalid.base_pack_id = "base-v1";
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
                fixture.root / "active-runtime.json", invalid, error) &&
                fixture.ActiveText() == before,
            "invalid state must be rejected before the atomic publication boundary");
    }
    {
        Fixture fixture;
        cyxwiz::runtime::RuntimeExecutionLease execution;
        cyxwiz::runtime::BackendPackStateService service(fixture.root);
        const std::string before = fixture.ActiveText();
        const auto blocked =
            service.ActivateOptionalPack("cuda", "cuda-v1");
        failures += !Expect(
            blocked.status ==
                cyxwiz::runtime::BackendPackStateStatus::ExecutionActive &&
                fixture.ActiveText() == before,
            "an active execution context must block runtime mutation");
    }
    {
        auto mutation =
            std::make_unique<cyxwiz::runtime::RuntimeMutationLease>();
        failures += !Expect(
            mutation->OwnsMutation(),
            "the transaction owner must acquire the exclusive runtime lease");
        std::promise<void> execution_acquired;
        auto acquired = execution_acquired.get_future();
        auto execution = std::async(
            std::launch::async,
            [&execution_acquired] {
                cyxwiz::runtime::RuntimeExecutionLease lease;
                execution_acquired.set_value();
            });
        failures += !Expect(
            acquired.wait_for(std::chrono::milliseconds(25)) ==
                std::future_status::timeout,
            "a run starting during runtime publication must wait");
        mutation.reset();
        failures += !Expect(
            acquired.wait_for(std::chrono::seconds(2)) ==
                std::future_status::ready,
            "the waiting run must proceed when atomic publication releases its lease");
        execution.get();
    }

    {
        // Plugin packs live in active-plugins.json only.
        Fixture fixture;
        namespace rt = cyxwiz::runtime;
        const auto plugins_path = fixture.root / rt::kActivePluginsFileName;
        rt::ActivePluginsState plugins;
        std::string error;
        failures += !Expect(rt::LoadActivePluginsState(plugins_path, plugins, error) &&
                                plugins.plugins.empty(),
                            "a missing active-plugins.json must read as no active plugins");
        const auto before = fixture.ActiveText();
        rt::BackendPackStateService service(fixture.root);
        failures += !Expect(service.ActivatePluginPack("com.example.sim", "sim-v1").status ==
                                rt::BackendPackStateStatus::InvalidRuntime,
                            "a plugin pack that is not installed must not activate");
        Fixture::Touch(rt::PluginPackDirectory(fixture.root, "com.example.sim", "sim-v1") /
                       "plugin.json");
        Fixture::Touch(rt::PluginPackDirectory(fixture.root, "com.example.sim", "sim-v2") /
                       "plugin.json");
        const auto first = service.ActivatePluginPack("com.example.sim", "sim-v1");
        const auto second = service.ActivatePluginPack("com.example.sim", "sim-v2");
        failures += !Expect(
            first.status == rt::BackendPackStateStatus::Completed &&
                second.status == rt::BackendPackStateStatus::Completed &&
                rt::LoadActivePluginsState(plugins_path, plugins, error) &&
                plugins.plugins.size() == 1 && plugins.plugins.front().pack_id == "sim-v2" &&
                fixture.ActiveText() == before,
            "one pack per plugin is active and active-runtime.json never changes");
        failures += !Expect(
            service.DeactivatePluginPack("com.example.sim").status ==
                    rt::BackendPackStateStatus::Completed &&
                service.DeactivatePluginPack("com.example.sim").status ==
                    rt::BackendPackStateStatus::InvalidRequest &&
                rt::LoadActivePluginsState(plugins_path, plugins, error) &&
                plugins.plugins.empty() && fixture.ActiveText() == before,
            "deactivation removes the plugin and rejects an inactive one");

        // Entries recorded for an earlier base are dropped on the next change.
        plugins = {"base-old", {{"com.example.other", "other-v1"}}};
        failures += !Expect(rt::SaveActivePluginsStateAtomic(plugins_path, plugins, error),
                            error);
        failures += !Expect(
            service.ActivatePluginPack("com.example.sim", "sim-v1").status ==
                    rt::BackendPackStateStatus::Completed &&
                rt::LoadActivePluginsState(plugins_path, plugins, error) &&
                plugins.base_pack_id == "base-v1" && plugins.plugins.size() == 1 &&
                plugins.plugins.front().plugin_id == "com.example.sim",
            "plugins of another base must not survive into the active base");

        std::ofstream(plugins_path, std::ios::binary | std::ios::trunc)
            << R"({"schema_version":1,"base_pack_id":"base-v1","plugins":[)"
            << R"({"plugin_id":"a","pack_id":"a1"},{"plugin_id":"a","pack_id":"a2"}]})";
        failures += !Expect(!rt::LoadActivePluginsState(plugins_path, plugins, error),
                            "duplicate active plugins must fail closed");
        std::ofstream(plugins_path, std::ios::binary | std::ios::trunc)
            << R"({"schema_version":1,"base_pack_id":"base-v1","plugins":[],"generation":2})";
        failures += !Expect(!rt::LoadActivePluginsState(plugins_path, plugins, error),
                            "unknown active-plugins fields must fail closed");
    }

    if (failures == 0) {
        std::cout << "backend pack state service contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
