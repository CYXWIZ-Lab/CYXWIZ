#include "backend_pack_state_service.h"
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
        Touch(root / "base" / "base-v1" / "cyxwiz-engine.exe");
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
              ("af" + backend + ".dll"));
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
        const std::string before = fixture.ActiveText();
        const auto rejected =
            service.ActivateOptionalPack("opencl", "missing-v1");
        failures += !Expect(
            rejected.status ==
                cyxwiz::runtime::BackendPackStateStatus::InvalidRuntime &&
                fixture.ActiveText() == before,
            "a missing pack closure must leave active state byte-for-byte unchanged");

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

    if (failures == 0) {
        std::cout << "backend pack state service contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
