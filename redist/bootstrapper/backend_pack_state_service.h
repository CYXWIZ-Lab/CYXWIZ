#pragma once

#include "runtime_layout.h"

#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace cyxwiz::runtime {

enum class BackendPackStateStatus {
    Completed,
    Busy,
    ExecutionActive,
    InvalidRequest,
    InvalidRuntime,
    PublishFailed
};

enum class BackendPackStateStage {
    Idle,
    Reading,
    Validating,
    SavingRollback,
    Publishing,
    Complete,
    Failed
};

struct BackendPackStateProgress {
    BackendPackStateStage stage = BackendPackStateStage::Idle;
    std::string operation;
    std::string backend;
    std::string pack_id;
    std::uint64_t generation = 0;
    std::string message;
};

struct BackendPackStateResult {
    BackendPackStateResult() = default;

    BackendPackStateResult(
        BackendPackStateStatus status_value,
        std::string message_value,
        std::optional<ActiveRuntimeState> previous_value = std::nullopt,
        std::optional<ActiveRuntimeState> current_value = std::nullopt)
        : status(status_value),
          message(std::move(message_value)),
          previous(std::move(previous_value)),
          current(std::move(current_value)) {}

    BackendPackStateStatus status =
        BackendPackStateStatus::InvalidRequest;
    std::string message;
    std::optional<ActiveRuntimeState> previous;
    std::optional<ActiveRuntimeState> current;
};

using BackendPackExecutionActiveCheck = std::function<bool()>;
using BackendPackStateObserver =
    std::function<void(const BackendPackStateProgress&)>;

class BackendPackStateService {
public:
    explicit BackendPackStateService(
        std::filesystem::path runtime_root,
        BackendPackExecutionActiveCheck execution_active = {},
        BackendPackStateObserver observer = {});

    BackendPackStateResult InitializeBase(
        std::string runtime_set_id,
        std::string base_pack_id);
    BackendPackStateResult UpdateBase(
        std::string runtime_set_id,
        std::string base_pack_id);
    BackendPackStateResult ActivateOptionalPack(
        std::string backend,
        std::string pack_id);
    BackendPackStateResult DeactivateOptionalPack(
        std::string backend);
    BackendPackStateResult Rollback();

    BackendPackStateProgress GetProgress() const;

private:
    using CandidateBuilder = std::function<bool(
        const ActiveRuntimeState&,
        ActiveRuntimeState&,
        std::string&)>;

    BackendPackStateResult Mutate(
        std::string operation,
        std::string backend,
        std::string pack_id,
        const CandidateBuilder& build_candidate);
    void SetProgress(BackendPackStateProgress progress);
    std::filesystem::path RollbackPath(
        const ActiveRuntimeState& state) const;

    std::filesystem::path runtime_root_;
    BackendPackExecutionActiveCheck execution_active_;
    BackendPackStateObserver observer_;
    mutable std::mutex progress_mutex_;
    std::mutex mutation_mutex_;
    BackendPackStateProgress progress_;
};

const char* BackendPackStateStatusName(BackendPackStateStatus status);
const char* BackendPackStateStageName(BackendPackStateStage stage);

}  // namespace cyxwiz::runtime
