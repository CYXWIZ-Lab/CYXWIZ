#pragma once

#include "checkpoint_manifest.h"

#include <cyxwiz/optimizer.h>
#include <cyxwiz/optimizers/lr_warmup.h>
#include <cyxwiz/scheduler.h>
#include <cyxwiz/sequential.h>

#include <filesystem>
#include <string>

namespace cyxwiz {

bool VerifyCheckpointPayloadFile(
    const std::filesystem::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    std::string& error);

bool SaveModelPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const std::string& relative_path,
    const SequentialModel& model,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error);

bool LoadModelPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    SequentialModel& model,
    std::string& error);

bool SaveOptimizerPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const std::string& relative_path,
    const Optimizer& optimizer,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error);

bool LoadOptimizerPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    Optimizer& optimizer,
    std::string& error);

bool SaveSchedulerPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const std::string& relative_path,
    const LRScheduler& scheduler,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error);

bool LoadSchedulerPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    LRScheduler& scheduler,
    std::string& error);

bool SaveLRWarmupPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const std::string& relative_path,
    const LRWarmup& warmup,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error);

bool LoadLRWarmupPayloadV2(
    const std::filesystem::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    LRWarmup& warmup,
    std::string& error);

} // namespace cyxwiz
