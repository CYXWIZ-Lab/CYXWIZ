#pragma once

// Why a headless training job did not complete (TOFIX118 P4a). One category
// per failure, carried from the shared runner to the node and on to the
// Engine as TrainingError.error_code, so a user can tell "fix the graph"
// from "the data is wrong" from "this device cannot run it".

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace cyxwiz {

enum class TrainingFailureKind {
    None,
    Refused,      // the graph cannot train on this host (does not parse or compile, unsupported step)
    DataError,    // an input is missing, unreadable, or does not match the graph
    DeviceError,  // the compute device or route failed or is not verified
    OutOfMemory,  // the device or host ran out of memory
    Cancelled,    // stopped on request
    Internal,     // anything else: a CyxWiz fault to report
};

// Wire code (TrainingError.error_code).
inline const char* TrainingFailureCode(TrainingFailureKind kind) {
    switch (kind) {
        case TrainingFailureKind::None: return "";
        case TrainingFailureKind::Refused: return "REFUSED";
        case TrainingFailureKind::DataError: return "DATA_ERROR";
        case TrainingFailureKind::DeviceError: return "DEVICE_ERROR";
        case TrainingFailureKind::OutOfMemory: return "OUT_OF_MEMORY";
        case TrainingFailureKind::Cancelled: return "CANCELLED";
        case TrainingFailureKind::Internal: return "INTERNAL";
    }
    return "INTERNAL";
}

// What the Engine shows for a wire code; unknown codes (older nodes send
// TRAINING_FAILED) read as a plain training failure.
inline const char* TrainingFailureLabel(std::string_view code) {
    if (code == "REFUSED") return "Refused before start";
    if (code == "DATA_ERROR") return "Data problem";
    if (code == "DEVICE_ERROR") return "Device problem";
    if (code == "OUT_OF_MEMORY") return "Out of memory";
    if (code == "CANCELLED") return "Cancelled";
    if (code == "INTERNAL") return "Internal error";
    return "Training failed";
}

// Category of a failure the training executor reported (terminal_reason or
// an exception message) once training had started.
inline TrainingFailureKind ClassifyTrainingFailure(const std::string& reason) {
    std::string text = reason;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto has = [&text](std::string_view part) { return text.find(part) != std::string::npos; };
    if (has("out of memory") || has("af_err_no_mem") || has("bad_alloc") || has("memory allocation failed") ||
        has("cl_mem_object_allocation_failure") || has("cuda_error_out_of_memory")) {
        return TrainingFailureKind::OutOfMemory;
    }
    if (has("device_preflight_failed") || has("device lost") || has("device_lost") ||
        has("qualification evidence") || has("cl_device_not_available")) {
        return TrainingFailureKind::DeviceError;
    }
    if (has("user_cancelled") || has("cancelled")) return TrainingFailureKind::Cancelled;
    return TrainingFailureKind::Internal;
}

}  // namespace cyxwiz
