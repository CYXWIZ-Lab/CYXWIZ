#pragma once

#include <string>

namespace cyxwiz::errors {

namespace Compiler {
inline constexpr const char* GenericIssue = "CW-C-0001";
inline constexpr const char* MissingTrainingPathNode = "CW-C-0101";
inline constexpr const char* UnsupportedTrainingNode = "CW-C-0102";
inline constexpr const char* InvalidConnectivity = "CW-C-0103";
inline constexpr const char* TensorShapeMismatch = "CW-C-0301";
inline constexpr const char* LabelOutputShapeMismatch = "CW-C-0302";
inline constexpr const char* InvalidParameter = "CW-C-0401";
inline constexpr const char* InvariantViolation = "CW-C-0901";
} // namespace Compiler

namespace Runtime {
inline constexpr const char* PipelineMalformed = "CW-R-0101";
inline constexpr const char* InvalidState = "CW-R-0102";
inline constexpr const char* UnsupportedNode = "CW-R-0201";
inline constexpr const char* OperatorMissing = "CW-R-0202";
inline constexpr const char* InputDatasetMissing = "CW-R-0301";
inline constexpr const char* InvalidParameter = "CW-R-0401";
inline constexpr const char* ExecutionFailed = "CW-R-0501";
inline constexpr const char* InvariantViolation = "CW-R-0901";
} // namespace Runtime

namespace Gpu {
inline constexpr const char* BackendUnavailable = "CW-G-0101";
inline constexpr const char* PathDisabledByPolicy = "CW-G-0201";
inline constexpr const char* UnsupportedShape = "CW-G-0301";
inline constexpr const char* KernelExecutionFailed = "CW-G-0501";
inline constexpr const char* DependencyFailure = "CW-G-0601";
inline constexpr const char* MemoryExhausted = "CW-G-0701";
} // namespace Gpu

namespace Cpu {
inline constexpr const char* BackendUnavailable = "CW-P-0101";
inline constexpr const char* UnsupportedShape = "CW-P-0301";
inline constexpr const char* OperationFailed = "CW-P-0501";
inline constexpr const char* MemoryExhausted = "CW-P-0701";
} // namespace Cpu

namespace Data {
inline constexpr const char* RequiredColumnMissing = "CW-D-0101";
inline constexpr const char* RequiredLabelColumnMissing = "CW-D-0102";
inline constexpr const char* ColumnTypeMismatch = "CW-D-0301";
inline constexpr const char* RowCountMismatch = "CW-D-0302";
inline constexpr const char* ClassLabelMismatch = "CW-D-0303";
inline constexpr const char* VocabularyCoverageWarning = "CW-D-0304";
inline constexpr const char* InvalidSplit = "CW-D-0401";
inline constexpr const char* MaterializationFailed = "CW-D-0501";
} // namespace Data

namespace File {
inline constexpr const char* PathMissing = "CW-F-0101";
inline constexpr const char* NotFound = "CW-F-0102";
inline constexpr const char* UnsupportedFormat = "CW-F-0201";
inline constexpr const char* InvalidOption = "CW-F-0401";
inline constexpr const char* ReadFailed = "CW-F-0501";
inline constexpr const char* WriteFailed = "CW-F-0502";
inline constexpr const char* PermissionDenied = "CW-F-0601";
} // namespace File

namespace Memory {
inline constexpr const char* ResourceCheckFailed = "CW-M-0101";
inline constexpr const char* HostExhausted = "CW-M-0701";
inline constexpr const char* GpuExhausted = "CW-M-0702";
inline constexpr const char* BatchTooLarge = "CW-M-0703";
} // namespace Memory

namespace Ui {
inline constexpr const char* InvalidWorkflowState = "CW-U-0101";
inline constexpr const char* MissingSelection = "CW-U-0102";
inline constexpr const char* RuntimeOnlyNode = "CW-U-0201";
inline constexpr const char* InvalidParameter = "CW-U-0401";
inline constexpr const char* StateInvariantViolation = "CW-U-0901";
} // namespace Ui

namespace Serialization {
inline constexpr const char* ArtifactPathMissing = "CW-S-0101";
inline constexpr const char* ExportFormatUnavailable = "CW-S-0201";
inline constexpr const char* ExportFormatNotCompiled = "CW-S-0202";
inline constexpr const char* ModelSaveFailed = "CW-S-0501";
inline constexpr const char* ModelLoadFailed = "CW-S-0502";
inline constexpr const char* CheckpointSerializationFailed = "CW-S-0801";
} // namespace Serialization

namespace Training {
inline constexpr const char* InvalidTrainingSetup = "CW-T-0101";
inline constexpr const char* ModelBuildFailed = "CW-T-0102";
inline constexpr const char* LossSetupFailed = "CW-T-0401";
inline constexpr const char* OptimizerSetupFailed = "CW-T-0402";
inline constexpr const char* TrainingExecutionFailed = "CW-T-0501";
} // namespace Training

namespace External {
inline constexpr const char* OptionalIntegrationUnavailable = "CW-X-0201";
inline constexpr const char* ThirdPartyCallFailed = "CW-X-0501";
inline constexpr const char* PluginDependencyFailure = "CW-X-0601";
inline constexpr const char* PythonBridgeFailure = "CW-X-0602";
inline constexpr const char* RemoteServiceUnavailable = "CW-X-0701";
} // namespace External

inline bool HasCodePrefix(const std::string& message) {
    return message.size() >= 12 &&
           message[0] == '[' &&
           message[1] == 'C' &&
           message[2] == 'W' &&
           message[3] == '-' &&
           message[5] == '-' &&
           message[10] == ']';
}

inline std::string FormatMessage(const char* code,
                                 const std::string& message,
                                 const std::string& detail = {},
                                 const std::string& hint = {}) {
    if (HasCodePrefix(message)) {
        return message;
    }

    std::string formatted = "[";
    formatted += code;
    formatted += "] ";
    formatted += message;
    if (!detail.empty()) {
        formatted += ". Detail: ";
        formatted += detail;
    }
    if (!hint.empty()) {
        formatted += ". Hint: ";
        formatted += hint;
    }
    return formatted;
}

inline std::string FormatError(const char* code,
                               const std::string& message,
                               const std::string& detail = {},
                               const std::string& hint = {}) {
    return FormatMessage(code, message, detail, hint);
}

inline std::string FormatWarning(const char* code,
                                 const std::string& message,
                                 const std::string& detail = {},
                                 const std::string& hint = {}) {
    return FormatMessage(code, message, detail, hint);
}

} // namespace cyxwiz::errors
