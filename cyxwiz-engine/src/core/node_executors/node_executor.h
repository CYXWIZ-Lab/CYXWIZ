#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <arrow/table.h>

namespace cyxwiz {

/**
 * Execution state for node executors
 */
enum class ExecutorState {
    Idle,           // Not started
    Configuring,    // Being configured
    Executing,      // Running
    Completed,      // Finished successfully
    Error,          // Failed
    Cancelled       // User cancelled
};

/**
 * Code generation framework targets
 */
enum class CodeFramework {
    Sklearn,        // scikit-learn
    PyCyxWiz,       // pycyxwiz native
    PyTorch,        // PyTorch
    TensorFlow      // TensorFlow
};

/**
 * INodeExecutor - Base interface for all node executors
 *
 * Each analytics node type (KMeans, PCA, etc.) has a corresponding executor
 * that handles configuration, execution, visualization, and code generation.
 */
class INodeExecutor {
public:
    virtual ~INodeExecutor() = default;

    // === Identity ===
    virtual std::string GetName() const = 0;
    virtual std::string GetCategory() const = 0;

    // === Execution ===
    virtual void Execute() = 0;
    virtual void Cancel() { cancel_requested_ = true; }
    virtual ExecutorState GetState() const { return state_; }
    virtual float GetProgress() const { return progress_; }
    virtual std::string GetStatusMessage() const { return status_message_; }
    virtual std::string GetErrorMessage() const { return error_message_; }

    // === Input Data ===
    virtual void SetInputData(const std::vector<std::vector<double>>& data) {
        input_data_ = data;
    }
    virtual void SetColumnNames(const std::vector<std::string>& names) {
        column_names_ = names;
    }

    // === UI Rendering ===
    // Renders configuration controls in Properties Panel
    virtual void RenderConfigUI() = 0;
    // Renders results/visualization (scatter plots, tables, etc.)
    virtual void RenderResultsUI() = 0;
    // Whether this executor has visualization output
    virtual bool HasVisualization() const { return true; }

    // === Code Generation ===
    virtual std::string GenerateCode(CodeFramework framework) const = 0;

    // === Callbacks ===
    using ProgressCallback = std::function<void(float progress, const std::string& message)>;
    void SetProgressCallback(ProgressCallback cb) { progress_callback_ = cb; }

protected:
    // State
    ExecutorState state_ = ExecutorState::Idle;
    float progress_ = 0.0f;
    std::string status_message_;
    std::string error_message_;
    std::atomic<bool> cancel_requested_{false};

    // Input data
    std::vector<std::vector<double>> input_data_;
    std::vector<std::string> column_names_;

    // Callbacks
    ProgressCallback progress_callback_;

    // Helper to report progress
    void ReportProgress(float progress, const std::string& message) {
        progress_ = progress;
        status_message_ = message;
        if (progress_callback_) {
            progress_callback_(progress, message);
        }
    }
};

} // namespace cyxwiz
