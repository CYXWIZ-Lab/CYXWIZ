#include "scripting_engine.h"

namespace scripting {

std::atomic<int> ScriptingEngine::shared_cancel_flag_{0};

namespace {
ExecutionResult DisabledResult() {
    ExecutionResult result{};
    result.success = false;
    result.error_message =
        "Python scripting is unavailable in this Engine build";
    return result;
}
}  // namespace

ScriptingEngine::ScriptingEngine()
    : python_engine_(std::make_unique<PythonEngine>()),
      sandbox_(std::make_unique<PythonSandbox>()),
      sandbox_enabled_(false) {}

ScriptingEngine::~ScriptingEngine() {
    if (command_thread_ && command_thread_->joinable()) command_thread_->join();
    if (script_thread_ && script_thread_->joinable()) script_thread_->join();
}

ExecutionResult ScriptingEngine::ExecuteCommand(const std::string&) {
    return DisabledResult();
}

void ScriptingEngine::ExecuteCommandAsync(const std::string&) {
    std::lock_guard<std::mutex> lock(command_result_mutex_);
    async_command_result_ = DisabledResult();
}

void ScriptingEngine::StopCommand() { command_running_ = false; }

std::optional<ExecutionResult> ScriptingEngine::GetCommandResult() {
    std::lock_guard<std::mutex> lock(command_result_mutex_);
    auto result = std::move(async_command_result_);
    async_command_result_.reset();
    return result;
}

ExecutionResult ScriptingEngine::ExecuteScript(const std::string&) {
    return DisabledResult();
}

ExecutionResult ScriptingEngine::ExecuteFile(const std::string&) {
    return DisabledResult();
}

void ScriptingEngine::ExecuteScriptAsync(const std::string&) {
    std::lock_guard<std::mutex> lock(result_mutex_);
    async_result_ = DisabledResult();
}

void ScriptingEngine::StopScript() { script_running_ = false; }
bool ScriptingEngine::IsScriptRunning() const { return false; }

std::optional<ExecutionResult> ScriptingEngine::GetAsyncResult() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    auto result = std::move(async_result_);
    async_result_.reset();
    return result;
}

std::string ScriptingEngine::GetPendingOutput() { return {}; }

void ScriptingEngine::SetCompletionCallback(CompletionCallback callback) {
    completion_callback_ = std::move(callback);
}

void ScriptingEngine::SetOutputCallback(OutputCallback callback) {
    output_callback_ = std::move(callback);
}

void ScriptingEngine::EnableSandbox(bool enable) { sandbox_enabled_ = enable; }

void ScriptingEngine::SetSandboxConfig(const PythonSandbox::Config& config) {
    sandbox_->SetConfig(config);
}

PythonSandbox::Config ScriptingEngine::GetSandboxConfig() const {
    return sandbox_->GetConfig();
}

bool ScriptingEngine::IsInitialized() const { return false; }

std::string ScriptingEngine::GetPythonRuntimeDiagnostics() {
    return "Python scripting is disabled in this Engine build";
}

bool ScriptingEngine::ReloadPythonForProject() { return false; }

void ScriptingEngine::RegisterTrainingDashboard(
    cyxwiz::TrainingPlotPanel* panel) {
    training_plot_panel_ = panel;
}

void ScriptingEngine::EnsureTrainingDashboardRegistered() {}
bool ScriptingEngine::IsSafeForNewCommand() const { return true; }

}  // namespace scripting
