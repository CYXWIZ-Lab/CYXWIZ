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

bool ScriptingEngine::ExecuteScriptAsync(const std::string&, RunCallbacks callbacks) {
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        async_result_ = DisabledResult();
    }
    if (callbacks.on_complete) callbacks.on_complete(DisabledResult());
    return true;
}

bool ScriptingEngine::DropNotebookNamespace(const std::string&) { return true; }

std::string ScriptingEngine::CallLanguageTool(const std::string&, const std::string&, const std::string&) { return {}; }

LanguageService& ScriptingEngine::Language() {
    std::lock_guard<std::mutex> lock(language_mutex_);
    if (!language_) language_ = std::make_unique<LanguageService>(this);
    return *language_;
}

std::string ScriptingEngine::LanguageToolsError() const { return "This build has no Python scripting"; }

VariablesService& ScriptingEngine::Variables() {
    std::lock_guard<std::mutex> lock(language_mutex_);
    if (!variables_) variables_ = std::make_unique<VariablesService>(this);
    return *variables_;
}

bool ScriptingEngine::RunActive() const { return false; }

ScriptingEngine::DebugSnapshot ScriptingEngine::GetDebugSnapshot() const { return {}; }
bool ScriptingEngine::DebugCommand(const std::string&) { return false; }
bool ScriptingEngine::DebugPause() { return false; }
void ScriptingEngine::DebugSetBreakpoints(const std::vector<DebugBreakpoint>&) {}
std::string ScriptingEngine::CallDebugTool(const std::string&, const std::string&, bool* busy) {
    if (busy) *busy = false;
    return {};
}

std::string ScriptingEngine::CallVariablesTool(const std::string&, const std::string&, const std::string&, bool* busy) {
    if (busy) *busy = false;
    return {};
}

bool ScriptingEngine::ReadVariableTable(const std::string&, const std::string&, long long, const std::vector<int>&,
                                        VariableTable* out, bool* busy) {
    if (busy) *busy = false;
    if (out) out->error = "This build has no Python scripting";
    return false;
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

void ScriptingEngine::EnableSandbox(bool enable) { sandbox_enabled_ = enable; }

void ScriptingEngine::SetSandboxConfig(const PythonSandbox::Config& config) {
    sandbox_->SetConfig(config);
}

PythonSandbox::Config ScriptingEngine::GetSandboxConfig() const {
    return sandbox_->GetConfig();
}

bool ScriptingEngine::IsInitialized() const { return false; }
bool ScriptingEngine::EnsurePythonInitialized(std::string* error_out) {
    if (error_out) *error_out = "This build has no Python scripting";
    return false;
}

std::string ScriptingEngine::GetPythonRuntimeDiagnostics() {
    return "Python scripting is disabled in this Engine build";
}

bool ScriptingEngine::ReloadPythonForProject() { return false; }

ScriptingEngine::InterpreterInfo ScriptingEngine::GetInterpreterInfo() { return {}; }

std::vector<std::string> ScriptingEngine::CompleteSync(const std::string&, size_t) {
    return {};
}

bool ScriptingEngine::ResetSession(std::string* error_out) {
    if (error_out) *error_out = "Python scripting is disabled in this Engine build.";
    return false;
}

std::string ScriptingEngine::GetLastInitError() const {
    return "Python scripting is disabled in this Engine build.";
}

bool ScriptingEngine::IsSafeForNewCommand() const { return true; }

}  // namespace scripting
