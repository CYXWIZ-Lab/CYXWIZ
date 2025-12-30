#include "debugger.h"
#include "scripting_engine.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace scripting {

DebuggerManager::DebuggerManager() = default;

DebuggerManager::~DebuggerManager() {
    Shutdown();
}

bool DebuggerManager::Initialize(ScriptingEngine* engine) {
    if (!engine) {
        spdlog::error("DebuggerManager: Cannot initialize with null engine");
        return false;
    }

    scripting_engine_ = engine;

    // Check if debugpy is available
    std::string check_debugpy = R"(
try:
    import debugpy
    _cyxwiz_debugpy_available = True
except ImportError:
    _cyxwiz_debugpy_available = False
print(_cyxwiz_debugpy_available)
)";

    auto result = scripting_engine_->ExecuteScript(check_debugpy);
    if (result.success) {
        std::string output = result.output;
        // Trim whitespace
        output.erase(0, output.find_first_not_of(" \t\n\r"));
        output.erase(output.find_last_not_of(" \t\n\r") + 1);
        debugpy_available_ = (output == "True");
    }

    if (debugpy_available_) {
        spdlog::info("DebuggerManager: debugpy is available");
    } else {
        spdlog::info("DebuggerManager: debugpy not available, using sys.settrace fallback");
    }

    // Setup the debug trace infrastructure
    SetupTraceFunction();

    initialized_ = true;
    state_ = DebugState::Disconnected;

    spdlog::info("DebuggerManager initialized");
    return true;
}

void DebuggerManager::Shutdown() {
    if (!initialized_) return;

    RemoveTraceFunction();
    ClearAllBreakpoints();

    scripting_engine_ = nullptr;
    initialized_ = false;
    state_ = DebugState::Disconnected;

    spdlog::info("DebuggerManager shutdown");
}

bool DebuggerManager::IsConnected() const {
    return initialized_ && state_ != DebugState::Disconnected;
}

bool DebuggerManager::HasDebugpy() const {
    return debugpy_available_;
}

void DebuggerManager::SetupTraceFunction() {
    if (!scripting_engine_) return;

    // Install a Python trace function for debugging
    std::string trace_setup = R"(
import sys

# CyxWiz debugger state
class _CyxWizDebugger:
    def __init__(self):
        self.breakpoints = {}  # {cell_id: {line: Breakpoint}}
        self.paused = False
        self.step_mode = None  # 'over', 'into', 'out', or None
        self.step_depth = 0
        self.current_depth = 0

    def add_breakpoint(self, cell_id, line, condition=''):
        if cell_id not in self.breakpoints:
            self.breakpoints[cell_id] = {}
        self.breakpoints[cell_id][line] = {'condition': condition, 'enabled': True}

    def remove_breakpoint(self, cell_id, line):
        if cell_id in self.breakpoints and line in self.breakpoints[cell_id]:
            del self.breakpoints[cell_id][line]

    def clear_breakpoints(self):
        self.breakpoints = {}

    def check_breakpoint(self, cell_id, line):
        if cell_id not in self.breakpoints:
            return False
        if line not in self.breakpoints[cell_id]:
            return False
        bp = self.breakpoints[cell_id][line]
        if not bp.get('enabled', True):
            return False
        condition = bp.get('condition', '')
        if condition:
            try:
                return eval(condition)
            except:
                return False
        return True

    def trace_calls(self, frame, event, arg):
        if event == 'call':
            self.current_depth += 1
        elif event == 'return':
            self.current_depth -= 1

        if event == 'line':
            # Get current location
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            # Check for breakpoint
            # For now, we use filename as cell_id (will be mapped properly)
            if self.check_breakpoint(filename, lineno):
                self.paused = True
                # In a full implementation, this would signal the C++ side
                print(f"[CYXWIZ_BREAKPOINT_HIT] {filename}:{lineno}")

            # Handle stepping
            if self.step_mode == 'into':
                self.paused = True
                self.step_mode = None
            elif self.step_mode == 'over':
                if self.current_depth <= self.step_depth:
                    self.paused = True
                    self.step_mode = None
            elif self.step_mode == 'out':
                if self.current_depth < self.step_depth:
                    self.paused = True
                    self.step_mode = None

        return self.trace_calls

_cyxwiz_debugger = _CyxWizDebugger()

def _cyxwiz_enable_trace():
    sys.settrace(_cyxwiz_debugger.trace_calls)

def _cyxwiz_disable_trace():
    sys.settrace(None)

def _cyxwiz_add_breakpoint(cell_id, line, condition=''):
    _cyxwiz_debugger.add_breakpoint(cell_id, line, condition)

def _cyxwiz_remove_breakpoint(cell_id, line):
    _cyxwiz_debugger.remove_breakpoint(cell_id, line)

def _cyxwiz_clear_breakpoints():
    _cyxwiz_debugger.clear_breakpoints()

def _cyxwiz_step_over():
    _cyxwiz_debugger.step_mode = 'over'
    _cyxwiz_debugger.step_depth = _cyxwiz_debugger.current_depth
    _cyxwiz_debugger.paused = False

def _cyxwiz_step_into():
    _cyxwiz_debugger.step_mode = 'into'
    _cyxwiz_debugger.paused = False

def _cyxwiz_step_out():
    _cyxwiz_debugger.step_mode = 'out'
    _cyxwiz_debugger.step_depth = _cyxwiz_debugger.current_depth
    _cyxwiz_debugger.paused = False

def _cyxwiz_continue():
    _cyxwiz_debugger.step_mode = None
    _cyxwiz_debugger.paused = False

def _cyxwiz_get_locals():
    import json
    import sys
    frame = sys._getframe(1)
    result = {}
    for k, v in frame.f_locals.items():
        if not k.startswith('_'):
            try:
                result[k] = repr(v)[:100]
            except:
                result[k] = '<error>'
    return json.dumps(result)

def _cyxwiz_get_stack():
    import sys
    import json
    frames = []
    frame = sys._getframe()
    frame_id = 0
    # Skip the _cyxwiz_get_stack frame itself
    frame = frame.f_back
    while frame is not None:
        # Skip internal frames
        filename = frame.f_code.co_filename
        if not filename.startswith('<') or filename == '<string>':
            frame_info = {
                'id': frame_id,
                'name': frame.f_code.co_name,
                'filename': filename,
                'line': frame.f_lineno,
                'locals': {}
            }
            # Get local variables (limited repr)
            for k, v in frame.f_locals.items():
                if not k.startswith('_'):
                    try:
                        frame_info['locals'][k] = repr(v)[:100]
                    except:
                        frame_info['locals'][k] = '<error>'
            frames.append(frame_info)
            frame_id += 1
        frame = frame.f_back
    return json.dumps(frames)
)";

    auto result = scripting_engine_->ExecuteScript(trace_setup);
    if (!result.success) {
        spdlog::error("DebuggerManager: Failed to setup trace function: {}", result.error_message);
    }
}

void DebuggerManager::RemoveTraceFunction() {
    if (!scripting_engine_) return;

    scripting_engine_->ExecuteScript("_cyxwiz_disable_trace()");
}

int DebuggerManager::AddBreakpoint(const std::string& cell_id, int line, const std::string& condition) {
    std::lock_guard<std::mutex> lock(breakpoints_mutex_);

    Breakpoint bp;
    bp.id = next_breakpoint_id_++;
    bp.cell_id = cell_id;
    bp.line = line;
    bp.condition = condition;
    bp.enabled = true;
    bp.hit_count = 0;

    breakpoints_.push_back(bp);

    // Notify Python
    if (scripting_engine_) {
        std::string escaped_condition = condition;
        // Basic escape for Python string
        size_t pos = 0;
        while ((pos = escaped_condition.find("'", pos)) != std::string::npos) {
            escaped_condition.replace(pos, 1, "\\'");
            pos += 2;
        }

        std::string cmd = "_cyxwiz_add_breakpoint('" + cell_id + "', " +
                          std::to_string(line) + ", '" + escaped_condition + "')";
        scripting_engine_->ExecuteScript(cmd);
    }

    spdlog::info("Added breakpoint {} at {}:{}", bp.id, cell_id, line);
    return bp.id;
}

void DebuggerManager::RemoveBreakpoint(int id) {
    std::lock_guard<std::mutex> lock(breakpoints_mutex_);

    auto it = std::find_if(breakpoints_.begin(), breakpoints_.end(),
                           [id](const Breakpoint& bp) { return bp.id == id; });

    if (it != breakpoints_.end()) {
        // Notify Python
        if (scripting_engine_) {
            std::string cmd = "_cyxwiz_remove_breakpoint('" + it->cell_id + "', " +
                              std::to_string(it->line) + ")";
            scripting_engine_->ExecuteScript(cmd);
        }

        spdlog::info("Removed breakpoint {} at {}:{}", it->id, it->cell_id, it->line);
        breakpoints_.erase(it);
    }
}

void DebuggerManager::EnableBreakpoint(int id, bool enabled) {
    std::lock_guard<std::mutex> lock(breakpoints_mutex_);

    auto it = std::find_if(breakpoints_.begin(), breakpoints_.end(),
                           [id](const Breakpoint& bp) { return bp.id == id; });

    if (it != breakpoints_.end()) {
        it->enabled = enabled;
        spdlog::info("Breakpoint {} {}", id, enabled ? "enabled" : "disabled");
    }
}

void DebuggerManager::ClearAllBreakpoints() {
    std::lock_guard<std::mutex> lock(breakpoints_mutex_);

    breakpoints_.clear();
    next_breakpoint_id_ = 1;

    if (scripting_engine_) {
        scripting_engine_->ExecuteScript("_cyxwiz_clear_breakpoints()");
    }

    spdlog::info("Cleared all breakpoints");
}

const std::vector<Breakpoint>& DebuggerManager::GetBreakpoints() const {
    return breakpoints_;
}

std::vector<Breakpoint> DebuggerManager::GetBreakpointsForCell(const std::string& cell_id) const {
    std::lock_guard<std::mutex> lock(breakpoints_mutex_);

    std::vector<Breakpoint> result;
    for (const auto& bp : breakpoints_) {
        if (bp.cell_id == cell_id) {
            result.push_back(bp);
        }
    }
    return result;
}

void DebuggerManager::Continue() {
    if (state_ != DebugState::Paused) return;

    if (scripting_engine_) {
        scripting_engine_->ExecuteScript("_cyxwiz_continue()");
    }

    state_ = DebugState::Running;
    NotifyStateChange(DebugState::Running);
}

void DebuggerManager::StepOver() {
    if (state_ != DebugState::Paused) return;

    step_mode_ = StepMode::Over;
    if (scripting_engine_) {
        scripting_engine_->ExecuteScript("_cyxwiz_step_over()");
    }

    state_ = DebugState::Stepping;
    NotifyStateChange(DebugState::Stepping);
}

void DebuggerManager::StepInto() {
    if (state_ != DebugState::Paused) return;

    step_mode_ = StepMode::Into;
    if (scripting_engine_) {
        scripting_engine_->ExecuteScript("_cyxwiz_step_into()");
    }

    state_ = DebugState::Stepping;
    NotifyStateChange(DebugState::Stepping);
}

void DebuggerManager::StepOut() {
    if (state_ != DebugState::Paused) return;

    step_mode_ = StepMode::Out;
    if (scripting_engine_) {
        scripting_engine_->ExecuteScript("_cyxwiz_step_out()");
    }

    state_ = DebugState::Stepping;
    NotifyStateChange(DebugState::Stepping);
}

void DebuggerManager::Pause() {
    if (state_ != DebugState::Running) return;

    // Request pause (will pause on next line)
    if (scripting_engine_) {
        scripting_engine_->ExecuteScript("_cyxwiz_debugger.paused = True");
    }
}

void DebuggerManager::Stop() {
    RemoveTraceFunction();
    state_ = DebugState::Disconnected;
    NotifyStateChange(DebugState::Disconnected);
}

DebugState DebuggerManager::GetState() const {
    return state_;
}

const std::vector<StackFrame>& DebuggerManager::GetCallStack() const {
    return call_stack_;
}

int DebuggerManager::GetCurrentLine() const {
    return current_line_;
}

std::string DebuggerManager::GetCurrentCellId() const {
    std::lock_guard<std::mutex> lock(position_mutex_);
    return current_cell_id_;
}

std::string DebuggerManager::EvaluateExpression(const std::string& expr) {
    if (!scripting_engine_) return "";

    // Escape the expression for Python
    std::string escaped = expr;
    size_t pos = 0;
    while ((pos = escaped.find("'", pos)) != std::string::npos) {
        escaped.replace(pos, 1, "\\'");
        pos += 2;
    }

    std::string cmd = "print(repr(" + escaped + "))";
    auto result = scripting_engine_->ExecuteCommand(cmd);
    return result.output;
}

std::map<std::string, std::string> DebuggerManager::GetLocals() {
    std::map<std::string, std::string> result;

    if (!scripting_engine_) return result;

    auto exec_result = scripting_engine_->ExecuteCommand("print(_cyxwiz_get_locals())");
    if (exec_result.success && !exec_result.output.empty()) {
        // Parse JSON response
        // For now, return raw output - proper parsing would use nlohmann::json
        result["_raw"] = exec_result.output;
    }

    return result;
}

std::map<std::string, std::string> DebuggerManager::GetGlobals() {
    std::map<std::string, std::string> result;

    if (!scripting_engine_) return result;

    std::string cmd = R"(
import json
result = {}
for k, v in globals().items():
    if not k.startswith('_'):
        try:
            result[k] = repr(v)[:100]
        except:
            result[k] = '<error>'
print(json.dumps(result))
)";

    auto exec_result = scripting_engine_->ExecuteScript(cmd);
    if (exec_result.success && !exec_result.output.empty()) {
        result["_raw"] = exec_result.output;
    }

    return result;
}

void DebuggerManager::SetBreakpointHitCallback(BreakpointHitCallback callback) {
    breakpoint_callback_ = callback;
}

void DebuggerManager::SetStateChangedCallback(StateChangedCallback callback) {
    state_callback_ = callback;
}

void DebuggerManager::ExecuteWithDebug(const std::string& code, const std::string& cell_id) {
    if (!scripting_engine_) return;

    // Enable tracing
    scripting_engine_->ExecuteScript("_cyxwiz_enable_trace()");

    state_ = DebugState::Running;
    NotifyStateChange(DebugState::Running);

    {
        std::lock_guard<std::mutex> lock(position_mutex_);
        current_cell_id_ = cell_id;
    }

    // Execute the code
    auto result = scripting_engine_->ExecuteScript(code);

    // Disable tracing
    scripting_engine_->ExecuteScript("_cyxwiz_disable_trace()");

    state_ = DebugState::Disconnected;
    NotifyStateChange(DebugState::Disconnected);
}

void DebuggerManager::NotifyStateChange(DebugState new_state) {
    if (state_callback_) {
        state_callback_(new_state);
    }
}

void DebuggerManager::UpdateCallStack() {
    if (!scripting_engine_) {
        return;
    }

    // Get call stack from Python
    auto result = scripting_engine_->ExecuteCommand("print(_cyxwiz_get_stack())");
    if (!result.success || result.output.empty()) {
        spdlog::debug("Failed to get call stack from Python");
        return;
    }

    // Parse JSON response
    try {
        std::lock_guard<std::mutex> lock(stack_mutex_);
        call_stack_.clear();

        // Trim whitespace from output
        std::string json_str = result.output;
        json_str.erase(0, json_str.find_first_not_of(" \t\n\r"));
        json_str.erase(json_str.find_last_not_of(" \t\n\r") + 1);

        auto frames_json = nlohmann::json::parse(json_str);

        for (const auto& frame_json : frames_json) {
            StackFrame frame;
            frame.id = frame_json.value("id", 0);
            frame.name = frame_json.value("name", "<unknown>");
            frame.filename = frame_json.value("filename", "");
            frame.line = frame_json.value("line", 0);

            // Parse locals
            if (frame_json.contains("locals") && frame_json["locals"].is_object()) {
                for (auto& [key, value] : frame_json["locals"].items()) {
                    frame.locals[key] = value.get<std::string>();
                }
            }

            call_stack_.push_back(frame);
        }

        spdlog::debug("Updated call stack with {} frames", call_stack_.size());

    } catch (const nlohmann::json::exception& e) {
        spdlog::error("Failed to parse call stack JSON: {}", e.what());
    }
}

bool DebuggerManager::CheckBreakpoint(const std::string& cell_id, int line) {
    std::lock_guard<std::mutex> lock(breakpoints_mutex_);

    for (const auto& bp : breakpoints_) {
        if (bp.enabled && bp.cell_id == cell_id && bp.line == line) {
            return true;
        }
    }
    return false;
}

void DebuggerManager::HandlePause(const std::string& cell_id, int line) {
    {
        std::lock_guard<std::mutex> lock(position_mutex_);
        current_cell_id_ = cell_id;
    }
    current_line_ = line;

    // Update call stack when paused
    UpdateCallStack();

    state_ = DebugState::Paused;
    NotifyStateChange(DebugState::Paused);

    if (breakpoint_callback_) {
        breakpoint_callback_(cell_id, line);
    }
}

} // namespace scripting
