#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>

namespace scripting {

class ScriptingEngine;

/**
 * Breakpoint information
 */
struct Breakpoint {
    int id = 0;                 // Unique breakpoint ID
    std::string cell_id;        // Cell containing the breakpoint
    int line = 0;               // Line number within the cell
    std::string condition;      // Optional condition expression
    bool enabled = true;        // Breakpoint enabled state
    int hit_count = 0;          // Number of times hit
};

/**
 * Stack frame information
 */
struct StackFrame {
    int id = 0;                 // Frame ID
    std::string name;           // Function name
    std::string filename;       // File or cell ID
    int line = 0;               // Current line
    std::map<std::string, std::string> locals;  // Local variables
};

/**
 * Debug state enumeration
 */
enum class DebugState {
    Disconnected,   // Debugger not attached
    Running,        // Script executing normally
    Paused,         // Paused at breakpoint or step
    Stepping        // Currently stepping
};

/**
 * DebuggerManager - Manages debugging of Python scripts
 *
 * Uses Python's sys.settrace for basic stepping and breakpoint support.
 * Optionally integrates with debugpy for enhanced debugging features.
 */
class DebuggerManager {
public:
    DebuggerManager();
    ~DebuggerManager();

    // ========== Lifecycle ==========

    /**
     * Initialize the debugger with the scripting engine
     */
    bool Initialize(ScriptingEngine* engine);

    /**
     * Shutdown and cleanup
     */
    void Shutdown();

    /**
     * Check if debugger is connected/active
     */
    bool IsConnected() const;

    /**
     * Check if debugpy is available
     */
    bool HasDebugpy() const;

    // ========== Breakpoint Management ==========

    /**
     * Add a breakpoint
     * @return Breakpoint ID (0 if failed)
     */
    int AddBreakpoint(const std::string& cell_id, int line, const std::string& condition = "");

    /**
     * Remove a breakpoint by ID
     */
    void RemoveBreakpoint(int id);

    /**
     * Enable/disable a breakpoint
     */
    void EnableBreakpoint(int id, bool enabled);

    /**
     * Clear all breakpoints
     */
    void ClearAllBreakpoints();

    /**
     * Get all breakpoints
     */
    const std::vector<Breakpoint>& GetBreakpoints() const;

    /**
     * Get breakpoints for a specific cell
     */
    std::vector<Breakpoint> GetBreakpointsForCell(const std::string& cell_id) const;

    // ========== Execution Control ==========

    /**
     * Continue execution (resume after pause)
     */
    void Continue();

    /**
     * Step over current line
     */
    void StepOver();

    /**
     * Step into function call
     */
    void StepInto();

    /**
     * Step out of current function
     */
    void StepOut();

    /**
     * Pause execution
     */
    void Pause();

    /**
     * Stop debugging session
     */
    void Stop();

    // ========== State ==========

    /**
     * Get current debug state
     */
    DebugState GetState() const;

    /**
     * Get call stack (when paused)
     */
    const std::vector<StackFrame>& GetCallStack() const;

    /**
     * Get current line (when paused)
     */
    int GetCurrentLine() const;

    /**
     * Get current cell ID (when paused)
     */
    std::string GetCurrentCellId() const;

    // ========== Variable Inspection ==========

    /**
     * Evaluate an expression in current context
     */
    std::string EvaluateExpression(const std::string& expr);

    /**
     * Get local variables at current frame
     */
    std::map<std::string, std::string> GetLocals();

    /**
     * Get global variables
     */
    std::map<std::string, std::string> GetGlobals();

    // ========== Callbacks ==========

    using BreakpointHitCallback = std::function<void(const std::string& cell_id, int line)>;
    using StateChangedCallback = std::function<void(DebugState state)>;

    /**
     * Set callback for breakpoint hit events
     */
    void SetBreakpointHitCallback(BreakpointHitCallback callback);

    /**
     * Set callback for state change events
     */
    void SetStateChangedCallback(StateChangedCallback callback);

    // ========== Debug Execution ==========

    /**
     * Execute code with debugging enabled
     * @param code The Python code to execute
     * @param cell_id The cell ID for breakpoint mapping
     */
    void ExecuteWithDebug(const std::string& code, const std::string& cell_id);

private:
    ScriptingEngine* scripting_engine_ = nullptr;

    // State
    std::atomic<DebugState> state_{DebugState::Disconnected};
    std::atomic<bool> debugpy_available_{false};
    std::atomic<bool> initialized_{false};

    // Breakpoints
    std::vector<Breakpoint> breakpoints_;
    int next_breakpoint_id_ = 1;
    mutable std::mutex breakpoints_mutex_;

    // Call stack
    std::vector<StackFrame> call_stack_;
    mutable std::mutex stack_mutex_;

    // Current position
    std::atomic<int> current_line_{0};
    std::string current_cell_id_;
    mutable std::mutex position_mutex_;

    // Step control
    enum class StepMode { None, Over, Into, Out };
    std::atomic<StepMode> step_mode_{StepMode::None};
    int step_depth_ = 0;

    // Callbacks
    BreakpointHitCallback breakpoint_callback_;
    StateChangedCallback state_callback_;

    // Internal methods
    void SetupTraceFunction();
    void RemoveTraceFunction();
    void NotifyStateChange(DebugState new_state);
    void UpdateCallStack();
    bool CheckBreakpoint(const std::string& cell_id, int line);
    void HandlePause(const std::string& cell_id, int line);
};

} // namespace scripting
