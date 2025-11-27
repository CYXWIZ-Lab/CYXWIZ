# Python Infinite Loop Handling in CyxWiz Engine

## Problem Statement

When users run Python scripts containing infinite loops (e.g., `while True: pass`) in the Script Editor, the application exhibited two critical issues:

1. **GUI Freeze**: The entire ImGui interface became unresponsive
2. **Application Crash**: Attempting to stop the script with Shift+F5 caused a segmentation fault (exit code 139)

## Root Cause Analysis

### Why the GUI Froze

Python uses a **Global Interpreter Lock (GIL)** that allows only one thread to execute Python bytecode at a time. When a Python script runs in a tight loop like `while True: pass`, it holds the GIL continuously without yielding, blocking any other Python operations.

Although we run Python scripts in a separate worker thread (`script_thread_`), the GIL coordination between threads still caused issues with pybind11's Python object management on the main thread.

### Why the Application Crashed

The initial solution used `PyThreadState_SetAsyncExc()` to inject a `KeyboardInterrupt` exception into the running Python thread:

```cpp
// The problematic approach (removed)
unsigned long tid = python_thread_id_.load();
if (tid != 0) {
    int result = PyThreadState_SetAsyncExc(tid, PyExc_KeyboardInterrupt);
    // This successfully stopped the script...
}
```

**The problem**: While `PyThreadState_SetAsyncExc()` successfully raised the exception and stopped the script, it **corrupted Python's internal interpreter state**. This corruption manifested when pybind11 objects went out of scope and their destructors attempted to decrement Python reference counts (`Py_DECREF`). The corrupted state caused a segmentation fault.

The crash consistently occurred in the cleanup phase after the script was cancelled, with logs showing:
```
[info] Script cancelled via KeyboardInterrupt (cooperative)
[CRASH - segmentation fault]
```

## Attempted Solutions (Failed)

### Attempt 1: Try-Catch Around Cleanup

Added comprehensive exception handling around all cleanup code:

```cpp
try {
    output_obj.attr("_callback") = py::none();
    output_obj.attr("_buffer") = "";
} catch (...) {
    spdlog::warn("Error clearing output callback, ignoring");
}
```

**Result**: Still crashed. The segfault occurred in pybind11's automatic destructor calls, which happen implicitly when `py::object` variables go out of scope - not in our explicit cleanup code.

### Attempt 2: Separate Cleanup Paths

Created different cleanup code paths for cancelled vs non-cancelled scripts:

```cpp
if (cancel_requested_) {
    // Minimal cleanup for cancelled scripts
    PyEval_SetTrace(nullptr, nullptr);
    PyErr_Clear();
} else {
    // Full cleanup for normal completion
    sys.attr("stdout") = original_stdout;
    // ...
}
```

**Result**: Still crashed. The issue wasn't in our cleanup code but in pybind11's automatic reference counting when `py::object` instances were destroyed.

### Attempt 3: Raw Python C API

Bypassed pybind11 entirely and used raw Python C API for cleanup:

```cpp
PyObject* pySys = PyImport_ImportModule("sys");
PyObject_SetAttrString(pySys, "stdout", original_stdout_raw);
Py_DECREF(pySys);
```

**Result**: Still crashed. While this avoided pybind11's cleanup, the `py::object` instances created earlier in the function still had corrupted state from the async exception.

### Attempt 4: Release All Python Objects

Called `.release()` on all `py::object` instances to prevent their destructors from running:

```cpp
// Release to prevent pybind11 from touching corrupted state
output_obj.release();
stdout_capture.release();
stderr_capture.release();
```

**Result**: Still crashed. The corruption from `PyThreadState_SetAsyncExc()` was too deep - even preventing destructor calls didn't help because other pybind11 internal structures were affected.

## Final Solution: Cooperative Cancellation Only

After extensive debugging, we concluded that **`PyThreadState_SetAsyncExc()` cannot be safely used with pybind11**. The async exception injection fundamentally corrupts interpreter state in ways that are incompatible with pybind11's reference counting.

### The Working Approach

We removed all uses of `PyThreadState_SetAsyncExc()` and rely entirely on **cooperative cancellation**:

1. **Trace Function**: A Python trace function (`sys.settrace()`) that checks a cancellation flag on every line:
   ```python
   def _cyxwiz_trace(frame, event, arg):
       if _cyxwiz_is_cancelled():
           raise KeyboardInterrupt("Script cancelled by user")
       return _cyxwiz_trace
   ```

2. **Output Callback**: The stdout/stderr wrapper checks the flag on every write:
   ```python
   def write(self, text):
       if _cyxwiz_is_cancelled():
           raise KeyboardInterrupt("Script cancelled")
       # ... normal write logic
   ```

3. **Shared Atomic Flag**: C++ sets an atomic flag that Python reads via ctypes:
   ```cpp
   // C++ side
   shared_cancel_flag_.store(1);

   // Python reads this via ctypes pointer
   _cyxwiz_cancel_ptr = ctypes.cast(flag_addr, ctypes.POINTER(ctypes.c_int))
   ```

### Implementation in `StopScript()`

```cpp
void ScriptingEngine::StopScript() {
    if (!script_running_) {
        return;
    }

    spdlog::info("Requesting script cancellation...");
    cancel_requested_ = true;

    // Set the shared atomic flag - the trace function will check this
    shared_cancel_flag_.store(1);

    // NOTE: We deliberately do NOT use PyThreadState_SetAsyncExc anymore.
    // While it can stop tight loops, it corrupts Python's internal state
    // and causes crashes when pybind11 tries to clean up.
    //
    // Instead, we rely on:
    // 1. The trace function checking shared_cancel_flag_ on each line
    // 2. The output write() function checking the flag
    // 3. Cooperative cancellation for well-behaved scripts
    //
    // For truly uncooperative scripts (like "while True: pass"), users
    // will need to wait or force-close the application.
    spdlog::info("Cancellation flag set. Script will stop at next cooperative check point.");
}
```

## Trade-offs and Limitations

### What Works

| Script Type | Can Be Stopped? | How |
|-------------|-----------------|-----|
| `while True: print("x")` | Yes | Output callback checks flag |
| `for i in range(1000000): x = i` | Yes | Trace function checks on each line |
| Any script calling functions | Yes | Trace function checks on function calls |
| Scripts with I/O operations | Yes | I/O typically releases GIL |

### What Doesn't Work

| Script Type | Can Be Stopped? | Workaround |
|-------------|-----------------|------------|
| `while True: pass` | No | Force close application |
| Tight loops with no function calls | No | Add cancellation checks in script |
| C extension code holding GIL | No | Force close application |

### User Experience Mitigation

For uncooperative scripts, we implemented a **close confirmation dialog** in the application:

```cpp
void CyxWizApp::HandleCloseConfirmation() {
    // Shows dialog when user tries to close while script is running
    // Options: "Cancel" (go back), "Wait" (let script finish), "Force Close"
}
```

## Conclusion

The Python infinite loop cancellation problem turned out to be a fundamental incompatibility between `PyThreadState_SetAsyncExc()` and pybind11's reference counting system. While the async exception injection successfully stops Python code, the resulting interpreter state corruption causes crashes during cleanup.

The final solution accepts a trade-off:
- **Stability over completeness**: We prioritize a stable, crash-free application over the ability to stop every possible script
- **Cooperative cancellation**: Most real-world scripts (those with output, function calls, or I/O) can be cleanly stopped
- **Force close fallback**: For truly uncooperative scripts, users can use the force close option

This approach aligns with how many other Python embedding scenarios handle cancellation - cooperative mechanisms that work for well-behaved code, with process termination as the ultimate fallback.

## Files Modified

- `cyxwiz-engine/src/scripting/scripting_engine.cpp` - Cancellation logic
- `cyxwiz-engine/src/scripting/scripting_engine.h` - Shared flag declaration
- `cyxwiz-engine/src/application.cpp` - Close confirmation dialog

## References

- [Python C API - Thread State](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock)
- [PyThreadState_SetAsyncExc limitations](https://docs.python.org/3/c-api/init.html#c.PyThreadState_SetAsyncExc)
- [pybind11 GIL documentation](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)
