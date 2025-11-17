# Phase 5, Task 3: Startup Scripts - Design Document

## Overview

Implement automatic execution of Python scripts (.cyx files) when the application starts or when a project is loaded.

## Use Cases

1. **Initialization**: Auto-import commonly used modules
2. **Setup**: Configure environment variables, paths
3. **Data Loading**: Automatically load datasets
4. **Customization**: User-specific preferences and settings

## Design

### Configuration File

**Location**: `startup_scripts.txt` (in project root or user directory)

**Format**: Simple text file, one script path per line
```
# Startup scripts for CyxWiz
# Lines starting with # are comments
# Paths can be absolute or relative to project root

scripts/init.cyx
scripts/load_data.cyx
C:/Users/username/my_setup.cyx
```

### Execution Flow

```
Application Start
    ├─ Initialize ScriptingEngine
    ├─ Load startup_scripts.txt
    ├─ For each script in list:
    │   ├─ Check if file exists
    │   ├─ Execute script (with sandbox if enabled)
    │   ├─ Log output to CommandWindow
    │   └─ Continue on error (don't block startup)
    └─ Show "Startup scripts complete" message
```

### Implementation Components

**1. StartupScriptManager Class**
- Reads configuration file
- Validates script paths
- Executes scripts in order
- Reports errors without blocking

**2. Integration Points**
- `Application::Initialize()` - Run after ScriptingEngine setup
- `MainWindow` - Access to CommandWindow for output display

**3. UI Controls**
- Settings → Startup Scripts
- Enable/disable startup scripts
- Edit startup script list
- Test startup scripts

### Error Handling

**Strategies**:
- **Continue on error**: Don't block application startup
- **Log errors**: Display in CommandWindow
- **Timeout protection**: Use sandbox timeout (60s per script)
- **Safe mode**: Hold Shift on startup to skip startup scripts

## File Structure

```
cyxwiz-engine/
├── src/
│   └── scripting/
│       ├── startup_script_manager.h
│       └── startup_script_manager.cpp
└── scripts/
    └── startup/
        ├── README.md
        └── examples/
            ├── init_imports.cyx
            ├── load_common_data.cyx
            └── setup_environment.cyx
```

## API

```cpp
// StartupScriptManager.h
class StartupScriptManager {
public:
    StartupScriptManager(std::shared_ptr<ScriptingEngine> engine);

    // Load configuration
    bool LoadConfig(const std::string& config_file = "startup_scripts.txt");

    // Execute all startup scripts
    bool ExecuteAll(CommandWindowPanel* output_window = nullptr);

    // Execute single script
    bool ExecuteScript(const std::string& filepath);

    // Configuration
    void AddScript(const std::string& filepath);
    void RemoveScript(const std::string& filepath);
    bool SaveConfig(const std::string& config_file = "startup_scripts.txt");

    // Query
    std::vector<std::string> GetScriptList() const;
    bool IsEnabled() const;
    void SetEnabled(bool enabled);

private:
    std::shared_ptr<ScriptingEngine> scripting_engine_;
    std::vector<std::string> script_paths_;
    bool enabled_;
};
```

## Usage Example

**Application Initialization** (`application.cpp`):
```cpp
void Application::Initialize() {
    // ... existing initialization ...

    // Create scripting engine
    scripting_engine_ = std::make_shared<ScriptingEngine>();

    // Create startup script manager
    startup_manager_ = std::make_unique<StartupScriptManager>(scripting_engine_);

    // Load and execute startup scripts
    if (startup_manager_->LoadConfig()) {
        startup_manager_->ExecuteAll(main_window_->GetCommandWindow());
    }

    // ... rest of initialization ...
}
```

**User Configuration** (`startup_scripts.txt`):
```
# My CyxWiz startup scripts

# Import common libraries
scripts/startup/init_imports.cyx

# Load my datasets
scripts/startup/load_data.cyx

# Custom helper functions
C:/Users/me/cyxwiz_helpers.cyx
```

**Example Startup Script** (`scripts/startup/init_imports.cyx`):
```python
# init_imports.cyx - Auto-import common libraries

print("Loading common imports...")

import math
import random
import json

# Try to import numpy if available
try:
    import numpy as np
    print("  NumPy loaded")
except ImportError:
    print("  NumPy not available")

# Try to import pandas if available
try:
    import pandas as pd
    print("  Pandas loaded")
except ImportError:
    print("  Pandas not available")

print("Imports complete!")
```

## UI Integration

### Settings Dialog

**Menu**: Edit → Preferences → Startup Scripts

**UI Elements**:
- [ ] Enable startup scripts (checkbox)
- List of scripts (with Add/Remove buttons)
- [ ] Run scripts in safe mode (with timeout)
- [Test] button - Run startup scripts now
- [Edit Config] button - Open startup_scripts.txt in editor

### CommandWindow Output

```
CyxWiz Python Command Window
Type 'help()' for help, 'clear' to clear output
Press Tab for auto-completion

=== Running startup scripts ===
Executing: scripts/startup/init_imports.cyx
Loading common imports...
  NumPy loaded
  Pandas loaded
Imports complete!

Executing: scripts/startup/load_data.cyx
Loading common datasets...
  Loaded: iris.csv (150 rows)
Dataset ready!

=== Startup scripts complete (2 scripts, 1.2s) ===

f:>
```

## Configuration Options

**Application-level settings**:
- `startup_scripts_enabled` (bool) - Enable/disable feature
- `startup_scripts_timeout` (int) - Max seconds per script
- `startup_scripts_continue_on_error` (bool) - Continue if script fails
- `startup_scripts_safe_mode` (bool) - Run with sandbox enabled

**Storage**: Could use JSON config file:
```json
{
  "startup_scripts": {
    "enabled": true,
    "timeout": 60,
    "continue_on_error": true,
    "safe_mode": true,
    "scripts": [
      "scripts/startup/init_imports.cyx",
      "scripts/startup/load_data.cyx"
    ]
  }
}
```

## Safe Mode

**Hold Shift on startup** to skip startup scripts:
- Useful if startup script causes crash
- Shows message: "Shift detected - Skipping startup scripts"
- Allows user to fix broken scripts

**Implementation**:
```cpp
bool Application::ShouldSkipStartupScripts() {
    // Check if Shift key is held during startup
    return ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
           ImGui::IsKeyDown(ImGuiKey_RightShift);
}
```

## Testing

**Test Cases**:
1. **No config file**: Should start normally, no errors
2. **Empty config file**: Should start normally
3. **Valid scripts**: Execute in order, show output
4. **Invalid path**: Log error, continue with next script
5. **Script with error**: Log error, continue
6. **Script timeout**: Timeout after 60s, continue
7. **Safe mode**: Shift key skips all scripts

**Test Scripts**:
```
test_startup_simple.cyx    - Print "Hello from startup"
test_startup_imports.cyx   - Import math, random
test_startup_error.cyx     - Intentional error
test_startup_timeout.cyx   - Infinite loop (tests timeout)
```

## Implementation Plan

**Step 1**: Create StartupScriptManager class
- Read/write configuration file
- Execute scripts with error handling

**Step 2**: Integrate with Application
- Call startup manager after ScriptingEngine init
- Pass CommandWindow for output

**Step 3**: Create example scripts
- init_imports.cyx
- load_common_data.cyx
- README.md with examples

**Step 4**: Add UI controls (optional)
- Settings dialog
- Enable/disable toggle
- Script list editor

**Step 5**: Test and document
- Test all error cases
- Create user guide
- Add to main documentation

## Future Enhancements

1. **Project-specific scripts**: Different scripts per project
2. **Script groups**: Run different sets based on context
3. **Dependency order**: Declare script dependencies
4. **Script marketplace**: Share startup scripts
5. **IDE integration**: Edit scripts in Script Editor with auto-complete
6. **Performance**: Parallel execution for independent scripts

## Success Criteria

Task 3 is **SUCCESSFUL** if:

1. ✅ StartupScriptManager class implemented
2. ✅ Configuration file loading works
3. ✅ Scripts execute on application start
4. ✅ Output displayed in CommandWindow
5. ✅ Error handling (continue on error)
6. ✅ Example startup scripts created
7. ✅ Documentation written
8. ✅ Build succeeds without errors

---

**Status**: 📋 Design complete, ready for implementation

**Next**: Implement StartupScriptManager class
