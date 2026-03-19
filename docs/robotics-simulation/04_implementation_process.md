# MuJoCo Plugin — Implementation Process

> Started: 2026-01-31

## Engine Exploration Findings

Before implementation, we analyzed the CyxWiz Engine codebase to understand integration points. Key findings:

### OpenGL Context
- Engine uses **OpenGL 3.3 Core Profile** (Windows/Linux), 2.1 Compatibility (macOS)
- GLFW creates the context in `application.cpp`
- No existing FBO usage — MuJoCo viewport is greenfield
- ImGui texture display pattern: `ImGui::Image((ImTextureID)(intptr_t)tex_id, size)`
- Render loop: `NewFrame → MainWindow::Render() → ImGui::Render() → SwapBuffers`
- **Concern**: MuJoCo's built-in renderer uses OpenGL 1.5 compatibility profile functions. On Windows with Core Profile, we must use MuJoCo's offscreen rendering (`mjr_readPixels`) and upload to a Core Profile texture, OR create a secondary compatibility context. Phase 2 will address this.

### Node Editor Architecture
- Plugin nodes use `NodeType::PluginCustom` sentinel value
- Qualified name stored in `node.parameters["plugin_qualified_name"]` as `plugin_id:type_name`
- Appear in search palette as `Plugin/<category>/<name>`
- Code generation dispatches via `PluginNodeRegistry::GenerateCode()`
- Existing `GymEnvironment` node type provides a pattern for RL env nodes
- Node connections use typed pins (Tensor, Scalar, String, etc.)

### Training System & RL
- Supervised training loop in `TrainingExecutor::Train()` — epoch/batch structure
- Existing `GymConnector` bridges to Python Gymnasium via embedded interpreter
- Backend has `ReplayBuffer` and `EpsilonSchedule` for RL
- `TrainingContext::custom_metrics` map for plugin-injected metrics
- Plugin hooks invoked via `PluginTrainingHookManager` at epoch boundaries
- **Key decision**: Phase 1 will use headless physics stepping callable from training hooks. Full RL training loop integration comes later.

### Plugin Panel Rendering
- Plugin panels rendered at `MainWindow::Render()` line 2259
- Called via `PluginPanelRegistry::RenderAllVisible()`
- Uses snapshot pattern (copy panel list under mutex, render without lock)
- `TextureManager::GetOrCreateCachedTexture()` available for texture display
- Automatic ImGui docking — panels appear in dockspace like native panels

---

## Phase 1: Core Plugin Skeleton + Physics Wrapper

### Goal
Create a loadable MuJoCo plugin DLL that can:
1. Initialize/shutdown MuJoCo library
2. Load an MJCF model (InvertedPendulum as test)
3. Step physics, reset, get observations/rewards
4. Register as a training hook to log RL metrics

### Files Created
```
plugins/simulation/mujoco/
├── plugin.json
├── CMakeLists.txt
└── src/
    ├── mujoco_plugin.h       # IPlugin + ITrainingHook
    ├── mujoco_plugin.cpp     # Plugin lifecycle
    ├── mj_env_manager.h      # Physics wrapper interface
    └── mj_env_manager.cpp    # MuJoCo C API calls
```

### Status: COMPLETE

### Build Notes
- Pre-built MuJoCo 3.2.7 binary used (FetchContent caused spdlog/sdflib conflict with vcpkg)
- Plugin DLL: 50 KB + 4 MB mujoco.dll
- Engine discovers and loads plugin; FileSystem permission requires user approval
- Audited and fixed: CRITICAL-2, HIGH-3, MEDIUM-6 issues
