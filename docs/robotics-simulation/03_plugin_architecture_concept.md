# Plugin Architecture Concept (MATLAB/Simulink Model)

> Date: 2026-01-20
> Status: CONCEPT - NOT FOR IMMEDIATE IMPLEMENTATION

---

## Inspiration: MATLAB/Simulink

| Component | MATLAB Equivalent | Purpose |
|-----------|-------------------|---------|
| CyxWiz Engine | MATLAB | Core computation + GUI |
| CyxWiz RL | Reinforcement Learning Toolbox | RL training |
| CyxWiz Robotics | Robotics System Toolbox | ROS, kinematics |
| CyxWiz AutoDrive | Automated Driving Toolbox | ADAS, simulation |
| CyxWiz Vision | Computer Vision Toolbox | Detection, segmentation |
| CyxWiz NLP | Text Analytics Toolbox | NLP, transformers |

---

## Proposed Architecture

### Core Engine (Always Present)

```
CyxWiz Engine
├── GUI Framework (ImGui + docking)
├── Node Editor (base node types)
├── Dataset Manager
├── Training Executor
├── Model Serialization
├── Python Scripting
├── Plugin Manager ← Extension point
└── P2P Network Client
```

### Plugin Interface (Conceptual)

```cpp
namespace cyxwiz {

// Base plugin interface
class IPlugin {
public:
    virtual ~IPlugin() = default;

    // Lifecycle
    virtual PluginInfo GetInfo() const = 0;
    virtual bool OnLoad(PluginContext& ctx) = 0;
    virtual void OnUnload() = 0;

    // UI Integration
    virtual void OnRenderMenuItems() {}  // Add menu items
    virtual void OnRenderPanels() {}     // Add custom panels
    virtual void OnRenderNodeEditor() {} // Add node types

    // Training Integration
    virtual void OnPreTraining(TrainingContext& ctx) {}
    virtual void OnPostEpoch(TrainingContext& ctx) {}
    virtual void OnPostTraining(TrainingContext& ctx) {}

    // Data Integration
    virtual std::vector<DataLoaderFactory> GetDataLoaders() { return {}; }

    // Node Types
    virtual std::vector<NodeTypeInfo> GetNodeTypes() { return {}; }
};

struct PluginInfo {
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    std::vector<std::string> dependencies;  // Other plugins required
};

struct PluginContext {
    NodeEditor* node_editor;
    DataRegistry* data_registry;
    TrainingExecutor* training_executor;
    ScriptingEngine* scripting_engine;
    // ... access to core systems
};

} // namespace cyxwiz
```

### Plugin Discovery

```
CyxWiz/
├── bin/
│   └── cyxwiz-engine.exe
├── plugins/
│   ├── cyxwiz-rl/
│   │   ├── plugin.json        # Metadata
│   │   ├── cyxwiz-rl.dll      # Binary
│   │   └── resources/         # Icons, etc.
│   ├── cyxwiz-robotics/
│   └── cyxwiz-vision/
└── user-plugins/              # User-installed
```

### plugin.json Example

```json
{
    "name": "CyxWiz RL",
    "version": "1.0.0",
    "author": "CyxWiz Team",
    "description": "Reinforcement Learning support for CyxWiz",
    "engine_version": ">=2.0.0",
    "dependencies": [],
    "entry_point": "cyxwiz-rl.dll",
    "node_types": [
        "GymEnvironment",
        "PolicyNetwork",
        "ValueNetwork",
        "ReplayBuffer",
        "RolloutCollector"
    ],
    "panels": [
        "RLTrainingPanel",
        "EnvironmentViewer"
    ]
}
```

---

## Plugin Ideas

### Tier 1: Official Plugins (Built by CyxWiz team)

| Plugin | Purpose | Priority |
|--------|---------|----------|
| CyxWiz RL | Reinforcement learning | Medium |
| CyxWiz Vision | Computer vision tools | High |
| CyxWiz NLP | NLP/Transformers | Medium |
| CyxWiz AutoML | Hyperparameter tuning | High |

### Tier 2: Partner Plugins (Built with partners)

| Plugin | Purpose | Partner |
|--------|---------|---------|
| CyxWiz Robotics | ROS integration | TBD |
| CyxWiz AutoDrive | CARLA integration | TBD |
| CyxWiz Medical | Medical imaging | TBD |

### Tier 3: Community Plugins

Open for community contributions with plugin marketplace.

---

## CyxWiz RL Plugin (Detailed Concept)

### Features

1. **Gym/Gymnasium Integration**
   - Load any Gym-compatible environment
   - Render environment in viewport
   - Step/reset controls

2. **RL-Specific Nodes**
   - `GymEnvironment` - Environment wrapper
   - `PolicyNetwork` - Actor network
   - `ValueNetwork` - Critic network
   - `ReplayBuffer` - Experience storage
   - `RolloutCollector` - Episode collection

3. **Algorithms (Node Presets)**
   - DQN
   - PPO
   - SAC
   - TD3
   - A2C

4. **Distributed Rollouts**
   - Collect rollouts on Server Nodes
   - Centralized training on Engine
   - Leverages existing P2P infrastructure

### Node Graph Example

```
┌─────────────────┐     ┌─────────────────┐
│ GymEnvironment  │────▶│ PolicyNetwork   │
│ (CartPole-v1)   │     │ (MLP 64x64)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │ obs                   │ action
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ ReplayBuffer    │◀────│ RolloutCollector│
│ (size=100000)   │     │ (steps=2048)    │
└────────┬────────┘     └─────────────────┘
         │
         │ batch
         ▼
┌─────────────────┐     ┌─────────────────┐
│ ValueNetwork    │────▶│ PPOLoss         │
│ (MLP 64x64)     │     │ (clip=0.2)      │
└─────────────────┘     └─────────────────┘
```

---

## Business Model Options

### Model A: All Free (Open Source)
- Core: Free
- Plugins: Free
- Revenue: Support, training, enterprise features

### Model B: Freemium
- Core: Free
- Basic plugins: Free
- Advanced plugins: Paid ($X/month or one-time)

### Model C: Open Core
- Core: Open source (MIT/Apache)
- Official plugins: Paid
- Community plugins: Free marketplace

### Model D: Subscription Tiers
- Free: Core + 1 plugin
- Pro ($Y/month): Core + all plugins
- Team ($Z/month): Pro + collaboration features

---

## Implementation Phases

### Phase 0: Foundation (Current)
- Keep components loosely coupled
- Use interfaces where possible
- Document potential extension points
- **DO NOT build plugin system yet**

### Phase 1: API Design (When ready)
- Design `IPlugin` interface
- Design `PluginContext` (what plugins can access)
- Design node type registration
- Design panel registration

### Phase 2: Plugin Manager
- Plugin discovery (scan directories)
- Plugin loading (dynamic libraries)
- Dependency resolution
- Version compatibility checking

### Phase 3: First Plugin
- Build CyxWiz RL as first plugin
- Validate API design
- Document plugin development

### Phase 4: Plugin Marketplace
- Plugin submission
- Review process
- Distribution
- Licensing

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API instability | Plugins break on updates | Semantic versioning, deprecation policy |
| Poor plugin quality | Bad user experience | Review process, ratings |
| Security vulnerabilities | Code execution risks | Sandboxing, code signing |
| Maintenance burden | Support overhead | Clear plugin ownership |
| Fragmentation | Inconsistent UX | Design guidelines, UI components |

---

## Decision Framework

### When to build plugin system:

- [ ] Core product is stable and used
- [ ] Users are asking for extensibility
- [ ] Team has bandwidth beyond core features
- [ ] Clear business model for plugins

### When NOT to build yet:

- [x] Core product still in development
- [x] No users yet
- [x] Team is small
- [x] No validation of plugin demand

**Current status: NOT READY**

---

## Immediate Actions

### Do Now:
1. Design code with loose coupling
2. Use dependency injection where practical
3. Document "would-be extension points" in code comments
4. Keep this document updated as thinking evolves

### Do Later (Post-Users):
1. Survey users about extensibility needs
2. Design plugin API based on real requirements
3. Build plugin manager
4. Build first official plugin

---

## References

- MATLAB Plugin Architecture: https://www.mathworks.com/help/matlab/matlab_external/
- VS Code Extension API: https://code.visualstudio.com/api
- Unity Package Manager: https://docs.unity3d.com/Manual/Packages.html
- Unreal Engine Plugins: https://docs.unrealengine.com/en-US/ProductionPipelines/Plugins/

---

*"Make it work, make it right, make it fast. In that order."* - Kent Beck

*Translation: You can't optimize something that doesn't exist yet.*
