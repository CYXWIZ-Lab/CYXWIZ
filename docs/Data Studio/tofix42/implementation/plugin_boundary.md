# Assistant Plugin Boundary

## Purpose

Define the boundary between:

1. the CyxWiz engine
2. the optional assistant plugin frontend
3. the assistant backend/runtime

This keeps the assistant useful without making engine correctness depend on it.

## Design Goal

The assistant should be:

- read-only
- optional
- local-first
- replaceable
- version-aware

The engine should remain usable when the assistant is absent or unhealthy.

## High-Level Split

```text
CyxWiz Engine
  -> plugin host
  -> command window / panels / debugger / training views

Assistant Plugin Frontend
  -> panel UI
  -> future command routing
  -> engine-context collection
  -> backend request/response bridge

Assistant Backend
  -> retrieval index
  -> prompt builder
  -> local runtime adapter
  -> answer parser
  -> validation helpers
```

## Ownership

### Engine owns

- plugin discovery and loading
- panel hosting
- command window hosting
- debugger state and selected trace state
- training trace state
- graph/editor selection state
- engine version/build metadata

### Assistant plugin owns

- assistant UI surface in CyxWiz
- translation from engine context to assistant request
- lifecycle of backend calls
- rendering structured assistant output
- user-facing failure handling for assistant actions

### Assistant backend owns

- knowledge-pack loading
- retrieval and ranking
- packet generation
- prompt construction
- local model/runtime invocation
- structured answer parsing

## First Integration Shape

The first integration should use an Assistant panel via `IPanelProvider`.

Reason:

- panel plugins already exist
- command window assistant routing does not exist yet
- debugger and training context can be added later without changing the backend

The first panel should provide:

- one text input
- one Ask action
- one context selector or status area
- structured output view with citations

## Boundary Rule

The plugin should never expose raw engine internals directly to the model.

Instead:

1. engine exposes selected structured context to the plugin
2. plugin converts that into a bounded assistant request
3. backend performs retrieval and answering

This keeps the model-facing contract stable even if engine internals change.

## Engine to Plugin Contract

The engine should provide only bounded, read-only context.

### Minimum context for first panel version

```cpp
struct EngineAssistantContext {
    std::string engine_version;
    std::string build_id;
    std::string workspace_root;
    std::string active_graph_path;
    std::string selected_run_id;
    std::string selected_node_id;
    std::string selected_trace_id;
    std::string selected_panel;
};
```

Notes:

- all fields are optional except version/build metadata
- empty fields mean "no active selection"
- this is context, not a command surface

Current implementation:

- the engine stores this as `AssistantContextSnapshot` in `PluginContext`
- `PluginManager::SetAssistantContextSnapshotForAll` publishes the current
  snapshot to initialized plugin contexts
- `MainWindow` currently populates project root, graph path, graph hash, graph
  node/link counts, and selected node id/name/type
- `StudioDebuggerPanel` contributes a bounded debugger summary with selected
  trace identity and selected trace fields
- `StudioDebuggerPanel` contributes a bounded training summary with latest
  training state and terminal reason when available

### Extended debugger context

For debugger-specific actions, the engine may later provide:

```cpp
struct DebugTraceSelection {
    std::string run_id;
    std::string trace_id;
    std::string node_name;
    std::string node_type;
    std::string phase;
    std::string status;
    std::string payload_json;
};
```

### Extended training context

For training explanation, the engine may later provide:

```cpp
struct TrainingTraceSelection {
    std::string run_id;
    std::string terminal_reason;
    std::string terminal_message;
    std::string summary_json;
};
```

## Plugin to Backend Contract

The plugin should call the backend through a narrow request object.

```cpp
struct AssistantRequest {
    std::string command_name;
    std::string user_text;
    std::string engine_version;
    std::string build_id;
    std::string active_graph_path;
    std::string selected_run_id;
    std::string selected_node_id;
    std::string selected_trace_id;
    std::string debugger_context_json;
    std::string training_context_json;
    bool retrieval_only = false;
};
```

Rules:

- `command_name` identifies the intent, for example `ask`, `explain_trace`, or
  `explain_training`
- `user_text` is free text
- context JSON fields are optional and bounded
- request stays read-only

## Backend to Plugin Contract

The backend should return one structured response.

```cpp
struct AssistantCitation {
    std::string path;
    int line_start = 0;
    int line_end = 0;
    std::string title;
    std::string source_type;
};

struct AssistantResponse {
    bool success = false;
    bool parsed = false;
    std::string answer;
    std::string evidence;
    std::string unknowns;
    std::string unsupported_or_not_implemented;
    std::vector<AssistantCitation> citations;
    std::string raw_output;
    std::string error;
};
```

Rules:

- `success` means the backend call completed usefully
- `parsed` means required sections were extracted
- `raw_output` is for diagnostics, not default UI display
- citations should be explicit whenever evidence exists

## Recommended Commands

The frontend should start with a small fixed command set:

1. `ask`
   - general source-aware question
2. `explain_trace`
   - explain current debugger trace selection
3. `explain_training`
   - explain current training terminal state
4. `find_source`
   - retrieve source locations for selected concept

Do not start with open-ended mutation commands.

## Panel API Shape

For the first implementation, the plugin panel can be thin:

```cpp
class IAssistantBackend {
public:
    virtual ~IAssistantBackend() = default;
    virtual AssistantResponse Run(const AssistantRequest& request) = 0;
};
```

The panel should:

1. capture text and current selection context
2. build `AssistantRequest`
3. call `Run`
4. render the structured response

## Future Command Window Routing

Command Window routing should be added only after the panel path works.

Recommended future interface:

```cpp
class IAssistantProvider {
public:
    virtual ~IAssistantProvider() = default;
    virtual bool CanHandleCommand(const std::string& command_name) const = 0;
    virtual AssistantResponse Run(const AssistantRequest& request) = 0;
};
```

Suggested slash commands:

- `/ask <question>`
- `/find-source <query>`
- `/explain-trace`
- `/explain-training`

This should remain explicit and separate from the Python REPL behavior.

## Knowledge-Pack Boundary

The plugin should not require a live repo scan on every startup.

The backend should load either:

- a shipped knowledge pack, or
- a developer override pointing at a local repo/index

Minimum knowledge-pack fields:

- manifest version
- engine version or commit band
- indexed chunk store
- retrieval metadata
- optional diagnostics metadata

## Versioning Rule

The plugin must pass engine version/build metadata to the backend.

The backend should reject or warn when:

- knowledge pack version does not match engine version policy
- retrieval assets are missing
- runtime is unavailable

This keeps explanations tied to the engine build they describe.

## Failure Behavior

Assistant failures must stay local to the plugin surface.

### If plugin is absent

- engine remains fully usable
- assistant actions are hidden or disabled

### If backend is unavailable

- panel shows assistant unavailable
- optional retrieval-only mode may still work

### If model runtime fails

- show error
- keep citations or retrieval result if available
- do not block engine UI

### If answer parsing fails

- show bounded diagnostic state
- do not present malformed output as trusted explanation

## Security and Scope Rules

The first assistant boundary should enforce:

- read-only behavior
- no graph mutation
- no source mutation
- no training control actions
- no shell execution from user assistant prompts

The assistant is an explanation and navigation tool first.

## Example End-to-End Flow

```text
User opens Assistant panel
  -> asks: "What source file defines DebugTraceRecord?"
  -> plugin gathers engine/build metadata
  -> plugin builds AssistantRequest(command_name="ask")
  -> backend retrieves evidence
  -> backend prompts local model
  -> backend parses structured answer
  -> plugin renders answer + citations
```

Debugger example:

```text
User selects trace
  -> clicks Explain Trace
  -> plugin serializes selected trace context
  -> backend retrieves relevant source/docs
  -> backend answers from evidence
  -> panel shows cited explanation
```

## Recommended Implementation Order

1. define backend interface in plugin/module boundary
2. implement Assistant panel via `IPanelProvider`
3. pass engine/build metadata and plain text requests
4. add debugger-context request support
5. add training-context request support
6. only then add Command Window slash-command routing

## Recommendation

Use this concrete boundary:

```text
CyxWiz engine owns UI hosting and structured context.
Assistant plugin owns request translation and rendering.
Assistant backend owns retrieval, prompting, runtime calls, and parsing.
```

That is the cleanest path from current RAG proof to real CyxWiz integration.
