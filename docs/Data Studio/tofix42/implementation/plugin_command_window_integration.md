# Plugin and Command Window Integration

## Purpose

Explain what the local CyxWiz source-aware assistant would look like as a real
CyxWiz plugin/module, how a user would install and use it, and what existing
engine/plugin surfaces already exist today.

## What Exists Today

### Existing plugin system

CyxWiz already has a real plugin system.

Relevant code:

- `cyxwiz-engine/src/plugin/plugin_manager.h`
- `cyxwiz-engine/src/plugin/plugin_loader.h`
- `cyxwiz-engine/src/plugin/interfaces/`

Current built-in plugin interfaces:

- `INodeProvider`
- `IPanelProvider`
- `IDataProvider`
- `ITrainingHook`
- `IAnalyticsProvider`

Startup currently configures plugin search paths in:

- `cyxwiz-engine/src/main.cpp`

Current search paths:

1. `<exe_dir>/plugins`
2. `%APPDATA%/cyxwiz/plugins` on Windows
3. `~/.cyxwiz/plugins` on Linux/macOS

Plugins are discovered and loaded from directories containing `plugin.json`.

### Existing user surface

CyxWiz already has a Command Window panel.

Relevant code:

- `cyxwiz-engine/src/gui/panels/command_window.cpp`
- `cyxwiz-engine/src/gui/panels/command_window.h`

Important current truth:

- the Command Window is currently a Python REPL-style panel
- it is not a general assistant chat surface
- assistant behavior is explicit through slash commands
- the plugin system now exposes a narrow `IAssistantProvider` interface

So the plugin system is real, and the first assistant hook is now defined.

## What the User Experience Should Look Like

If implemented cleanly, the user flow should look like this.

### Install / enable

1. User installs the assistant plugin package into a CyxWiz plugin folder.
2. User opens **Plugin Manager** in CyxWiz.
3. User clicks **Scan & Load**.
4. Plugin becomes **Active**.
5. CyxWiz shows assistant features only when the plugin is active.

### First use

User opens the Command Window and types something like:

```text
/ask What source file defines DebugTraceRecord
```

or:

```text
/explain-trace
```

or:

```text
/find-source terminal_reason
```

The plugin then:

1. receives the command
2. builds retrieval context from the local knowledge index
3. optionally includes selected debugger or training context
4. calls the local answer runtime
5. returns a cited answer back into the Command Window

### Example result

```text
Answer:
cyxwiz-engine/src/core/debug_trace_record.h

Evidence:
[E1] cyxwiz-engine/src/core/debug_trace_record.h:31-46

Unknowns:
none

Unsupported or not implemented:
none
```

## Best First Use Cases

The first useful plugin-backed assistant use cases should be:

1. **Command Window source questions**
   - "What source file defines DebugTraceRecord?"
   - "Where does TFIDFVectorizer validate min_df?"

2. **Debugger explanation**
   - "Explain selected trace"
   - "Find related source"

3. **Training trace explanation**
   - "Why did training stop?"
   - "What does terminal_reason mean here?"

These are already aligned with the existing RAG harness.

## Recommended Product Shape

Use the plugin system, but keep the assistant architecture split into two
pieces:

1. **assistant backend**
   - retrieval index
   - prompt builder
   - runtime adapter
   - answer parser

2. **CyxWiz plugin frontend**
   - command window routing
   - optional panel(s)
   - debugger context collection
   - training context collection

That keeps the plugin thin and lets the assistant backend evolve separately.

## What the Plugin Package Should Contain

The plugin package should likely contain:

- `plugin.json`
- plugin DLL/shared library
- assistant config
- local knowledge index or knowledge-pack metadata
- optional helper scripts or runtime config files

Suggested structure:

```text
plugins/
  cyxwiz_assistant/
    plugin.json
    bin/
      cyxwiz_assistant.dll
    assistant/
      config.json
      knowledge_pack/
        index.json
        manifest.json
```

Do not require the plugin to rebuild the entire index on every startup.

## How the User Adds the Plugin

Based on the current plugin system, the user flow should be:

1. Copy the plugin directory into one of the CyxWiz plugin search paths:
   - `<exe_dir>/plugins/cyxwiz_assistant/`
   - `%APPDATA%/cyxwiz/plugins/cyxwiz_assistant/`

2. Ensure the folder contains:
   - `plugin.json`
   - platform library in `bin/`

3. Open CyxWiz.

4. Open **Plugin Manager**.

5. Click **Scan & Load**.

6. Confirm the plugin shows as active.

That flow matches the current plugin usage guide.

## What the Plugin Can Already Do Today

Using current interfaces, the assistant plugin could already expose:

- a panel via `IPanelProvider`
- analytics or diagnostics helpers via `IAnalyticsProvider`
- training lifecycle hooks via `ITrainingHook`

This means an assistant plugin could already add:

- an "Assistant" panel
- a "Trace Explanation" panel
- training-run insight capture hooks

## What It Cannot Cleanly Do Yet

The first `IAssistantProvider` path exists, but it is intentionally narrow. It
does not yet provide:

- general chat-like assistant session management
- graph/source mutation commands
- multi-turn assistant memory

That is intentional for the first product version.

## Implemented Interface

The current narrow interface is:

```text
IAssistantProvider
```

Minimum responsibilities:

- declare command names
- receive command text
- receive optional structured context
- return formatted assistant output

Example shape:

```cpp
struct AssistantRequest {
    std::string command_name;
    std::string user_text;
    std::string selected_run_id;
    std::string selected_node_id;
    std::string selected_trace_id;
    std::string active_graph_path;
};

struct AssistantResponse {
    bool success = false;
    std::string output_text;
    std::vector<std::string> citations;
    std::string error;
};
```

This should stay read-only.

## Best First Integration Step

The lowest-risk first integration is:

1. plugin adds an **Assistant panel** via `IPanelProvider`
2. panel exposes one text box and one "Ask" button
3. panel calls the existing assistant backend
4. panel renders the structured answer

Why this first:

- no need to modify Command Window routing yet
- plugin system already supports panels
- easy to test with current plugin infrastructure
- no ambiguity between Python REPL behavior and assistant behavior

After that, add a proper Command Window bridge.

## Command Window Path After That

Once the panel works, add a narrow command-window entry point:

- `/ask ...`
- `/explain-trace`
- `/explain-training`

That keeps the syntax explicit and avoids conflicting with normal Python REPL
input.

## Recommendation

Use this order:

1. assistant backend remains separate
2. first CyxWiz integration is an optional plugin
3. first plugin surface is an Assistant panel
4. second surface is Command Window slash-command routing
5. later add debugger-specific actions

## Practical Outcome

If implemented this way, the user story becomes:

1. install plugin
2. load plugin in Plugin Manager
3. open Assistant panel or use `/ask` in Command Window
4. ask CyxWiz-specific questions
5. receive cited answers grounded in local source/docs/graphs/traces

That is a realistic product path using the plugin system that already exists.
