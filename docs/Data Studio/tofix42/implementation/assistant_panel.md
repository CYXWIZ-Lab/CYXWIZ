# Assistant Panel

## Purpose

Define the first real CyxWiz UI surface for the assistant.

This panel is the recommended first integration point because it avoids
ambiguity with the current Command Window and fits the existing plugin panel
model.

## Why Start With a Panel

The Assistant panel should come before Command Window routing because:

- `IPanelProvider` already exists
- panel behavior is unambiguous
- assistant questions can be plain language here
- structured output is easier to render in a dedicated panel
- failure states are easier to communicate without interfering with existing
  console behavior

## User Goal

The panel gives the user a read-only, source-aware explanation surface inside
CyxWiz.

Typical user goals:

- ask what a type, field, operator, or graph element does
- locate source files
- explain selected debugger traces
- explain training terminal reasons

## First-Version Scope

The first panel version should support only:

1. plain text question entry
2. optional context mode selection
3. structured response rendering
4. citation display
5. clear assistant availability status

Do not start with:

- graph mutation
- source editing
- training control
- chat memory
- long-lived hidden session state

## Layout

Recommended panel layout:

```text
+------------------------------------------------------+
| Assistant                                            |
|------------------------------------------------------|
| Context: [General | Trace | Training]   Status: OK   |
|------------------------------------------------------|
| Ask CyxWiz...                                        |
| [ text input                                       ] |
| [ Ask ] [ Retrieval Only ] [ Show Citations ]        |
|------------------------------------------------------|
| Answer                                               |
| ...                                                  |
|                                                      |
| Evidence                                             |
| ...                                                  |
|                                                      |
| Unknowns                                             |
| ...                                                  |
|                                                      |
| Unsupported or not implemented                       |
| ...                                                  |
|------------------------------------------------------|
| Citations                                            |
| - path:line-line                                     |
+------------------------------------------------------+
```

## Input Model

The panel should accept plain language because it is a dedicated assistant
surface.

Examples:

```text
What source file defines DebugTraceRecord?
Where does TFIDFVectorizer validate min_df?
Why did this training run stop?
Explain the selected trace.
```

No slash prefix is needed in the panel.

## Context Modes

The first panel should support a small explicit context selector:

1. `General`
   - no special engine context required
2. `Trace`
   - uses selected debugger trace context if available
3. `Training`
   - uses active training summary/terminal context if available

This keeps the request path visible to the user.

## Panel-to-Backend Behavior

The panel should:

1. collect user text
2. collect current engine/build metadata
3. collect optional selected trace or training context
4. build `AssistantRequest`
5. send it to the backend
6. render structured output

The panel should not talk directly to the model runtime.

Current implementation note:

- the panel stores the `PluginContext` passed by the engine lifecycle
- each request copies the latest `AssistantContextSnapshot` into
  `AssistantRequest`
- the status area shows current project root, graph path, and selected node id
- Trace mode sends the current bounded debugger context bundle
- Training mode sends the current bounded training context, including terminal
  reason when available
- context action buttons are available for selected trace explanation, training
  stop reason explanation, and selected node source lookup
- missing trace, training terminal, or selected-node context returns a clear
  panel error instead of sending an ambiguous backend request

## Ask Flow

### General question

```text
User enters question
  -> panel builds AssistantRequest(command_name="ask")
  -> backend retrieves evidence
  -> backend returns structured response
  -> panel renders answer and citations
```

### Trace explanation

```text
User selects Trace mode
  -> panel gathers selected trace context
  -> panel builds AssistantRequest(command_name="explain_trace")
  -> backend retrieves source/docs related to context
  -> panel renders explanation and citations
```

### Training explanation

```text
User selects Training mode
  -> panel gathers training terminal context
  -> panel builds AssistantRequest(command_name="explain_training")
  -> backend responds with cited explanation
```

## Rendering Rules

Render assistant output in the same four sections already proven by the
tofix42 harness:

- `Answer`
- `Evidence`
- `Unknowns`
- `Unsupported or not implemented`

This should remain stable across panel and future Command Window integrations.

## Citation Rendering

Each citation should show:

- path
- line range
- title
- source type

Recommended compact format:

```text
cyxwiz-engine/src/core/debug_trace_record.h:31-46
DebugTraceRecord (source)
```

If clickable file navigation exists later, citations can open the file view.

## Status States

The panel should show a small explicit status indicator:

- `Ready`
- `Retrieval Only`
- `Runtime Unavailable`
- `Knowledge Pack Missing`
- `Version Mismatch`

This reduces confusion when the assistant backend is partially available.

## Failure Behavior

### No backend

Show:

```text
Assistant unavailable.
```

### Knowledge pack missing

Show:

```text
Knowledge pack missing or invalid.
```

### Runtime failure

Show:

```text
Model runtime unavailable. Retrieval results may still be shown.
```

### Missing context

If user selects `Trace` without a selected trace:

```text
No active trace selection.
```

If user selects `Training` without training context:

```text
No active training context.
```

## Retrieval-Only Mode

The panel should support a retrieval-only toggle in the first version.

Why:

- useful when runtime is down
- useful for debugging retrieval quality
- useful for trust-building

In this mode the panel should show:

- top citations
- retrieved snippets
- no model answer requirement

## Session Rules

The first version should be stateless per request.

Each request should depend only on:

- current text
- current panel mode
- current engine context
- current backend state

Do not add hidden conversation memory in the first product version.

## Minimal Panel API

The plugin panel can use a backend interface like:

```cpp
class IAssistantBackend {
public:
    virtual ~IAssistantBackend() = default;
    virtual AssistantResponse Run(const AssistantRequest& request) = 0;
};
```

The panel owns UI state. The backend owns retrieval/runtime work.

## Recommendation

Use this first UI shape:

```text
Assistant panel
  -> plain-language input
  -> explicit context mode
  -> structured four-section output
  -> citations
  -> clear availability state
```

That is the cleanest first product surface for the CyxWiz assistant.
