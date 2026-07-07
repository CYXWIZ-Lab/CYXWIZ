# Command Window Integration

## Purpose

Define how the assistant should coexist with the existing CyxWiz Command
Window without breaking current command or Python REPL behavior.

## Current Reality

The current Command Window is a command/console surface with Python REPL-style
behavior.

Current implementation status:

- plain input still follows the existing Python/console path
- `/ask`, `/find-source`, `/explain-trace`, and `/explain-training` are parsed
  before Python execution
- slash commands route through `IAssistantProvider`
- unknown slash commands show a clear assistant help/error message
- `/assistant-help` lists the supported assistant commands

That means this input is ambiguous:

```text
what is cyxwiz
```

It could be interpreted as:

1. a Python expression
2. a built-in command
3. an assistant question

The engine should not guess between those meanings.

## Design Rule

Assistant intent must be explicit.

Do not route plain natural-language input to the assistant automatically.

## Recommended Input Split

Use reserved slash commands for assistant behavior.

### Assistant input

```text
/ask what is cyxwiz
/find-source DebugTraceRecord
/explain-trace
/explain-training
```

### Existing command or REPL input

Everything that does not start with a supported assistant prefix keeps current
Command Window behavior.

Examples:

```text
print("hello")
help
some_existing_command
```

These should stay inside the current console/REPL handling path.

## Routing Rule

The Command Window should dispatch input in this order:

1. if line starts with `/`, attempt assistant slash-command routing
2. if slash command is recognized, send it to the assistant provider
3. if slash command is not recognized, show assistant-specific help/error
4. otherwise, keep existing Command Window behavior unchanged

This is now implemented with:

- `CommandWindowPanel::SetAssistantCommandHandler`
- `PluginManager::RunAssistantCommand`
- `IAssistantProvider::RunAssistantCommand`

## Why This Split Is Correct

This gives:

- no guessing
- no silent behavior changes
- no conflict with Python REPL-style input
- a clear opt-in assistant syntax

It also lets the Command Window remain deterministic for existing users.

## First-Phase UX Recommendation

Do not make the Command Window the first assistant surface.

First:

1. ship an Assistant panel
2. prove the backend and rendering flow
3. then add slash-command routing

Reason:

- panel behavior is unambiguous
- command routing needs an engine-side contract
- the current Command Window already has established semantics

## Supported Assistant Commands

Start with a very small command set:

### `/ask`

General source-aware question.

Examples:

```text
/ask what source file defines DebugTraceRecord
/ask where does TFIDFVectorizer validate min_df
```

### `/find-source`

Retrieve likely source locations.

Examples:

```text
/find-source terminal_reason
/find-source LinearRegressionOperator
```

### `/explain-trace`

Explain the current debugger trace selection.

Example:

```text
/explain-trace
```

This should require an active trace selection.

### `/explain-training`

Explain the current training terminal state.

Example:

```text
/explain-training
```

This should require active training context.

## What Should Happen For Plain Questions

If the user types:

```text
what is cyxwiz
```

the Command Window should not silently send it to the assistant.

Recommended response:

```text
Unknown command. Use /ask for assistant questions.
```

Alternative acceptable response:

```text
Plain-language assistant queries are not enabled here. Try /ask <question>.
```

## Help Text

The Command Window input area or help panel should show a short hint such as:

```text
Enter command or Python input. Use /ask for assistant questions.
```

If slash-command support is enabled, add one compact help block:

```text
Assistant commands:
  /ask <question>
  /find-source <query>
  /explain-trace
  /explain-training
```

## Error Handling

### Unknown slash command

Example:

```text
/foo
```

Response:

```text
Unknown assistant command: /foo
Available commands: /ask, /find-source, /explain-trace, /explain-training
```

### Missing required argument

Example:

```text
/ask
```

Response:

```text
Usage: /ask <question>
```

### Missing required context

Example:

```text
/explain-trace
```

without an active trace selection.

Response:

```text
No active trace selection. Select a trace and try again.
```

### Assistant unavailable

Response:

```text
Assistant unavailable. Check plugin status, knowledge pack, or local runtime.
```

## Response Rendering

Assistant output returned through the Command Window should remain visibly
distinct from normal console output.

Recommended format:

```text
[Assistant]
Answer:
...

Evidence:
...

Unknowns:
...

Unsupported or not implemented:
...
```

This avoids blending model output into raw command or REPL output.

## State Rules

The Command Window should not keep long hidden assistant session state in the
first version.

Each slash command should be treated as a bounded request using:

- current text
- current engine context
- current selection state

This keeps behavior simple and debuggable.

## Minimal Interface Requirement

To support this cleanly, the engine should later expose a narrow assistant
command provider interface, for example:

```cpp
class IAssistantProvider {
public:
    virtual ~IAssistantProvider() = default;
    virtual bool CanHandleCommand(const std::string& command_name) const = 0;
    virtual AssistantResponse Run(const AssistantRequest& request) = 0;
};
```

The Command Window should not call the model/runtime directly.

## Recommendation

Use this behavior:

```text
plain input -> existing Command Window behavior
/...        -> explicit assistant routing
```

That is the cleanest way to let users ask assistant questions in the Command
Window without corrupting existing semantics.
