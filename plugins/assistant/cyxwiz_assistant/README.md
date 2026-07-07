# CyxWiz Assistant Plugin

Experimental read-only assistant panel scaffold for CyxWiz.

This plugin currently provides:

- a `ProvidesPanels` plugin manifest
- a `CyxWiz Assistant` panel through `IPanelProvider`
- an in-code `AssistantRequest` / `AssistantResponse` backend contract
- a retrieval-only C++ backend that loads the local knowledge pack
- retrieval hits, snippets, and citations in the panel response
- local runtime calls through `http://127.0.0.1:8768/completion` when
  retrieval-only mode is off
- a panel knowledge-pack path field with `Reload Pack`
- a panel runtime endpoint field for local proxy testing

It does not yet provide:

- command-window slash commands
- debugger or training context collection

## Build

Configure with the assistant plugin enabled:

```powershell
cmake -S . -B build -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=ON
cmake --build build --config Debug --target cyxwiz_assistant
```

The plugin binary is written to:

```text
plugins/assistant/cyxwiz_assistant/bin/
```

## Load

Copy or point the CyxWiz Plugin Manager at:

```text
plugins/assistant/cyxwiz_assistant/
```

The folder must contain:

- `plugin.json`
- `bin/cyxwiz_assistant.dll` on Windows

## Current Panel Controls

- `Question`
- `Context`
- `Retrieval only`
- `Show citations`
- `Top K`
- `Timeout`
- `Knowledge pack`
- `Reload Pack`
- `Runtime endpoint`

## Next Implementation Step

Move from manual local testing to engine-side context wiring:

- debugger trace context
- training terminal context
- selected graph/node context
