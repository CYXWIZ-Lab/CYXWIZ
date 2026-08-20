# CyxWiz Assistant Plugin

Read-only source-aware assistant provider for CyxWiz Console.

This plugin currently provides:

- an `IAssistantProvider` consumed by `Console > Agent LLM`
- an in-code `AssistantRequest` / `AssistantResponse` backend contract
- a retrieval-only C++ backend that loads the local knowledge pack
- typed answer sections, retrieval hits, snippets, and citations
- local runtime calls through `http://127.0.0.1:8768/completion` when
  retrieval-only mode is off
- General, selected-trace, and training-terminal context requests

The former standalone `CyxWiz Assistant` panel is retired. The plugin owns
retrieval and runtime integration; Console owns the user experience.

## Build

Configure with the assistant plugin enabled:

```powershell
cmake -S . -B build-recovery -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=ON
cmake --build build-recovery --config Release --target cyxwiz_assistant
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

## Use

1. Open a CyxWiz project.
2. Open `Console`.
3. Select `+`, then `Agent LLM`.
4. Use `Retrieval only` for source lookup, or leave it disabled for full local
   RAG synthesis.

See `internal/repository-private/engineering/Data Studio/tofix42/engine_rag_quickstart.md`
for the accepted model, proxy, plugin-install, test, and troubleshooting flow.
