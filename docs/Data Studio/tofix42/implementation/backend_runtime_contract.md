# Backend Runtime Contract

## Purpose

Define the backend contract between the CyxWiz assistant frontend surfaces and
the assistant runtime.

This keeps the UI/plugin layer stable while retrieval, prompting, and model
runtime details evolve behind it.

## Design Goal

The backend contract should be:

- narrow
- explicit
- read-only
- testable
- transport-agnostic

The panel or Command Window should not care whether the backend uses:

- direct in-process retrieval only
- local HTTP runtime
- proxy-to-chat runtime
- future fine-tuned local runtime

## High-Level Role

The backend owns:

- knowledge-pack loading
- retrieval and ranking
- packet construction
- prompt construction
- runtime invocation
- answer parsing
- citations
- bounded diagnostics

The UI/plugin owns:

- text entry
- context selection
- engine context collection
- response rendering

## Request Contract

Recommended request object:

```cpp
struct AssistantRequest {
    std::string command_name;
    std::string user_text;
    std::string engine_version;
    std::string build_id;
    std::string workspace_root;
    std::string active_graph_path;
    std::string selected_run_id;
    std::string selected_node_id;
    std::string selected_trace_id;
    std::string selected_panel;
    std::string debugger_context_json;
    std::string training_context_json;
    bool retrieval_only = false;
    int top_k = 3;
    int timeout_seconds = 120;
};
```

## Request Rules

- `command_name` is required
- `user_text` may be empty only for context-driven commands such as
  `explain_trace`
- context JSON fields are optional
- `retrieval_only` skips runtime invocation
- `top_k` is bounded by backend policy
- `timeout_seconds` is a hint, not an unlimited user override

## Supported Commands

Recommended first backend commands:

- `ask`
- `find_source`
- `explain_trace`
- `explain_training`

Do not accept arbitrary mutation commands in the first version.

## Response Contract

Recommended response object:

```cpp
struct AssistantCitation {
    std::string path;
    int line_start = 0;
    int line_end = 0;
    std::string title;
    std::string source_type;
};

struct AssistantRetrievalHit {
    int rank = 0;
    double score = 0.0;
    AssistantCitation citation;
    std::string snippet;
};

struct AssistantResponse {
    bool success = false;
    bool retrieval_ok = false;
    bool runtime_ok = false;
    bool parsed = false;
    std::string answer;
    std::string evidence;
    std::string unknowns;
    std::string unsupported_or_not_implemented;
    std::vector<AssistantCitation> citations;
    std::vector<AssistantRetrievalHit> retrieval_hits;
    std::string raw_output;
    std::string error_code;
    std::string error_message;
};
```

## Response Rules

- `success` means the request completed usefully for the current mode
- `retrieval_ok` means retrieval produced usable evidence
- `runtime_ok` means runtime invocation completed
- `parsed` means the four-section answer was extracted
- `retrieval_hits` should still be returned when retrieval succeeds but runtime
  fails
- `raw_output` is diagnostic only

## Success Semantics

### Retrieval-only request

This can be successful when:

- retrieval succeeds
- citations and snippets are returned
- no runtime call is made

### Full answer request

This is successful when:

- retrieval succeeds
- runtime succeeds
- structured answer is available or bounded fallback behavior is returned

## Error Codes

Recommended stable backend error codes:

- `assistant_unavailable`
- `knowledge_pack_missing`
- `knowledge_pack_invalid`
- `knowledge_pack_version_mismatch`
- `retrieval_no_hits`
- `runtime_unavailable`
- `runtime_timeout`
- `runtime_parse_failed`
- `invalid_request`
- `missing_context`

These should be stable across UI surfaces.

## Retrieval Contract

Every successful retrieval should expose:

- ranked hits
- citations
- snippet previews

That allows:

- retrieval-only UI
- trust inspection
- fallback when the model is unavailable

## Runtime Contract

The backend may call:

- local direct runtime
- local HTTP completion endpoint
- OpenAI-compatible proxy endpoint

The frontend should not depend on which one is used.

The backend should normalize runtime output into the same `AssistantResponse`.

## Parse Contract

For full answer requests, the backend should normalize output into:

- `Answer`
- `Evidence`
- `Unknowns`
- `Unsupported or not implemented`

If parsing fails:

- set `parsed = false`
- preserve bounded diagnostics
- do not pretend the answer is trusted

## Transport Options

The contract should work for:

1. in-process backend library call
2. local IPC
3. local HTTP service

Recommended first implementation:

- plugin calls backend in-process if practical, or
- plugin calls a local backend adapter library that may itself use local HTTP

Do not expose external network dependency in the first version.

## Current Implementation Status

The first C++ implementation is retrieval-only:

- [assistant_backend_contract.h](</D:/Dev/CyxWiz_Claude/plugins/assistant/cyxwiz_assistant/assistant_backend_contract.h>)
- [knowledge_pack_backend.h](</D:/Dev/CyxWiz_Claude/plugins/assistant/cyxwiz_assistant/knowledge_pack_backend.h>)
- [knowledge_pack_backend.cpp](</D:/Dev/CyxWiz_Claude/plugins/assistant/cyxwiz_assistant/knowledge_pack_backend.cpp>)

Current behavior:

- loads `docs/Data Studio/tofix42/knowledge_pack`
- reads `manifest.json`, `chunks.jsonl`, and `postings.json`
- searches against the prebuilt postings and chunk store
- returns retrieval hits, snippets, citations, and a retrieval-only answer
- calls `http://127.0.0.1:8768/completion` when retrieval-only mode is off
- allows the panel to override the runtime endpoint with a localhost-only URL
- parses the four required answer sections when runtime output is returned
- falls back to retrieval hits plus a runtime error when the proxy is unavailable

## Versioning

Each backend load should validate:

- supported request schema
- knowledge-pack version policy
- runtime configuration health

Each request should include:

- engine version
- build id

This lets the backend reject stale or mismatched assets.

## Performance Policy

The backend should:

- load knowledge pack once
- reuse retrieval structures across requests
- bound request timeouts
- avoid rebuilding indexes at query time

The backend should not:

- rescan the repo every request
- spawn uncontrolled runtime processes per query

## Logging and Diagnostics

The backend should expose minimal diagnostics for UI and tests:

- retrieval count
- selected citations
- runtime status
- parse status
- stable error code

This supports:

- panel status messages
- regression tests
- local debugging

## Example Request

```json
{
  "command_name": "ask",
  "user_text": "What source file defines DebugTraceRecord?",
  "engine_version": "0.9.0",
  "build_id": "local-dev",
  "retrieval_only": false,
  "top_k": 3,
  "timeout_seconds": 120
}
```

## Example Response

```json
{
  "success": true,
  "retrieval_ok": true,
  "runtime_ok": true,
  "parsed": true,
  "answer": "cyxwiz-engine/src/core/debug_trace_record.h",
  "evidence": "[E1] cyxwiz-engine/src/core/debug_trace_record.h:31-46",
  "unknowns": "none",
  "unsupported_or_not_implemented": "none",
  "citations": [
    {
      "path": "cyxwiz-engine/src/core/debug_trace_record.h",
      "line_start": 31,
      "line_end": 46,
      "title": "DebugTraceRecord",
      "source_type": "source"
    }
  ],
  "error_code": "",
  "error_message": ""
}
```

## Recommendation

Use one stable backend contract for all assistant surfaces:

```text
frontend sends AssistantRequest
backend returns AssistantResponse
```

Let retrieval/runtime implementation details vary behind that contract. That is
the cleanest way to move from the current tofix42 harness to a real CyxWiz
assistant backend.
