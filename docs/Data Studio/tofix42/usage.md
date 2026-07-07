# tofix42 Usage Notes
      - /ask <question>
      - /find-source <query>
      - /explain-trace [question]
      - /explain-training [question]
      - /assistant-help

## Purpose

Keep the current hands-on usage steps for the local CyxWiz source-aware
assistant work. Update this file as each phase adds a new command, workflow, or
Studio entry point.

## Current Phase

Current active slice: read-only local assistant harness through Phase 7 is
complete for deterministic checks. Phase 2 now has an accepted real localhost
model validation path through `llama-server` plus the OpenAI-compatible proxy
using a stronger local GGUF model.

Next gate: write a fine-tuning experiment plan with dataset provenance and
rollback criteria while keeping retrieval mandatory.

Current experiment plan:

- [phase7_finetune_experiment_plan.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase7_finetune_experiment_plan.md>)
- [implementation/README.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/README.md>)
- [implementation/deployment_model.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/deployment_model.md>)
- [implementation/fine_tuning_position.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/fine_tuning_position.md>)
- [rag_question_guide.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/rag_question_guide.md>)
- [phase8_knowledge_pack.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase8_knowledge_pack.md>)

Status:

- local text indexing works
- lexical search works
- deterministic retrieval gate works
- retrieval-only answer packets work
- index freshness checks work
- strict evidence prompt generation works
- local runner adapter exists
- successful runner output is normalized into answer sections
- local runner errors produce structured diagnostics
- `llama-cli` stdout prompt echoes are stripped with an output sentinel
- runtime-unavailable answer envelope works
- Phase 1B handoff exists
- Phase 2 runtime boundary is documented
- optional local JSON HTTP runtime adapter exists
- Phase 2 JSON runtime has a localhost fixture check
- JSON runtime endpoints are restricted to localhost
- Phase 2 handoff exists
- local JSON runtime stub exists for manual end-to-end adapter checks
- local JSON runtime stub end-to-end check passed
- `probe` command validates local JSON runtime answer-section compliance
- `probe` now fails when any expected answer section is missing
- Phase 2 real-model validation checklist exists
- Phase 2 probe suite exists for localhost JSON runtime validation
- Phase 2 endpoint doctor exists for reachability and stub detection
- Phase 2 local endpoint scan exists for common localhost ports
- Phase 2 OpenAI-compatible proxy exists for local `/v1/chat/completions`
  servers
- Phase 2 real-model check orchestrates doctor, probe suite, report validation,
  optional bad-answer capture, and Phase 7 summary
- Phase 2 real-model check reports `real_model_probe_accepted` separately from
  `fine_tuning_ready`
- manual `llama-cli` validation failed; use localhost JSON runtime for Phase 2
- `llama-server` was cloned and built in `D:\tmp\llama.cpp`
- tiny `smollm-135m.Q4_K_M.gguf` remains insufficient for accepted Phase 2
  validation
- downloaded real validation model:
  `D:\tmp\models\qwen2.5-coder-3b-instruct-q4_k_m.gguf`
- accepted real local upstream:
  `http://127.0.0.1:1235/v1/chat/completions`
- accepted patched proxy endpoint:
  `http://127.0.0.1:8768/completion`
- accepted full real-model report:
  `$env:TEMP\tofix42_phase2_real_model_check_qwen3b_8768.json`
- `real_model_probe_accepted` is now `true`
- reviewed bad-answer records: `20`
- reviewed correction export:
  `docs/Data Studio/tofix42/phase6_reviewed_corrections.jsonl`
- Phase 7 decision now approves a fine-tuning experiment
- Phase 7 dataset prep script exists and generates:
  `docs/Data Studio/tofix42/phase7_dataset/train.chat.jsonl`
  `docs/Data Studio/tofix42/phase7_dataset/validation.chat.jsonl`
  `docs/Data Studio/tofix42/phase7_dataset/manifest.json`
- Phase 7 dataset exporter also generates:
  `docs/Data Studio/tofix42/phase7_dataset/train.messages.jsonl`
  `docs/Data Studio/tofix42/phase7_dataset/validation.messages.jsonl`
  `docs/Data Studio/tofix42/phase7_dataset/train.instruction.jsonl`
  `docs/Data Studio/tofix42/phase7_dataset/validation.instruction.jsonl`
- Phase 3 debugger context packet harness exists
- Phase 3 debugger context can emit a retrieval-backed Phase 1B-compatible
  answer packet
- Phase 3 deterministic trace explanation exists for model-free validation
- Phase 4 training trace context/explanation harness exists
- Phase 5 graph selected-node/path/suggestion/audit/draft-plan harness exists
- Phase 6 deterministic evaluation/QA capture harness exists
- Phase 7 fine-tuning readiness decision harness exists
- consolidated deterministic check-all harness exists
- consolidated current-status summary exists
- current deterministic gates pass: 14 passed, 0 failed
- current index checkpoint: 52 files, 603 chunks, fresh
- no embeddings yet
- no Studio UI yet
- no graph/source mutation
- no fine-tuning

## Phase 8 Knowledge Pack Commands

Build the default knowledge pack:

```powershell
python "docs/Data Studio/tofix42/phase8_knowledge_pack.py" build `
  --engine-version dev `
  --build-id local
```

Validate the pack:

```powershell
python "docs/Data Studio/tofix42/phase8_knowledge_pack.py" validate
```

Run the retrieval gate against the pack:

```powershell
python "docs/Data Studio/tofix42/phase8_knowledge_pack.py" check
```

Search without rescanning the repo:

```powershell
python "docs/Data Studio/tofix42/phase8_knowledge_pack.py" search `
  "What source file defines DebugTraceRecord" `
  --top 3
```

Build a retrieval-only packet from the pack:

```powershell
python "docs/Data Studio/tofix42/phase8_knowledge_pack.py" packet `
  "TrainingTraceEvent terminal_reason field" `
  --top 1 `
  --json
```

## Assistant Plugin Commands

Configure with the experimental assistant plugin enabled:

```powershell
cmake -S . -B build -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=ON
```

Build the plugin:

```powershell
cmake --build build --config Debug --target cyxwiz_assistant
```

Current plugin binary:

```text
plugins/assistant/cyxwiz_assistant/bin/cyxwiz_assistant.dll
```

Current plugin behavior:

- loads as a `ProvidesPanels` plugin
- exposes a `CyxWiz Assistant` panel
- loads the local knowledge pack from the panel `Knowledge pack` field
- supports `Reload Pack`
- returns retrieval citations/snippets
- when retrieval-only is off, calls the panel runtime endpoint
- default runtime endpoint: `http://127.0.0.1:8768/completion`
- runtime endpoint is restricted to `http://localhost/...` or
  `http://127.0.0.1/...`
- parses the four required answer sections from runtime output
- falls back to retrieval hits if the runtime proxy is unavailable

For full answer mode, the accepted local runtime path must already be running:

```powershell
python "docs/Data Studio/tofix42/phase2_openai_compat_proxy.py" serve `
  --upstream "http://127.0.0.1:1235/v1/chat/completions" `
  --model "local-model" `
  --port 8768
```

## How To Use In CyxWiz Studio

### 1. Load the plugin

1. Open **Plugin Manager**.
2. Scan and load the assistant plugin folder:

```text
D:\Dev\CyxWiz_Claude\plugins\assistant\cyxwiz_assistant
```

3. Confirm `CyxWiz Assistant` is active.

### 2. Use the Assistant panel

Open the `CyxWiz Assistant` panel and enter a question such as:

```text
What source file defines DebugTraceRecord?
Where does TFIDFVectorizer validate min_df?
```

Useful panel controls:

- `Retrieval only` for source-only answers
- `Show citations` to show the matching snippets and line ranges
- `Context` to switch between `General`, `Trace`, and `Training`

### 3. Use the Command Window

The Command Window still behaves like the existing Python/console surface.

Use assistant commands only with a slash prefix:

```text
/assistant-help
/ask What source file defines DebugTraceRecord
/find-source terminal_reason
/explain-trace
/explain-training
```

Rules:

- plain text without `/` still goes to Python/console handling
- unknown slash commands show an assistant error instead of running Python
- `/ask` is for general source questions
- `/find-source` is for looking up source locations
- `/explain-trace` uses the current Studio Debugger trace selection
- `/explain-training` uses the current training terminal context

### 4. Use graph context

When a graph node is selected in the Node Editor, the assistant can use that
selection as context.

Examples:

```text
/find-source LinearRegressionOperator
/ask what does the selected node do
```

The plugin reads the current graph snapshot and selected node id from the
engine, so the question should be about the active graph or the selected node.

### 5. Use Studio Debugger context

Open **Studio Debugger** and select a trace, then use:

```text
/explain-trace
```

For training runs, use:

```text
/explain-training
```

These actions only work when the corresponding context exists. If no trace or
training terminal context is available, the plugin shows a clear missing-
context message instead of guessing.

## Troubleshooting

### Plugin does not appear

- confirm the plugin folder is `D:\Dev\CyxWiz_Claude\plugins\assistant\cyxwiz_assistant`
- confirm `plugin.json` exists in that folder
- confirm `bin\cyxwiz_assistant.dll` exists
- reload the plugin from Plugin Manager after rebuilding

### Command Window sends `/assistant-help` to Python

- make sure the Command Window is the active input surface
- type the command exactly as `/assistant-help`
- verify the plugin is loaded and active
- restart the Release engine if the Command Window was open before the plugin loaded

### Assistant returns missing context

- select a node in the graph before using `/find-source`
- select a trace in Studio Debugger before using `/explain-trace`
- make sure a training terminal event exists before using `/explain-training`
- use the Assistant panel `Context` selector when you want a general question instead of a trace or training action

### Runtime unavailable

- retrieval-only mode still works without the local runtime
- full answer mode requires the localhost proxy at `http://127.0.0.1:8768/completion`
- the proxy must forward to a real local model server

## Phase 1A Commands

Run from the repository root.

Build or rebuild the local index:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" build
```

Search the local index:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "DebugTraceRecord"
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "TrainingTraceEvent terminal_reason field"
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "TFIDFVectorizer sentiment graph"
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "DataLoader pin_memory unsupported current batchers compatibility"
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "TFIDFVectorizer min_df must be >= 1 Configure"
```

Run the Phase 1A success gate:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" check
```

Check whether the index is fresh:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" status
```

The `status` command checks both indexed file hashes and the config hash.

Build a retrieval-only answer packet:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet "What source file defines DebugTraceRecord"
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet "RecordTerminalEvent terminal_reason message" --json
```

Inspect the expanded search report:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "TrainingTraceEvent terminal_reason field"
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "DataLoader pin_memory unsupported current batchers compatibility" --top 8 --json
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "DebugTraceRecord" --top 3 --include-full-text
```

Narrow broad searches with metadata filters:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "terminal_reason" --source-type source --path-contains training
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "sentiment" --source-type cyxgraph --path-contains examples/cyxgraph --json
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet "terminal_reason" --source-type source --path-contains training --top 2 --json
```

List and use saved presets:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" presets
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search --preset training_terminal --top 3
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet --preset sentiment_graphs --top 2 --json
```

## Phase 1B Commands

Create a packet file:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet "DataLoader pin_memory unsupported current batchers compatibility" --top 1 --json > "D:/tmp/tofix42_packet.json"
```

Build a strict evidence prompt:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" prompt --packet "D:/tmp/tofix42_packet.json"
```

Return a clean runtime-unavailable envelope:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" answer --packet "D:/tmp/tofix42_packet.json" --runtime none --json
```

Pipe a packet directly into the answer adapter:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet "DataLoader pin_memory unsupported current batchers compatibility" --top 1 --json |
  python "docs/Data Studio/tofix42/phase1b_answer.py" answer --runtime none --json
```

Check runtime readiness:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" doctor --runtime none --json
python "docs/Data Studio/tofix42/phase1b_answer.py" doctor --runtime llama-cli --runner "llama-cli" --model "smollm-135m.Q4_K_M.gguf" --json
```

`doctor` checks availability only. It does not load the model or prove inference
will complete.

Run the Phase 1B adapter gate:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" check --packet "D:/tmp/tofix42_packet.json"
```

Read the Phase 1B handoff before starting the next runtime slice:

```powershell
Get-Content "docs/Data Studio/tofix42/phase1b_handoff.md"
```

Read the Phase 2 runtime boundary:

```powershell
Get-Content "docs/Data Studio/tofix42/phase2_runtime.md"
```

Read the Phase 2 handoff:

```powershell
Get-Content "docs/Data Studio/tofix42/phase2_handoff.md"
```

Read the Phase 2 real-model validation checklist:

```powershell
Get-Content "docs/Data Studio/tofix42/phase2_model_validation.md"
```

Check a local JSON runtime endpoint configuration without probing it:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" doctor `
  --packet "D:/tmp/tofix42_packet.json" `
  --runtime json-http `
  --endpoint "http://127.0.0.1:8080/completion" `
  --json
```

Call a running local JSON runtime endpoint:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" answer `
  --packet "D:/tmp/tofix42_packet.json" `
  --runtime json-http `
  --endpoint "http://127.0.0.1:8080/completion" `
  --max-tokens 384 `
  --timeout-seconds 120 `
  --json
```

The adapter does not start the server or download a model. It only accepts
`localhost`, `127.0.0.1`, or `::1` endpoints that are already running.

Start the local JSON runtime stub manually:

```powershell
python "docs/Data Studio/tofix42/phase2_stub_runtime.py" --port 8765
```

Call the stub from a second shell:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" answer `
  --packet "D:/tmp/tofix42_packet.json" `
  --runtime json-http `
  --endpoint "http://127.0.0.1:8765/completion" `
  --max-tokens 384 `
  --timeout-seconds 30 `
  --json
```

Probe the stub or a real local JSON runtime for answer-section compliance:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" probe `
  --packet "D:/tmp/tofix42_packet.json" `
  --endpoint "http://127.0.0.1:8765/completion" `
  --max-tokens 384 `
  --timeout-seconds 30 `
  --json
```

Run all Phase 2 model-validation probe cases:

```powershell
python "docs/Data Studio/tofix42/phase2_probe_suite.py" `
  --endpoint "http://127.0.0.1:8765/completion" `
  --output "$env:TEMP/tofix42_phase2_probe_suite.json" `
  --json
```

Run the full real-model validation sequence:

```powershell
python "docs/Data Studio/tofix42/phase2_real_model_check.py" run `
  --endpoint "http://127.0.0.1:8765/completion" `
  --output "$env:TEMP/tofix42_phase2_probe_suite.json" `
  --capture-failures
```

List or run one Phase 2 probe case:

```powershell
python "docs/Data Studio/tofix42/phase2_probe_suite.py" --list-cases
python "docs/Data Studio/tofix42/phase2_probe_suite.py" `
  --endpoint "http://127.0.0.1:8765/completion" `
  --case "dataloader_pin_memory_truth" `
  --json
```

Capture a failing Phase 2 probe case for review:

```powershell
python "docs/Data Studio/tofix42/phase2_probe_suite.py" `
  --endpoint "http://127.0.0.1:8765/completion" `
  --case "dataloader_pin_memory_truth" `
  --include-raw-output `
  --include-prompt `
  --output "$env:TEMP/tofix42_bad_answer_capture.json" `
  --json
```

Call an explicit local runner if available:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" answer `
  --packet "D:/tmp/tofix42_packet.json" `
  --runtime llama-cli `
  --runner "llama-cli" `
  --model "smollm-135m.Q4_K_M.gguf" `
  --max-tokens 32 `
  --timeout-seconds 120 `
  --json
```

Known `llama-cli` stdout issue:

Some `llama-cli` builds enter the chat-style UI path and write the banner,
command help, and prompt echo to stdout. Redirecting stderr does not clean this
up because stderr may be empty. Do not assume stdout is pure model text.

For direct `llama-cli` smoke checks, only use a separate known-good
`llama-cli.exe`; the temporary `D:\tmp\llama.cpp` clone was removed after the
Phase 2 attempt failed. Suppress logs and close stdin:

```powershell
$input = ""
$input | & "<path-to-llama-cli.exe>" `
  -m "smollm-135m.Q4_K_M.gguf" `
  -p "Answer: hello" `
  -n 8 `
  --no-conversation `
  --no-display-prompt `
  --simple-io `
  --no-warmup `
  --log-disable
```

Parser rule: use a unique output sentinel in the prompt and discard everything
before the final sentinel. If this path becomes more than a smoke test, prefer
`llama-server` JSON responses over parsing `llama-cli` stdout.

The Phase 1B adapter now applies this rule internally for `--runtime llama-cli`.

Manual Phase 2 validation with `llama-cli` failed: the subprocess hung and was
interrupted before returning a usable answer. Do not use `llama-cli` as the
Phase 2 validation path unless its process behavior is fixed outside the
adapter.

With `--json`, failed local inference returns a `runtime_error` envelope with the
runner command, exit code when available, timeout flag, and bounded stdout/stderr
excerpts. The adapter exits with code `2` on failed local inference.

Successful local inference returns `raw_model_output` plus
`model_output_sections` parsed from `Answer:`, `Evidence:`, `Unknowns:`, and
`Unsupported or not implemented:` headings.

Validate a saved Phase 2 probe-suite report without calling the model again:

```powershell
python "docs/Data Studio/tofix42/phase2_probe_report_check.py" validate `
  --report "$env:TEMP/tofix42_phase2_probe_suite.json"
```

Stub reports are flagged as `stub_runtime_detected`; they validate the JSON
pipeline but do not satisfy the Phase 7 real-model criterion.

Check whether a local endpoint is reachable and whether it is the stub:

```powershell
python "docs/Data Studio/tofix42/phase2_endpoint_doctor.py" check `
  --endpoint "http://127.0.0.1:8765/completion"
```

Scan common local endpoint locations:

```powershell
python "docs/Data Studio/tofix42/phase2_local_endpoint_scan.py" scan
```

Build and start a temporary `llama-server` test runtime:

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git D:\tmp\llama.cpp
cd D:\tmp\llama.cpp
cmake -S . -B build -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release --target llama-server -j 8
```

Start the original tiny-model server from a separate PowerShell window:

```powershell
& "D:\tmp\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "D:\Dev\CyxWiz_Claude\smollm-135m.Q4_K_M.gguf" `
  --host "127.0.0.1" `
  --port 1234 `
  --ctx-size 2048 `
  --threads 4
```

Start the accepted real-model server:

```powershell
& "D:\tmp\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "D:\tmp\models\qwen2.5-coder-3b-instruct-q4_k_m.gguf" `
  --host "127.0.0.1" `
  --port 1235 `
  --ctx-size 4096 `
  --threads 4
```

Direct `llama-server` smoke check:

```powershell
$body = @{
  model = "local-model"
  messages = @(@{ role = "user"; content = "Reply with exactly: hello" })
  max_tokens = 16
  temperature = 0
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:1234/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Bridge a local OpenAI-compatible chat endpoint to the tofix42 `/completion`
contract:

```powershell
python "docs/Data Studio/tofix42/phase2_openai_compat_proxy.py" serve `
  --upstream "http://127.0.0.1:1235/v1/chat/completions" `
  --model "local-model" `
  --port 8768
```

Then probe through the proxy from another PowerShell window:

```powershell
python "docs/Data Studio/tofix42/phase2_probe_suite.py" `
  --endpoint "http://127.0.0.1:8768/completion" `
  --output "$env:TEMP/tofix42_phase2_probe_suite.json" `
  --timeout-seconds 120 `
  --json
```

Or run the full validation sequence through the proxy:

```powershell
python "docs/Data Studio/tofix42/phase2_real_model_check.py" run `
  --endpoint "http://127.0.0.1:8768/completion" `
  --output "$env:TEMP\tofix42_phase2_real_model_check_qwen3b_8768.json" `
  --timeout-seconds 120 `
  --capture-failures
```

Current `llama-server` proof result:

- direct OpenAI-compatible endpoint: reachable at `127.0.0.1:1235`
- patched tofix42 proxy endpoint: reachable at `127.0.0.1:8768`
- full Phase 2 validation: passes all five probe cases with
  `--timeout-seconds 120`
- readiness result: `real_model_probe_accepted = true`
- Phase 7 still blocks fine-tuning until reviewed corrections exist
- report path: `$env:TEMP\tofix42_phase2_real_model_check_qwen3b_8768.json`

## End-to-End Proof Example

The cleanest proof case is `debug_trace_record_definition`.

Question:

```text
What source file defines DebugTraceRecord
```

Step 1: build the retrieval packet:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet `
  "What source file defines DebugTraceRecord" `
  --top 1 `
  --json
```

Expected top citation:

```text
cyxwiz-engine/src/core/debug_trace_record.h:31-46
```

Step 2: build the exact Phase 1B prompt from that packet:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet `
  "What source file defines DebugTraceRecord" `
  --top 1 `
  --json > "$env:TEMP\tofix42_probe_packet.json"

python "docs/Data Studio/tofix42/phase1b_answer.py" prompt `
  --packet "$env:TEMP\tofix42_probe_packet.json"
```

Step 3: run the full real-model validation through the accepted proxy:

```powershell
python "docs/Data Studio/tofix42/phase2_real_model_check.py" run `
  --endpoint "http://127.0.0.1:8768/completion" `
  --output "$env:TEMP\tofix42_phase2_real_model_check_qwen3b_8768.json" `
  --timeout-seconds 120 `
  --capture-failures
```

Accepted result for this probe from the saved report:

```text
name: debug_trace_record_definition
runtime_ok: true
parsed: true
expected_path_in_output: true
answer: cyxwiz-engine/src/core/debug_trace_record.h
evidence: [E1]
unknowns: none
unsupported_or_not_implemented: none
ok: true
```

Why this is proof:

- retrieval found the correct local file
- the prompt included the exact cited snippet
- the real localhost model answered through the proxy
- the parser recovered all four required sections
- the expected path appeared in the answer output

This proves the system can answer from retrieved local evidence rather than
model memory.

## Question Quality

The current RAG works best for:

- file-definition questions
- field and validation questions
- explicit operator-behavior questions
- example-graph questions
- trace/training explanations tied to current context

Examples:

```text
What source file defines DebugTraceRecord?
TrainingTraceEvent terminal_reason field
Where does TFIDFVectorizer validate min_df?
What example graph uses TFIDFVectorizer?
```

The current RAG is weak for:

- broad product-help questions
- open-ended tutorials
- vague capability prompts

Examples:

```text
What can u assist me with in CyxWiz engine?
How do I build everything for linear regression?
What is CyxWiz?
```

Why:

- retrieval may find a structurally valid but semantically wrong source chunk
- the model can then answer from the wrong evidence

Use the current system as a codebase evidence assistant, not a general help
bot.

For a full guide, read:

- [rag_question_guide.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/rag_question_guide.md>)

## How to Prompt

Use the system in this order:

1. run retrieval for the question
2. build a packet
3. build the strict Phase 1B prompt or send the packet to the adapter
4. send the prompt to the accepted localhost endpoint

Good retrieval questions are short and specific:

```text
What source file defines DebugTraceRecord
TrainingTraceEvent terminal_reason field
DataLoader pin_memory unsupported current batchers compatibility
TFIDFVectorizer min_df must be >= 1 Configure
```

Build a prompt for inspection:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" packet `
  "TrainingTraceEvent terminal_reason field" `
  --top 1 `
  --json > "$env:TEMP\tofix42_packet.json"

python "docs/Data Studio/tofix42/phase1b_answer.py" prompt `
  --packet "$env:TEMP\tofix42_packet.json"
```

Ask the live local runtime through the accepted proxy:

```powershell
python "docs/Data Studio/tofix42/phase1b_answer.py" probe `
  --packet "$env:TEMP\tofix42_packet.json" `
  --endpoint "http://127.0.0.1:8768/completion" `
  --timeout-seconds 120 `
  --json
```

Use the helper script to run one custom test with one command:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "docs/Data Studio/tofix42/run_my_test.ps1" `
  "TrainingTraceEvent terminal_reason field"
```

Run the helper with retrieval filters:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "docs/Data Studio/tofix42/run_my_test.ps1" `
  "terminal_reason" `
  -SourceType source `
  -PathContains training
```

Run the helper from a preset:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "docs/Data Studio/tofix42/run_my_test.ps1" `
  -Preset training_terminal `
  -RetrievalOnly
```

The helper now prints a classification before the raw probe JSON:

- `retrieval_pass`
  - retrieval returned evidence, and if a live probe ran, the model returned all
    required sections
- `format_fail`
  - the live runtime answered, but the required structured sections were missing
    or not parsed
- `runtime_fail`
  - the endpoint failed, timed out, or returned an execution error
- `answer_suspect`
  - retrieval returned no evidence, or the probe completed without satisfying
    the success checks

When the helper detects procedural or command-like content that is not grounded
in the retrieved evidence, it also prints:

- `Suspicion: ...`
  - for example, when a "how to" answer is generated from raw implementation
    source instead of explicit usage documentation

Preview the exact prompt before probing:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "docs/Data Studio/tofix42/run_my_test.ps1" `
  "What source file defines DebugTraceRecord" `
  -ShowPrompt
```

Run retrieval only without calling the model:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "docs/Data Studio/tofix42/run_my_test.ps1" `
  "TFIDFVectorizer min_df must be >= 1 Configure" `
  -RetrievalOnly
```

Prompting rules that matter:

- ask for facts that should exist in the indexed files
- keep the question aligned to code, docs, or graph artifacts
- rely on the evidence packet, not on free-form model memory
- expect the answer in this structure:
  - `Answer`
  - `Evidence`
  - `Unknowns`
  - `Unsupported or not implemented`

If the output misses that structure, the failure is in answer formatting or
runtime behavior, not retrieval.

Limit result count:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" search "DataLoader pin_memory" --top 10
```

Use a different index path:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" --index "D:/tmp/tofix42_index.json" build
python "docs/Data Studio/tofix42/phase1a_retrieval.py" --index "D:/tmp/tofix42_index.json" search "DebugRunStore"
```

Use a different config path:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" --config "D:/tmp/tofix42_config.json" build
python "docs/Data Studio/tofix42/phase1a_retrieval.py" --config "D:/tmp/tofix42_config.json" check
```

## Current Indexed Inputs

Configured in `docs/Data Studio/tofix42/phase1a_config.json`.

- `docs/Data Studio/*.md`
- `docs/usage/*.md`
- `examples/cyxgraph/**/*.md`
- `examples/cyxgraph/**/*.cyxgraph`
- `cyxwiz-engine/src/core/**/*.h`
- `cyxwiz-engine/src/core/**/*.hpp`
- `cyxwiz-engine/src/core/**/*.cpp`
- `cyxwiz-engine/src/core/**/*.cc`
- `cyxwiz-engine/src/core/**/*.cxx`

Current rebuilt index footprint:

- `432` files
- `4498` chunks

## Phase 3 Debugger Context Commands

Build a selected-trace context packet from an existing debug run:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" context `
  --run ".cyxwiz/debug_runs/studio/<run-id>/session.json" `
  --pretty
```

Select a specific node/role:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" context `
  --run ".cyxwiz/debug_runs/studio/<run-id>/session.json" `
  --node-id 5 `
  --role "PreprocessingOutput" `
  --pretty
```

Run the deterministic Phase 3 context gate:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" check
```

Build a retrieval-backed Phase 1B-compatible packet for the selected trace:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" packet `
  --run ".cyxwiz/debug_runs/studio/<run-id>/session.json" `
  --top 5 `
  --pretty
```

Build the strict evidence prompt for that selected trace:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" packet `
  --run ".cyxwiz/debug_runs/studio/<run-id>/session.json" `
  --top 5 |
  python "docs/Data Studio/tofix42/phase1b_answer.py" prompt --packet -
```

Generate a deterministic selected-trace explanation without a model server:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" explain `
  --run ".cyxwiz/debug_runs/studio/<run-id>/session.json"
```

Emit that explanation as JSON:

```powershell
python "docs/Data Studio/tofix42/phase3_debug_context.py" explain `
  --run ".cyxwiz/debug_runs/studio/<run-id>/session.json" `
  --json `
  --pretty
```

The explanation JSON includes explicit `trace_evidence` and `source_evidence`.
Trace evidence cites selected trace fields such as `selected_trace.status` and
`selected_trace.issues[0].message`; source evidence cites retrieved file and
line ranges.

## Phase 4 Training Trace Commands

Build training trace context:

```powershell
python "docs/Data Studio/tofix42/phase4_training_context.py" context `
  --trace ".cyxwiz/debug_runs/current_training_trace.json" `
  --pretty
```

Explain the latest terminal state:

```powershell
python "docs/Data Studio/tofix42/phase4_training_context.py" explain `
  --trace ".cyxwiz/debug_runs/current_training_trace.json"
```

Build a retrieval-backed Phase 1B-compatible packet for the latest terminal
state:

```powershell
python "docs/Data Studio/tofix42/phase4_training_context.py" packet `
  --trace ".cyxwiz/debug_runs/current_training_trace.json" `
  --top 5 `
  --pretty
```

Build the strict evidence prompt for that terminal state:

```powershell
python "docs/Data Studio/tofix42/phase4_training_context.py" packet `
  --trace ".cyxwiz/debug_runs/current_training_trace.json" `
  --top 5 |
  python "docs/Data Studio/tofix42/phase1b_answer.py" prompt --packet -
```

Run the deterministic Phase 4 gate:

```powershell
python "docs/Data Studio/tofix42/phase4_training_context.py" check
```

## Phase 5 Graph Context Commands

Build selected-node context:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" context `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --node-id 2 `
  --pretty
```

Explain the selected node:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" explain `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --node-id 2
```

Explain one selected node parameter:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" explain `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --node-id 2 `
  --parameter min_df
```

Build a Phase 1B-compatible packet:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" packet `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --node-id 2 `
  --top 5
```

Build a Phase 1B-compatible packet for one selected parameter:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" packet `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --node-id 2 `
  --parameter min_df `
  --top 5
```

Explain a directed graph path:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" path `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --from-node-id 1 `
  --to-node-id 4
```

Build a Phase 1B-compatible packet for a directed graph path:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" path-packet `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --from-node-id 1 `
  --to-node-id 4 `
  --top 5
```

Emit read-only graph improvement suggestions:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" suggest `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --pretty
```

Build a Phase 1B-compatible packet for graph suggestions:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" suggest-packet `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --top 5
```

Emit a read-only graph audit:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" audit `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --pretty
```

Build a Phase 1B-compatible packet for a graph audit:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" audit-packet `
  --graph "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph" `
  --top 5
```

Emit a read-only graph draft plan:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" draft-plan `
  --template text-classification-tfidf-mlp `
  --pretty
```

Build a Phase 1B-compatible packet for a graph draft plan:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" draft-plan-packet `
  --template text-classification-tfidf-mlp `
  --top 5
```

Run the deterministic Phase 5 gate:

```powershell
python "docs/Data Studio/tofix42/phase5_graph_context.py" check
```

## Phase 6 Evaluation Commands

Run deterministic evaluation cases:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" run
```

List deterministic evaluation cases:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" list-cases
```

Capture a bad model answer for later review:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" capture-bad-answer `
  --case-id "phase2.manual.localhost.answer" `
  --query "What does DataLoader pin_memory guarantee today?" `
  --expected-citation "cyxwiz-engine/src/core/graph_compiler.cpp" `
  --actual-output "<paste model output>" `
  --failure-mode "missed_required_citation"
```

Capture failed cases from a saved Phase 2 probe-suite report:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" capture-probe-failures `
  --report "$env:TEMP/tofix42_phase2_probe_suite.json"
```

Summarize reviewed bad-answer corrections:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-status
```

List captured bad-answer records:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" list-bad-answers `
  --unreviewed
```

Show unreviewed bad-answer records with review command templates:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-queue
```

Show one captured bad-answer record in detail:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" show-bad-answer `
  --record-number 1
```

Mark a captured bad answer as reviewed by record number:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-bad-answer `
  --record-number 1 `
  --reviewed-correction "pin_memory=true is unsupported by current batchers and may be ignored; cite cyxwiz-engine/src/core/graph_compiler.cpp."
```

Mark a captured bad answer as reviewed by case filters:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-bad-answer `
  --case-id "phase2.probe.dataloader_pin_memory_truth" `
  --failure-mode "expected_path_missing_from_output,required_terms_missing" `
  --reviewed-correction "pin_memory=true is unsupported by current batchers and may be ignored; cite cyxwiz-engine/src/core/graph_compiler.cpp."
```

Export reviewed corrections as JSONL training candidates:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" export-reviewed `
  --output "docs/Data Studio/tofix42/phase6_reviewed_corrections.jsonl"
```

Run the deterministic Phase 6 gate:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" check
```

## Phase 7 Fine-Tuning Decision Commands

Emit the current fine-tuning readiness decision:

```powershell
python "docs/Data Studio/tofix42/phase7_finetune_decision.py" decide
```

Use a saved real-model probe report:

```powershell
python "docs/Data Studio/tofix42/phase7_finetune_decision.py" decide `
  --real-model-probe-report "$env:TEMP/tofix42_phase2_probe_suite.json"
```

Run the deterministic Phase 7 gate:

```powershell
python "docs/Data Studio/tofix42/phase7_finetune_decision.py" check
```

## Check-All Command

Run every deterministic tofix42 gate:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" run
```

Run the check-all self gate:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" check
```

## Status Command

Show current tofix42 status and next actions:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show
```

Show status and check a running localhost JSON endpoint:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show `
  --endpoint "http://127.0.0.1:8765/completion"
```

Run the status self gate:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" check
```

## Current Generated Files

- `docs/Data Studio/tofix42/phase1a_config.json`
- `docs/Data Studio/tofix42/phase1a_index.json`
- `docs/Data Studio/tofix42/phase1a_handoff.md`
- `docs/Data Studio/tofix42/phase1b_answer.py`
- `docs/Data Studio/tofix42/phase1b_answer.md`
- `docs/Data Studio/tofix42/phase1b_handoff.md`
- `docs/Data Studio/tofix42/phase2_runtime.md`
- `docs/Data Studio/tofix42/phase2_handoff.md`
- `docs/Data Studio/tofix42/phase2_stub_runtime.py`
- `docs/Data Studio/tofix42/phase2_model_validation.md`
- `docs/Data Studio/tofix42/phase2_probe_suite.py`
- `docs/Data Studio/tofix42/phase2_probe_report_check.py`
- `docs/Data Studio/tofix42/phase2_probe_report_check.md`
- `docs/Data Studio/tofix42/phase2_endpoint_doctor.py`
- `docs/Data Studio/tofix42/phase2_endpoint_doctor.md`
- `docs/Data Studio/tofix42/phase2_local_endpoint_scan.py`
- `docs/Data Studio/tofix42/phase2_local_endpoint_scan.md`
- `docs/Data Studio/tofix42/phase2_openai_compat_proxy.py`
- `docs/Data Studio/tofix42/phase2_openai_compat_proxy.md`
- `docs/Data Studio/tofix42/phase2_real_model_check.py`
- `docs/Data Studio/tofix42/phase2_real_model_check.md`
- `docs/Data Studio/tofix42/phase2_real_model_runbook.md`
- `docs/Data Studio/tofix42/phase3_debug_context.py`
- `docs/Data Studio/tofix42/phase3_debug_context.md`
- `docs/Data Studio/tofix42/phase4_training_context.py`
- `docs/Data Studio/tofix42/phase4_training_context.md`
- `docs/Data Studio/tofix42/phase5_graph_context.py`
- `docs/Data Studio/tofix42/phase5_graph_context.md`
- `docs/Data Studio/tofix42/phase6_eval_capture.py`
- `docs/Data Studio/tofix42/phase6_eval_capture.md`
- `docs/Data Studio/tofix42/phase7_finetune_decision.py`
- `docs/Data Studio/tofix42/phase7_finetune_decision.md`
- `docs/Data Studio/tofix42/tofix42_check_all.py`
- `docs/Data Studio/tofix42/tofix42_check_all.md`
- `docs/Data Studio/tofix42/tofix42_current_handoff.md`
- `docs/Data Studio/tofix42/tofix42_status.py`
- `docs/Data Studio/tofix42/tofix42_status.md`

This file is generated. Rebuild it after changing indexed docs or adding indexed
source paths. It also stores the config hash and file-level content hashes for
`status`.

## Current Success Checks

These should return the expected top evidence:

| Query | Expected top evidence |
| --- | --- |
| `What source file defines DebugTraceRecord` | `cyxwiz-engine/src/core/debug_trace_record.h` |
| `TrainingTraceEvent terminal_reason field` | `cyxwiz-engine/src/core/training_trace_collector.h` |
| `TFIDFVectorizer sentiment graph` | `examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph` |
| `RecordTerminalEvent terminal_reason message` | `cyxwiz-engine/src/core/training_trace_collector.cpp` |
| `completed_all_epochs terminal_reason user_cancelled` | `cyxwiz-engine/src/core/training_executor.cpp` |
| `pin_memory no pinned host-memory transfer backend` | `docs/Data Studio/tofix41.md` |
| `DataLoader pin_memory unsupported current batchers compatibility` | `cyxwiz-engine/src/core/graph_compiler.cpp` |
| `TFIDFVectorizer min_df must be >= 1 Configure` | `cyxwiz-engine/src/core/node_executors/tfidf_vectorizer_operator.cpp` |

## Update Log

### Phase 1A

- Added `phase1a_retrieval.py`.
- Added `phase1a_retrieval.md`.
- Built local JSON index.
- Verified first three retrieval success checks.
- Added `check` command for repeatable Phase 1A verification.
- Added `packet` command for retrieval-only answer packets.
- Added `status` command for file-hash freshness checks.
- Added `phase1a_config.json` for source patterns and success checks.
- Added selected terminal-reason implementation files and checks.
- Added method-level C++ chunks and source-type trust boosts for better
  implementation evidence.
- Split oversized C++ chunks into bounded symbol windows for tighter citations.
- Added query-centered previews for long search and packet evidence.
- Added targeted `pin_memory` and `TFIDFVectorizer` evidence checks.
- Added Phase 1A handoff and Phase 1B input contract.

### Phase 1B

- Added `phase1b_answer.py`.
- Added `phase1b_answer.md`.
- Added strict evidence prompt builder.
- Added `runtime none` answer envelope for clean unavailable behavior.
- Added optional explicit `llama-cli` adapter.
- Added Phase 1B `check` command.
- Added UTF-8/UTF-16 packet input handling for PowerShell redirected JSON.
- Added Phase 1B `doctor` command for runner/model/packet readiness.
- Documented pipe-based packet-to-answer usage.
- Added bounded local-runner timeout and structured runtime-error diagnostics.
- Added successful model-output section parsing while preserving raw output.
- Documented `llama-cli` stdout contamination workaround and sentinel parsing.
- Added Phase 1B output sentinel stripping for echoed `llama-cli` prompts.
- Added Phase 1B handoff and aligned next work with init call Phase 2.
- Added Phase 2 runtime boundary and kept `llama-cli` as smoke fallback only.
- Added optional `json-http` local runtime adapter with stubbed success/failure
  checks.
- Added an in-process localhost fixture check for the JSON HTTP adapter.
- Restricted JSON runtime endpoints to localhost-only addresses.
- Added Phase 2 handoff for the completed local JSON runtime boundary.
- Added a localhost-only Phase 2 stub runtime for manual end-to-end checks.
- Verified the stub runtime through the `json-http` answer path.
- Added `probe` for manually validating local JSON runtime answer sections.
- Tightened `probe` so `ok` requires all expected answer sections.
- Added Phase 2 real-model validation checklist and probe cases.
- Recorded failed manual `llama-cli` validation and kept Phase 2 on JSON runtime.
- Added Phase 2 probe suite for repeated localhost JSON runtime validation.
- Added probe-suite JSON report output, top citation details, and output-write
  error reporting.
- Added parsed model output sections to each probe-suite case report.
- Enforced expected evidence path citation in probe-suite model output.
- Added probe-suite case listing and per-case filtering.
- Added explicit probe-suite bad-answer capture flags for raw output and prompt.
- Added Phase 6 bad-answer review queue and record-number review updates.
- Added offline Phase 2 probe report validation and real-model runbook.
- Tightened Phase 2/Phase 7 so stub reports do not count as real-model
  validation evidence.
- Added Phase 2 endpoint doctor for local reachability and stub detection.
- Added Phase 2 local endpoint scan for common localhost ports.
- Added optional endpoint doctor output to the status command.
- Added Phase 2 OpenAI-compatible local proxy for real-model probe servers.
- Added Phase 2 real-model validation orchestrator command.

### Phase 3

- Added deterministic debugger trace context packet builder.
- Added redaction for sensitive payload fields before context handoff.
- Added Phase 3 context check using DebugRunStore-shaped sample data.
- Added Phase 3 retrieval-backed packet output compatible with Phase 1B.
- Added PowerShell stdin BOM handling for piped Phase 3/Phase 1B JSON.
- Added deterministic Phase 3 selected-trace explanation output.
- Added explicit trace-field and source-citation evidence in Phase 3
  explanation output.

### Phase 4

- Added deterministic training trace context and terminal-state explanation.
- Added trace-field and source-citation evidence for terminal reasons.
- Added Phase 4 retrieval-backed packet output compatible with Phase 1B.

### Phase 5

- Added deterministic graph selected-node context and explanation.
- Added Phase 5 graph-node packet output compatible with Phase 1B.
- Added selected graph-node parameter focus with `--parameter`.
- Added directed graph path explanation and Phase 1B-compatible path packets.
- Added read-only graph improvement suggestions and Phase 1B-compatible
  suggestion packets.
- Added read-only graph audit checks and Phase 1B-compatible audit packets.
- Added read-only graph draft plans and Phase 1B-compatible draft-plan packets.

### Phase 6

- Added deterministic evaluation report over retrieval, debugger trace, training
  trace, graph audit, and graph draft-plan fixtures.
- Added bad-answer JSONL capture for reviewed QA corrections.
- Added bad-answer review-status reporting for fine-tuning readiness counts.
- Added capture of failed Phase 2 probe-suite cases into the bad-answer JSONL
  log.
- Added reviewed-correction export as JSONL training candidates.
- Added review-bad-answer for marking captured records as reviewed after
  inspection.
- Added list-bad-answers for reviewing captured QA records.
- Added show-bad-answer for inspecting one captured QA record in detail.

### Phase 7

- Added deterministic fine-tuning readiness decision gate.
- Default decision defers fine-tuning until real local model probes and reviewed
  correction examples exist.

### Consolidation

- Added a check-all harness for running every deterministic tofix42 gate from
  one command.
- Added a current-state handoff with completed phases, blockers, and next
  commands.
- Added a status summary command for index freshness, deterministic checks,
  evaluation, fine-tuning decision, and next actions.

## Current Answer Packet Contract

Schema: `cyxwiz.tofix42.phase1a.answer_packet.v1`

Mode: `retrieval_only`

Model runtime: `not_used`

Fields:

- question
- answer contract rules
- ranked evidence with citation metadata and chunk text
- missing evidence notes

## Next Usage Updates

Add sections here when these exist:

- embedding rebuild command if embeddings are added
- Studio Debugger explanation entry point
- training trace explanation entry point
- graph-aware help workflow
