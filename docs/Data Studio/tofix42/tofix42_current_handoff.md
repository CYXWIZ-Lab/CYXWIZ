# tofix42 Current Handoff

## Status

The local CyxWiz source-aware assistant harness is implemented through the
read-only planning and evaluation phases. The current implementation is command
first, deterministic by default, and does not require Studio UI or a live model
server for checks.

## Verification

Run every deterministic gate:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" run
```

Expected current result:

```text
OK: True
Checks: 14 passed=14 failed=0
```

The Phase 1A index is fresh when:

```powershell
python "docs/Data Studio/tofix42/phase1a_retrieval.py" status
```

reports matching indexed/current file counts.

Current checkpoint:

```text
Indexed files: 52
Current files: 52
Indexed chunks: 602
Status: fresh
```

## Implemented Phases

- Phase 1A: local JSON index, lexical retrieval, citations, freshness checks.
- Phase 1B: strict answer packet and local runtime adapter boundary.
- Phase 2: localhost JSON runtime adapter, stub runtime, probe suite, real-model
  validation checklist, endpoint doctor, local endpoint scan,
  OpenAI-compatible proxy, and one-command real-model check.
- Phase 3: debugger trace context, packets, and deterministic explanation.
- Phase 4: training trace context, packets, and deterministic terminal-state
  explanation.
- Phase 5: graph selected-node help, parameter focus, path explanation,
  suggestions, audit, and draft-plan packets.
- Phase 6: deterministic evaluation report and bad-answer JSONL capture.
- Phase 7: fine-tuning readiness decision gate.
- Consolidation: check-all runner for deterministic gates.
- Status: current-state command with optional endpoint reachability and stub
  detection.

## Current Decisions

- Retrieval remains the truth layer.
- Fine-tuning is deferred by default.
- `llama-cli` is not the validation path because the tested build entered a
  chat-style stdout path and later hung under subprocess validation.
- Real model validation should use a manually started localhost JSON endpoint.
- OpenAI-compatible local model servers can be bridged through
  `phase2_openai_compat_proxy.py` on port `8766`.
- `phase2_local_endpoint_scan.py` checks common local ports before validation.
- `phase2_real_model_check.py` separates `real_model_probe_accepted` from
  `fine_tuning_ready`.
- Stub runtime reports validate the JSON pipeline only and do not satisfy the
  real-model Phase 7 criterion.
- No graph/source mutation is implemented.
- No final `.cyxgraph` generation is implemented.
- No Studio UI integration is implemented.

## Remaining Blockers

1. A real localhost JSON model server still needs to pass the Phase 2 probe
   suite.
2. Reviewed bad-answer corrections need to be captured before any fine-tuning
   experiment can be justified.
3. Studio UI integration is still a later optional step.
4. Final graph JSON generation and apply flows remain out of scope until a
   future approval/preflight path exists.

## Useful Commands

Show current status and next actions:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show
```

Show current status plus local endpoint reachability/stub detection:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show `
  --endpoint "http://127.0.0.1:8765/completion"
```

Run deterministic evaluation:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" run
```

Emit the current fine-tuning decision:

```powershell
python "docs/Data Studio/tofix42/phase7_finetune_decision.py" decide
```

Run real-model probes after starting a localhost JSON model server:

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

Run the full validation sequence through the OpenAI-compatible proxy:

```powershell
python "docs/Data Studio/tofix42/phase2_real_model_check.py" run `
  --endpoint "http://127.0.0.1:8766/completion" `
  --output "$env:TEMP/tofix42_phase2_probe_suite.json" `
  --capture-failures
```

Check endpoint reachability and stub detection:

```powershell
python "docs/Data Studio/tofix42/phase2_endpoint_doctor.py" check `
  --endpoint "http://127.0.0.1:8765/completion"
```

Scan common local endpoint locations:

```powershell
python "docs/Data Studio/tofix42/phase2_local_endpoint_scan.py" scan
```

Bridge a local OpenAI-compatible chat server to the tofix42 `/completion`
contract:

```powershell
python "docs/Data Studio/tofix42/phase2_openai_compat_proxy.py" serve `
  --upstream "http://127.0.0.1:1234/v1/chat/completions" `
  --model "local-model" `
  --port 8766
```

Capture failed probe cases into the bad-answer log:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" capture-probe-failures `
  --report "$env:TEMP/tofix42_phase2_probe_suite.json"
```

Show the bad-answer review queue:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-queue
```

Capture a reviewed bad answer:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" capture-bad-answer `
  --case-id "phase2.manual.localhost.answer" `
  --query "<question>" `
  --expected-citation "<expected file path>" `
  --actual-output "<model output>" `
  --failure-mode "<short failure mode>" `
  --reviewed-correction "<reviewed correction>"
```

Check reviewed correction counts:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-status
```

List unreviewed bad answers:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" list-bad-answers `
  --unreviewed
```

Show a bad answer for review:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" show-bad-answer `
  --record-number 1
```

Mark a captured bad answer as reviewed:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" review-bad-answer `
  --case-id "<case id>" `
  --reviewed-correction "<reviewed correction>"
```

Export reviewed corrections:

```powershell
python "docs/Data Studio/tofix42/phase6_eval_capture.py" export-reviewed `
  --output "docs/Data Studio/tofix42/phase6_reviewed_corrections.jsonl"
```

## Handoff Rule

Before adding new features, run `tofix42_check_all.py run`. If it fails, fix the
deterministic gate before expanding scope.
