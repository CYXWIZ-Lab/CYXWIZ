# Data Studio Parallel To Fix Workflow

Created: 2026-06-08

Purpose: coordinate parallel `tofix` work without duplicating implementation,
reverting unrelated edits, or letting stale docs become the source of truth.

## Roles

- Main integrator owns final scope decisions, conflict review, tests, commits,
  and pushes.
- Worker A owns bounded runtime architecture work, currently `tofix20`.
- Worker B owns bounded model-family truth and guardrail work, currently
  `tofix19`.

## Current Assignments

### Worker A - Runtime Architecture

Primary doc:

- `docs/Data Studio/tofix20.md`

Allowed write scope:

- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.h`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`
- `cyxwiz-engine/tests/test_pipeline_operator_metadata.cpp`
- `cyxwiz-engine/tests/test_pipeline_executor_operator_routing.cpp`
- `docs/Data Studio/tofix20.md`
- `docs/Data Studio/tofix_doc_truth_audit.md`

Focus:

- canonical runtime ownership
- remaining string-only legacy alias migration
- pre-execution schema/type validation parity

### Worker B - Model Family Truth

Primary doc:

- `docs/Data Studio/tofix19.md`

Allowed write scope:

- `docs/Data Studio/tofix19.md`
- `docs/Data Studio/tofix14.md`, only if needed for direct NER/Siamese truth
- focused compiler/training guardrail files under `cyxwiz-engine/src/core`
- focused tests under `cyxwiz-engine/tests`

Focus:

- unsupported model-family import/compile/training guardrails
- truthful comparison against PyTorch-level capabilities
- examples or graph paths that could imply unsupported families train

## Worker Rules

- Read previous implementation before coding.
- Do not duplicate an existing implementation.
- Do not revert changes made by the user, main integrator, or another worker.
- Keep patches inside the assigned write scope.
- Prefer one small, testable fix over a broad architectural rewrite.
- If no safe small fix exists, report evidence instead of forcing a change.

## Required Worker Report

Each worker must report:

- files changed
- exact issue fixed or evidence that no safe fix was found
- commands run and whether they passed
- remaining risk
- recommended next item

## Integration Rules

- Main integrator reviews every worker patch before staging.
- Only one worker's patch is integrated at a time.
- Relevant narrow tests run before commit.
- `git diff --check` runs before every commit.
- Commits include only the files for the accepted slice.
- Unrelated dirty workspace files remain untouched.
