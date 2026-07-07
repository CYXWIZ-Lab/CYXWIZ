# tofix42 Check-All Harness

## Purpose

Run every deterministic tofix42 gate from one command.

This wrapper does not start model servers, call external endpoints, launch
Studio, mutate graphs, or run training. The Phase 2 entry only verifies that the
probe cases are registered; real localhost model validation still uses the
Phase 2 probe suite separately.

Current deterministic gate count: `14`.

Phase 2 deterministic checks include probe-case registration, saved-report
validation, endpoint-doctor self-check, local-endpoint-scan self-check,
OpenAI-compatible proxy self-check, and real-model-check self-check. They do
not call a real model.

## Commands

Run all checks:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" run
```

Run all checks and emit JSON:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" run --json
```

Stop after the first failure:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" run --stop-on-fail
```

Run the check-all self gate:

```powershell
python "docs/Data Studio/tofix42/tofix42_check_all.py" check
```

## Output

Report schema: `cyxwiz.tofix42.check_all.v1`

The report includes phase, check name, command arguments, return code, and
captured stdout/stderr tail for each deterministic check.
