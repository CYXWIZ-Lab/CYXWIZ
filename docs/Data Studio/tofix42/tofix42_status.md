# tofix42 Status Summary

## Purpose

Show the current operational state of the tofix42 harness in one command.

This command is read-only and local. It aggregates index freshness, deterministic
checks, Phase 6 evaluation, and the Phase 7 fine-tuning decision. It does not
start model servers, launch Studio, mutate graphs, run training, or fine-tune a
model.

## Commands

Show status:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show
```

Show status as JSON:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show --json
```

Show status and probe a local JSON endpoint:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" show `
  --endpoint "http://127.0.0.1:8765/completion"
```

Run the status self gate:

```powershell
python "docs/Data Studio/tofix42/tofix42_status.py" check
```

## Output

Status schema: `cyxwiz.tofix42.status.v1`

The summary includes:

- index freshness and chunk count
- deterministic check-all result
- Phase 6 evaluation result
- optional localhost JSON endpoint reachability and stub detection
- bad-answer reviewed-correction counts
- Phase 7 fine-tuning recommendation
- blocked or pending criteria
- next actions

The endpoint probe is optional. Default `show` and `check` commands do not
require a live model server and do not start one.
