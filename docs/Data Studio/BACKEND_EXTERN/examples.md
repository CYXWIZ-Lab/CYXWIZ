# BackendExtern Examples

## Example 1: User-visible Data Studio run

```text
Input table: customer_churn.parquet
Node: Approved External Tabular Model
Task: binary classification
Runtime: cyxwiz-tabular-2026.1
Backend: Auto (selected PyTorch CPU)
Model: approved/model@immutable-revision

Outputs:
  prediction table
  probability column
  External Run Report
```

The report states that the computation was external and was not executed by the
native ArrayFire path.

## Example 2: Managed request/result sequence

```text
Engine -> worker: health(protocol=1, runtime=cyxwiz-tabular-2026.1)
worker -> Engine: healthy(framework=pytorch, device=cpu)

Engine -> worker: run(predict, provider=approved.tabular,
                      input=table.arrow, model=revision-hash)
worker -> Engine: event(started)
worker -> Engine: event(progress, 0.5)
worker -> Engine: completed(predictions.arrow, sha256, run-metadata.json)

Engine: validate artifacts -> import result -> mark graph run complete
```

## Example 3: User script remains separate

```python
# Runs in the current project Python environment through ScriptingEngine.
import torch
import pycyxwiz

print(torch.__version__)
# The user owns package installation and model semantics on this path.
```

This does not register a managed external provider or make the model
reproducible by itself.

