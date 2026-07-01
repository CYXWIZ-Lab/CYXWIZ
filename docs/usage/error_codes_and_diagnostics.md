# Error codes and diagnostics

CyxWiz shows readable error messages together with stable `CW-*` error codes.

The message explains what happened. The code helps users, engineers, QA, and
support route the issue to the right subsystem.

Example:

```text
[CW-C-0302] CrossEntropy class count (3) does not match the model output size (2)
```

Read this as:

- `CW` means CyxWiz.
- `C` means the compiler or graph validation layer.
- `0302` identifies a label/output shape mismatch.
- The readable message explains the exact graph problem.

## Where codes appear

Error codes can appear in:

- compile result popups
- training launch failures
- Studio Debugger issue panels
- trace rows in Studio Debugger
- engine logs
- support bundles
- data import/export errors
- GPU fallback warnings
- checkpoint, model export, and model import failures

If you report a problem, include both the code and the full readable message.

## Common domains

| Prefix | Area | What it usually means |
|---|---|---|
| `CW-C-*` | Compiler / graph validation | The graph structure or node configuration is invalid before runtime starts. |
| `CW-R-*` | Runtime / pipeline execution | A graph or operator failed while executing. |
| `CW-D-*` | Data contract / schema | Dataset columns, labels, shapes, row counts, or text vocabulary coverage are wrong or suspicious. |
| `CW-F-*` | File and IO | A file path, file format, read, write, or permission problem occurred. |
| `CW-G-*` | GPU backend | CUDA, ArrayFire, kernel execution, GPU placement, or GPU fallback issue. |
| `CW-P-*` | CPU backend | CPU fallback or host tensor execution issue. |
| `CW-M-*` | Memory / resource | Host RAM, GPU memory, batch size, or resource exhaustion issue. |
| `CW-T-*` | Training loop | Model build, loss setup, optimizer setup, or training execution issue. |
| `CW-U-*` | UI / workflow | Studio state or user workflow is invalid, such as trying to train from an invalid state. |
| `CW-S-*` | Serialization / artifacts | Model save/load, checkpoint, `.cyxmodel`, or manifest issue. |
| `CW-X-*` | External integration | Optional integration or third-party dependency failure. |

## Common codes users may see

| Code | Meaning | Typical action |
|---|---|---|
| `CW-C-0101` | Missing required training path node | Add the required DataInput, model layer, loss, or optimizer node. |
| `CW-C-0102` | Unsupported training node | Replace the selected node or wait for backend support for that node. |
| `CW-C-0103` | Invalid graph connectivity | Check that Data, model, loss, and optimizer pins are connected in the correct direction. |
| `CW-C-0301` | Tensor shape mismatch | Check layer shape expectations, sequence dimensions, reshape parameters, or tensor operation dimensions. |
| `CW-C-0302` | Label/output mismatch | Make model output units match class count, tags, or binary loss requirements. |
| `CW-C-0401` | Invalid compiler parameter | Fix a node property such as `label_smoothing`, split ratios, max length, or scalar tensor parameter. |
| `CW-C-0001` | Generic compiler diagnostic | Read the message; this is a compiler warning/error without a narrower code yet. |
| `CW-D-0101` | Required column missing | Select the correct dataset column or update the node property. |
| `CW-D-0102` | Required label column missing | Choose a valid label/target column for supervised training. |
| `CW-D-0301` | Data/domain type mismatch | Use preprocessing nodes that match the DataInput domain, such as text preprocessing for text data. |
| `CW-D-0302` | Row count or sample count mismatch | Check dataset split, labels, empty tables, or inconsistent feature/label rows. |
| `CW-D-0304` | Vocabulary coverage warning | Text sample has many unknown tokens; check tokenizer, vocabulary, casing, min frequency, or training corpus. |
| `CW-D-0401` | Invalid data split | Make train/validation/test ratios valid. |
| `CW-D-0501` | Materialization failed | Check vectorizer/preprocessing settings, memory pressure, and dataset schema. |
| `CW-R-0201` | Runtime node unsupported | The graph compiled into a node path that runtime does not execute yet. |
| `CW-R-0301` | Runtime input dataset missing | Load or select the required dataset before running. |
| `CW-R-0401` | Runtime parameter invalid | Fix the node parameter mentioned in the message. |
| `CW-R-0501` | Runtime/operator execution failed | Inspect the exact operator message and Studio Debugger trace. |
| `CW-G-0201` | GPU path disabled by policy | Engine intentionally routed a risky GPU path to CPU, often for recurrent kernels. |
| `CW-G-0501` | GPU kernel execution failed | GPU operation failed and may fall back to CPU if supported. |
| `CW-T-0101` | Invalid training setup | Training launch was blocked because the compiled setup is not runnable. |
| `CW-T-0102` | Model build failed | The compiled model/loss/optimizer could not be constructed. |
| `CW-T-0501` | Training execution failed | Training, smoke run, or local debug hit a runtime execution failure. |
| `CW-U-0101` | Invalid UI workflow state | Run the missing step first, such as compiling or local debugging before training. |
| `CW-S-0501` | Model save failed | Check output path, permissions, and artifact directory. |
| `CW-S-0502` | Model load failed | Check model artifact, manifest, or weights file validity. |
| `CW-M-0703` | Batch too large | Reduce batch size, feature width, sequence length, or model size. |

## How to use codes while debugging

Start with the readable message. The code only tells you the category.

Recommended workflow:

1. Read the full message next to the code.
2. Open Studio Debugger and inspect the related issue or trace row.
3. If the code is `CW-C-*`, fix the graph before trying to train.
4. If the code is `CW-D-*`, inspect dataset columns, labels, splits, schema, and materialization.
5. If the code is `CW-T-*`, run Local Debug or Smoke Run to isolate build, loss, gradient, or execution failure.
6. If the code is `CW-G-*`, check whether training continued on CPU fallback or failed completely.
7. If the code is `CW-M-*`, reduce memory pressure before retrying.

## What to send in a bug report

Include:

- the full `CW-*` code
- the complete readable message
- what action triggered it, such as Compile, Train, Local Debug, Smoke Run, Test, Import, or Export
- graph name or example file
- dataset type and shape, without private row contents
- whether GPU or CPU backend was selected
- the latest Studio Debugger support bundle if requested

Do not send private dataset rows, credentials, tokens, or secrets.

Support bundles redact sensitive paths, dataset previews, and token-like strings,
but users should still avoid intentionally sharing private data.

## Early stopping and error codes

Early stopping is not an error by itself.

If training stops with a status such as `early_stopped`, read the training
status and checkpoint metadata instead of looking for an error code.

Error codes are for blocked, failed, suspicious, or degraded paths. Early
stopping is a controlled training outcome when validation stops improving.

## GPU fallback warnings

A `CW-G-*` warning does not always mean training failed.

Some GPU paths intentionally fall back to CPU when the engine detects an
unsupported or risky backend path. In that case training may continue, but it
can be slower.

If a GPU warning is followed by a training failure, use the Studio Debugger
trace and the full message to identify whether the failure is GPU execution,
memory pressure, unsupported shape, or a model/runtime issue.

## Generic fallback codes

Some diagnostics may use a generic fallback code such as `CW-C-0001`.

This means the diagnostic is still structured and searchable, but it does not
yet have a narrower stable code. The readable message is the source of truth for
what to fix.

Generic codes are useful for support routing, but they should not be treated as
the final root cause by themselves.
