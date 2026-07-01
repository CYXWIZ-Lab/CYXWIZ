# done31e - Import dialog inspection-only controls

## Scope

This is the fifth completed small ticket extracted from `tofix31.md`.

The goal is to remove remaining UI affordances that imply the import dialog can
load trainable checkpoint state. The dialog remains an inspection and
`.cyxmodel` graph-extraction tool.

## Changes made

- `cyxwiz-engine/src/gui/panels/import_dialog.cpp`
  - Renamed `Import Options` to `Inspection Options`.
  - Replaced active `Load Optimizer State` and `Load Training History`
    checkboxes with inspection-only status text.
  - Explicitly resets `load_optimizer_state_` and `load_training_history_` to
    false while rendering the inspection-only state.
  - Disables `Strict Mode` and `Allow Shape Mismatch`, because those controls
    only make sense once checkpoint tensors are mapped into trainable layers.
  - Keeps the existing result warnings that no weights are loaded into a
    trainable Studio model.

## Current dialog behavior

- Probe file metadata.
- Show model/layer/content metadata.
- Extract `.cyxmodel` graph JSON when present.
- Do not load weights.
- Do not load optimizer state.
- Do not resume training.
- Do not apply strict layer matching.
- Do not allow shape-mismatch checkpoint loading.

## Future implementation requirements

A real import-to-training dialog should add these controls back only after:

- parameter-name mapping exists,
- tensor shape validation exists,
- optimizer-state compatibility exists,
- freeze/unfreeze ownership exists,
- tokenizer/preprocessor packaging exists,
- runtime tests prove a loaded model can train and resume correctly.
