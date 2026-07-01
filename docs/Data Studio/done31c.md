# done31c - Pretrained/import truthfulness boundary

## Scope

This is the third completed small ticket extracted from `tofix31.md`.

The goal is to keep Studio truthful about model import and pretrained
fine-tuning. This ticket does not implement pretrained transformer import,
weight mapping, or fine-tuning. It narrows user-facing wording so the current
inspection-only behavior is not mistaken for trainable model import.

## Current engine contract

- Supported now: inspect model metadata through the import dialog.
- Supported now: extract `.cyxmodel` graph JSON when present.
- Not supported yet: loading imported checkpoint tensors into a trainable
  `SequentialModel`.
- Not supported yet: freeze/unfreeze ownership for imported weights.
- Not supported yet: pretrained transformer fine-tuning contract.
- Not supported yet: tokenizer/preprocessor packaging for arbitrary imported
  pretrained models.

## Code locations checked

- `cyxwiz-engine/src/gui/panels/import_dialog.cpp`
  - Button already says `Inspect`.
  - Result warnings already state that no weights are loaded into a trainable
    Studio model.
  - Updated progress text from `Importing...` to `Inspecting...`.
  - Updated `Transfer Learning` heading to `Import-to-Training Status`.
- `cyxwiz-engine/src/core/graph_compiler.cpp`
  - Rejects imported/pretrained fine-tuning sketches.
  - Error explains the missing parameter mapping, shape validation,
    freeze/unfreeze ownership, optimizer-state compatibility, and
    tokenizer/preprocessor packaging.
- `cyxwiz-engine/src/gui/node_documentation.cpp`
  - Updated AdamW docs to avoid implying imported Transformer fine-tuning is
    already a first-class Studio workflow.

## Not done in this ticket

- No pretrained model weight import.
- No ONNX/HuggingFace checkpoint-to-training parameter mapping.
- No freeze/unfreeze optimizer ownership.
- No imported optimizer-state compatibility.
- No tokenizer/preprocessor packaging for arbitrary pretrained models.

## Acceptance check

- Import dialog user-facing language consistently describes inspection, not
  trainable import.
- Compiler remains the source of truth for rejecting unsupported pretrained
  fine-tuning sketches.
- Future implementation can be split into a real import-to-training contract
  instead of partial UI promises.
