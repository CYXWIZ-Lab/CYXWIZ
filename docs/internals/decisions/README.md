---
title: Decision records
description: The index of CyxWiz architecture decisions, why each was made, and how to add a new record.
type: reference
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - ../coding-standards.md
  - ../architecture/overview.md
---

A decision record keeps the reason for a lasting architectural choice next
to the code, so it is not argued again without new facts. Each record is
short: its status, the context, the decision and its consequences.

## Records

| No. | Decision | Status |
| --- | --- | --- |
| [0001](0001-arrayfire-for-compute.md) | Tensors execute on ArrayFire with native providers; LibTorch is only the test oracle | Accepted |
| [0002](0002-arrow-data-path.md) | Arrow in memory and Parquet on disk are the one tabular data path | Accepted |
| [0003](0003-agpl-and-commercial-licence.md) | AGPL-3.0 for the Engine and Server Node, Apache-2.0 for the SDK, a commercial licence on request | Accepted |
| [0004](0004-no-legacy-loader-shims.md) | Old graphs are rebuilt on current contracts; the loader refuses retired nodes | Accepted |
| [0005](0005-docs-in-the-repository.md) | All documentation lives in `docs/` and changes with the code | Accepted |
| [0006](0006-four-pipeline-bands.md) | Every training graph is read as four bands | Accepted |

## Statuses

| Status | Meaning |
| --- | --- |
| Proposed | Under discussion; not yet followed |
| Accepted | Followed by the code and by reviews |
| Superseded | Replaced; the record names its successor |

## Add a record

1. Copy an existing record and give it the next number:
   `NNNN-short-name.md` in this folder.
2. Fill in **Status** (with the date), **Context** (the problem and the
   forces), **Decision** (what is done, in the present tense) and
   **Consequences** (what follows, good and bad). Keep it under a page.
3. Public reasons only: no internal issue numbers, private notes or
   customer details.
4. Add a row to the table above. When a record replaces another, set the old
   one to Superseded and link the new one.
