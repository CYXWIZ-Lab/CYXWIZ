---
title: Build a step from operations
description: Make an SQL step from guided operations and templates instead of typed SQL, preview it on a sample and add checks.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - sql-workbench.md
  - prepare.md
---

A guided step is an SQL step built from operations (keep rows, rename
columns, split text, ...) instead of typed SQL. The step stores its
operations and the SQL they generate, and graphs run the SQL, so a guided
step runs exactly like any [SQL step](sql-workbench.md). Data Studio's
**Prepare** job edits guided steps in place with the same editor (see
[Prepare data with a recipe](prepare.md)).

## Before you start

- An SQL step on the canvas, opened in the SQL Workbench with **Load
  inputs** done, so the operations can offer the input's real columns.

## Steps

1. In the SQL Workbench, choose **Guided**, then **Start a guided step
   instead** (or **Start from a template**).
2. In the operation library, click an operation. It is added as a card after
   the selected one.
3. Select the card and fill its plain fields; column lists come from the
   real input.
4. Select **Preview on sample**. The operations run on the first 1,000 rows
   (or a random 1,000, or all rows), and the rows after each operation show.
5. Optional: add **Add checks** as the last operation.
6. Select **Run step** (Ctrl+Enter) to run on all rows, then save the step.

## Operations

| Group | Operations |
| --- | --- |
| Rows | Keep rows where, Remove duplicate rows, Sort rows, Keep first / sample rows |
| Columns | Choose and rename columns, Add a calculated column, Change type, Split or merge columns |
| Text | Clean text, Find and replace, Keep text between markers, Split text into rows, Extract with a pattern |
| Combine and summarise | Look up from another table, Append rows, Group and summarise, Pivot / unpivot |
| Splits and checks | Assign train / validation / test, Number rows, Add checks |

Switch a card off without deleting it, or move it up or down. An operation
with a problem (a missing column, a bad pattern) is marked in red with the
reason, and the step cannot run until it is fixed.

## Templates

- Split documents into paragraphs.
- Clean a text column.
- Look up and keep matched rows.
- Train / validation / test split.

## Checks from a guided step

**Add checks** goes last. While you edit, its checks run with the preview,
and a failing one stops it with its message. When the step is saved, each
check becomes a **Check** node right after the step (row count, allowed
values, unique, not empty) and the operation leaves the step; Ctrl+Z on the
canvas undoes it. Placed before other operations, it is marked in red.

## The generated SQL

The generated SQL is shown read-only, one commented part per operation.
**Edit as SQL** turns the step into an ordinary SQL step. It asks first,
because the operations are not kept.

The step stores its operations (`guided_ops`) and the SQL they generate
(`query`). In Properties, a guided step lists its operations with on / off
switches and **Edit in guided editor**.

## Related

- [Write SQL steps in the SQL Workbench](sql-workbench.md)
- [Prepare data with a recipe](prepare.md)
- [Explore your data](explore.md)
