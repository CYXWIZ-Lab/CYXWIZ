---
title: Fix missing values before training
description: Fill or drop empty cells and NaN values from the report Train shows, so training never reads them as numbers.
type: how-to
area: graph-editor
audience: user
status: experimental
since: 1.0.8
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - compile-and-run.md
---

Use this when **Train** stops with **Cannot start training** and the report
shows cards about missing values.

Training never uses an empty cell, or NaN in a number column, as 0 or as
class 0. Before a run starts, **Train** counts them in the table the Data
Split reads (after the steps before it ran, or as kept on disk from an
earlier session) and in each Data Input on the Split's Validation or Test
pin. The count runs in the background (**Tasks**: "Checking the data for
missing values..."), so the Engine stays responsive on a large table.

When there are some, **Cannot start training** says how many columns have
them and shows a card per group. (If you open another graph during the
check, press Train again to see this graph's cards.)

## Before you start

- A graph with a Data Input, a Data Split and a model, and **Train**
  pressed once.
- Example: `examples/cyxgraph/missing_values_fix.cyxgraph`.

## Fix feature columns

The **Feature columns** card has a row per number column: how many cells
are empty, how many are NaN, and what a fill would use.

1. Choose **Fill numbers with**: **Mean**, **Median** or **0**.
2. Click **Fill these columns**. A Missing Value step is added. Its value
   is learned from the Train rows only.

Or click **Drop these rows** to add a Row Filter that keeps the rows where
each of these columns is not empty.

A column more than half empty can be left out of the features instead:
open its **Actions** and its role in the Data Input becomes Ignore.

A column whose name has a comma, spaces at its ends or a leading `"` is
written in double quotes in the step's **Feature columns** list
(`age, "income, net"`; a quote inside is doubled).

## Fix the label

Click **Drop rows without a label** on **The label** card. A Row Filter
keeping `"<label>" IS NOT EMPTY` is added. A label is never filled in. For
a regression target of several columns (`y`, `y_1`, ...), one Row Filter
keeps the rows that have every target.

## Train

The added steps go after the step that feeds the Data Split, one after the
other in the card's order, so Train runs them first. After each action the
card says what was added, with **Undo**. **Ctrl+Z** on the canvas removes
the last added step and reopens its group; opening another graph starts the
cards over.

- **Fix and train** fills the feature columns and drops the rows without a
  label, then trains. If the canvas could not add a step, the report opens
  again instead.
- Once every group is fixed, the button is **Train**.

Missing Value and Row Filter treat NaN in a number column as an empty cell.

## If something goes wrong

**A Validation or Test input has missing values**: it is read from its own
file, so a step after it would not apply. Fix the file, then click **Apply**
on its Data Input again.

**"Failed: the data has missing values the check before training could not
see"** on the Training Dashboard: a table kept on disk (Parquet) counts its
empty cells before the run but finds NaN only as rows are read, so the run
stops at that batch. The Dashboard shows the same card; add a step, then
click **Train again**.

## Related

- [Compile and run a graph](compile-and-run.md)
