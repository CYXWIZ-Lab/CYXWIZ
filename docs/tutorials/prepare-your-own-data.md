---
title: Bring in and prepare your own data
description: Lesson 3. Load a table into Data Studio, profile it, fill and drop its missing values, and add a check before training.
type: tutorial
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - build-a-graph-yourself.md
  - ../user-guide/data-studio/index.md
  - ../user-guide/data-studio/missing-values-and-outliers.md
  - ../main-window/data-input-dialog.md
---

In this lesson you bring a table with gaps into the Engine and make it fit
for training: you load it through a Data Input, read its profile, fill the
empty feature cells, drop the rows without a label, and add a check. Every
step you take in Data Studio becomes a node of the graph, so training later
reads exactly what you prepared.

The table stands in for your own data: `examples/datasets/patients_missing.csv`,
300 made-up rows (no real patients) with five features (`age`, `bmi`,
`blood_pressure`, `cholesterol`, `smoker`) and a risk class `diagnosis` of
0, 1 or 2. Some `age`, `bmi` and `cholesterol` cells are empty, and 5 rows
have no `diagnosis`.

**You need:** the `cyxwiz-tutorials` project from
[lesson 1](tour-the-main-window.md), the file `patients_missing.csv`, about 20 minutes.

## 1. Start a new graph

1. On the Studio bar, select the graph's button (it shows the graph's
   name), then **Clear the canvas...**, and confirm with **Clear the
   canvas**. Skip this when the canvas is already empty.
2. The graph's file has not changed: you give this graph its own name in
   part 2, so lesson 2's graph is not overwritten.

## 2. Bring the table in

1. Select the **Data Studio** tab, then the **Bring in** job.
2. Select **Add Data Input...**. A Data Input node appears on the canvas
   and its dialog opens on the **Source** tab.
3. Under **1. Source**, select **Browse file...** and pick
   `examples/datasets/patients_missing.csv` in the Engine's folder.
   **Detected** shows a CSV table with 6 columns.
4. Under **2. Use the rows as**, select **Table**, and set **Label column**
   to `diagnosis`. Every other column becomes an input feature.
5. Select **Apply**. The strip at the top of the dialog reads *Loaded*.
   The **Preview** tab shows the first rows; some cells are empty.
6. Select **OK**.
7. On the Studio bar, open the graph's button, select **Save As...** and
   save the graph in the project's `cyxgraph` folder as `patients`.

The **Bring in** table lists the node with 300 rows x 6 columns, *loaded*.

## 3. Read the profile

1. Select the **Explore** job, then **Profile**. The table you loaded is
   picked in the rail on the left (**Data Inputs**).
2. Read the second line: the columns per role, the **missing cells**, the
   **duplicate rows** and the **data quality**.
3. Read the table, one row per column. `diagnosis` has the role
   **Target**, from **contract**: the Data Input's label column. The other
   five are **Numeric** or **Category**.
4. Click `age`. Its details show its distribution and how many cells are
   missing.

## 4. Fill the empty features

1. In **Explore**, select **Missing**. The summary counts the missing
   cells and the incomplete rows; **Columns** lists `age`, `cholesterol`,
   `diagnosis` and `bmi` with their counts. (The CSV reader may read the
   `NaN` cells of `bmi` as empty; either way they count as missing.)
2. Click the `age` row.
3. In **Fill with**, pick `median`, then select **Fill in Prepare**. A
   Fill Missing Values step, *Fill age (median)*, goes in after the Data
   Input, and the line under the button says what was added.
4. Do the same for `bmi` and `cholesterol`.

A fill step learns its value from the training rows only, once the graph
has a Data Split (lesson 4 adds one), and then fills every row.

## 5. Drop the rows without a label

A label is never filled in: a guessed diagnosis would teach the model
something false.

1. In **Missing**, click the `diagnosis` row.
2. Select **Drop these rows in Prepare**. A guided step *Keep rows with
   diagnosis* goes in after the Data Input too.

## 6. Look at the recipe

1. Select the **Prepare** job. **RECIPE STEPS (NODES)** lists the four
   steps (three fills and *Keep rows with diagnosis*) in the order they
   run. Each new step went in right after the Data Input, in front of the
   ones before it; the order does not change the result here.
2. Click *Keep rows with diagnosis*. It opens in Guided mode.
3. Select **Run step**. The Data Pipeline runs the steps and shows the rows
   that are left: 295.
4. Select **Show graph** in the Data Studio header. On the canvas the
   steps follow the Data Input in one linked row. The last node of the row
   has an output pin with no link yet: lesson 4 connects the model there.

## 7. Add a check

A check stops a run when the data is not what you expect, for example
after the file is replaced.

1. Back in Data Studio, select the **Validate** job.
2. Select **Add a check**, then **Not empty**.
3. In **After**, pick *Keep rows with diagnosis*. Pick the column
   `diagnosis`. *Rule:* reads the rule in words.
4. Select **Add check**, then **Run checks**. The line above the table
   reads *1 of 1 checks pass*.

## 8. Save

Click the canvas and press **Ctrl+S**. The graph `patients` now holds the
Data Input, the preparation steps and the check, but no model yet.

## Clean up

Nothing to remove. Lesson 4 builds the model on this graph.

## Next steps

- Lesson 4: [Build a graph yourself](build-a-graph-yourself.md)
- The whole Data Studio path: [Data Studio](../user-guide/data-studio/index.md)
- Every view of Explore: [Explore a dataset](../user-guide/data-studio/explore.md)
- More kinds of step: [Prepare data with a recipe](../user-guide/data-studio/prepare.md)
- Every kind of check: [Check a dataset before training](../user-guide/data-studio/validate.md)
- Every option of the Data Input: [The Data Input dialog](../main-window/data-input-dialog.md)
