---
title: Prepare data with a recipe
description: Edit, run and add the steps of the graph's Preparation Recipe in Data Studio's Prepare job.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - validate.md
  - sql-workbench.md
  - ../../reference/nodes/categories/preprocessing.md
---

The **Prepare** job of Data Studio shows the graph's Preparation Recipe step
by step: the nodes that turn the loaded data into the table training reads.
Every change here is a change to the graph, shown on the canvas.

## The step list

On the left, **RECIPE STEPS (NODES)** lists each step node in run order
with its kind and its last run (*SQL Query - 15,349 rows - 3.3 s*, *Check -
pass*, *failed* in red with the error on hover, *not run*). Click a step to
show it; double-click it to show it on the canvas.

A graph with several recipes lists them first. Nodes that prepare data
outside a recipe are listed as *Steps outside a recipe*.

## Edit and run an SQL step

An SQL step opens in the SQL Workbench's editor (highlighting, completion
with its inputs' columns once they are loaded, find, **Format**). The header
names the step, *SQL Query node* and what it reads.

1. Edit the query.
2. Click **Run step** (or press Ctrl+Enter in the editor). The query is
   saved to the step, then the Data Pipeline runs through it and its Check
   nodes.
3. Read the rows it made, under the editor (*15,349 rows - 13 columns - at
   21:16:55*, the first 1,000 in the grid), with the rows each part of the
   query keeps (*src 7 > parts 17,083 > kept 15,385 > result 15,349*). The
   checks' *pass* or *fail* show in the step list.

| Button | What it does |
| --- | --- |
| **Save to step** | Writes the query into the node without running it, also inside a collapsed recipe; Ctrl+Z on the canvas undoes it. *not saved* shows while the editor differs from the step. |
| **Reset** | Goes back to the step's saved query |
| **Open in Workbench** | Opens the step's SQL Workbench dialog (Plan, Parts, Guided mode); a collapsed recipe opens on the canvas first |
| **Add a step after** | Adds an **SQL step** or a **Guided step** after this one in the recipe |

## Guided steps

**Guided | SQL** above the editor switches how the step is edited.

- A guided step opens in Guided mode: its operations as cards (with the rows
  after each one once the step has run) and the selected operation's fields
  beside them, or under them when the window is narrow.
- **Add an operation** lists the operations by group (type to find one) and
  adds it after the selected card. **Start from a template**, **Undo** and
  **Redo** work as in the [SQL Workbench](guided-steps.md).
- **Save to step** writes the operations and the SQL they make; **Run
  step** saves, then runs. An operation marked in red must be fixed before
  the step saves or runs.
- In SQL mode a guided step's query is read-only. **Edit as SQL** (it asks
  first) turns it into an ordinary SQL step when you save.
- An SQL step in Guided mode offers **Start a guided step instead**, which
  replaces the query when you save (**Reset** brings it back until then).

## Other kinds of step

- **Scale, power and log steps** (Standard, Min-Max, Robust and Max-Abs
  Scaler, Quantile, Power and Log Transform) show their settings as typed
  fields and switches. A scaler's **Method** switch replaces the node with
  another scaler, keeping its links and columns. The view shows the last
  run's before and after per column with two histograms, a power step's λ,
  skewness, normality test and likelihood curve, and **Compare methods**.
  See [scaling and transforms](../../reference/nodes/categories/preprocessing.md#the-step-in-prepare).
- **Other steps** (a Check, a step node) show their last outcome, their
  settings as saved in the graph, **Run step**, **Open dialog** and **Show
  on canvas**. Change their settings in the node's dialog.

## Add a step

1. In the step list, select the step the new one should follow.
2. Click **+ Add a step** under the list (or **Add a step after** beside an
   SQL step).
3. Pick one: an **SQL step**, a **Guided step**, a **Row Filter**,
   **Joiner**, **GroupBy** or **Text Clean**, a step from **Scale and
   transform** (Scale, Log transform, Power transform), or any other
   preparation node from **More nodes**.

The new node goes on the canvas to the right of the selected step, the nodes
after it move right to make room (a collapsed recipe opens first), and it is
selected there and in the list. Data Studio stays in front. Ctrl+Z on the
canvas undoes it.

A graph with no steps yet offers **Add a step after** its Data Input.
**Data > Transform** adds a scaling, log or power step after the data in use
and picks it.

## Checks

A Check node checks one of:

- a row count (exactly, at least, at most; all rows, or the rows where a
  column is true or has a value);
- the **allowed values** of a column;
- **unique** columns;
- a column **not empty**;
- a numeric column in a **range**.

A failing check stops the run with the count and the first offending value.
Nothing after it runs, and its result from an earlier run is dropped. A
guided step's **Add checks** become these nodes when the step is saved, in
Prepare, in the SQL Workbench, or by switching the operation on in
Properties (they need to be its last operation that is on). The
[Validate job](validate.md) lists them all and adds new ones.

## An image graph's recipe

A graph whose images go through a **Resize** shows its **Image recipe** in
Prepare instead of the table steps. See
[Prepare images with the image recipe](image-recipe.md).

Changed in 1.0.8: the scale, power and log step view, and the image recipe.

## Related

- [SQL Workbench](sql-workbench.md)
- [Check a dataset before training](validate.md)
- [Scaling and transforms](../../reference/nodes/categories/preprocessing.md#scaling-and-transforms)
- [Prepare images with the image recipe](image-recipe.md)
- [Explore a dataset](explore.md)
