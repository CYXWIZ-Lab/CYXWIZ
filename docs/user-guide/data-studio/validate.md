---
title: Check a dataset before training
description: Run and add the graph's checks in Data Studio's Validate job, and compare the train, validation and test splits for leakage and balance in ML readiness.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - prepare.md
  - publish-a-dataset.md
  - split-images-and-audio.md
  - profile.md
---

Two Data Studio jobs check the data before a model trains on it:
**Validate** runs the graph's Check nodes, and **ML readiness** compares the
splits the model trains and is scored on. Neither changes the data; a
failing check stops a run.

## Read the checks

**Validate** lists the graph's checks in one table. The line above it says
*4 of 4 checks pass*.

| Column | What it shows |
| --- | --- |
| **check (node)** | The Check node's name. Double-click it, or **Show**, to show the node on the canvas. |
| **reads** | The step whose rows it checks, past any checks in front of it |
| **rule** | The rule in words: *exactly 15,349 rows*, *exactly 3,712 rows where source_id = pg41633*, *role is one of book, chapter*, *paragraph_id is unique*, *text is never empty*, *word_count from 1 to 900* |
| **last run** | *pass - 15,349 rows* (a row count shows what it counted), *pass*, *fail* with the message under the rule, *could not run* (for example a column that is not there), or *not run* (not run yet in this session, or after a check that failed) |

**Run checks** runs the Data Pipeline through every check and the steps
before them. An export after the checks is not written.

When the graph cannot run, a red line under the summary says why and **Run
checks** is off. In a narrow Data Studio, what a check reads goes under its
rule, and **Edit** and **Show** under its last run.

## Add a check

1. In **Validate**, click **Add a check** and pick **Row count**, **Allowed
   values**, **Unique**, **Not empty** or **Range**. The form opens under
   the table.
2. In **After**, pick the step it checks. It starts on the dataset chosen in
   the rail, else the step the other checks read.
3. Optional: type a **Name**, shown when the check fails (by default the
   rule).
4. Fill in the rule's fields. They pick from the step's columns once it has
   run (type them before). *Rule:* shows the rule as it will read.
5. Click **Add check**. It stays off, with the reason on hover, until the
   form is complete.

The check is added as a Check node right after the step (a collapsed recipe
opens on the canvas; Data Studio stays in front). Ctrl+Z on the canvas
undoes it.

To change a check, click **Edit** in its row: the same form opens on that
node. Change the kind or the rule, then click **Save check**.

## Add a Data Validator

A Data Validator reports the rows that break its rules as an issues table
instead of stopping the run.

1. Click **Data Validator...**.
2. In **Reads**, pick the step.
3. Pick columns for **Required columns**, **No missing value** and **No
   repeated value** (checkboxes once the step has run).
4. Click **Add validator**.

The node reads the step, beside what already reads it; for a step inside a
recipe it reads the recipe's result, since every recipe step must lead to
the recipe's output. Its row shows its rules and *no issues* or *15,345
issues*; the issues table is its node result in the rail. **Edit** opens the
same form on it.

## Compare the splits (ML readiness)

**ML readiness** shows the splits a model trains and is scored on, side by
side, before training.

### Which splits it compares

- A Data Split node's inputs, in pin order: train, validation, test (a graph
  that feeds separate Data Inputs into them, such as a frozen family split).
- Without a Data Split, loaded datasets named `<name>_train`,
  `<name>_validation` (or `_val`, `_valid`, `_dev`) and `<name>_test` make a
  set called `<name>`. Several sets show as chips to pick from.
- A Data Split that splits one input by ratio does that during training; it
  is not previewed here.
- A Data Split input that is not loaded is named in an orange line.
- For an image dataset the job shows the split training will make, from the
  canvas graph's compile, without querying; see
  [Split images and audio](split-images-and-audio.md).

Opening the job compares the splits in the background (**Background Tasks**
shows it); so do a change to a split's data and **Check again**. The job's
line then says what it found (*no leakage*, *leakage found*, *not checked*).

### Read the result

The header names the set and its splits and gives the verdict: *No leakage
found*, *Leakage: 3 groups in two splits* (in red), *Leakage not checked*
(when no Group, ID or Text column is set), or *One split only*. **Show the
Data Split** selects it on the canvas; **Set roles in Profile** opens
Explore > Profile on the train split.

| Card | What it shows |
| --- | --- |
| Roles | The Text, ID, Group and Target columns, read from the train split's roles (yours first, then the graph's, then inferred). Group is never inferred: set it in [Profile](profile.md#set-a-columns-role). |
| Split sizes | Rows per split, distinct groups and IDs; a split whose rows repeat an ID says how many |
| Leakage across splits | Groups, IDs and identical texts found in two splits (exact matches), with the pairs when there are any |
| Text length (words) | Median and mean words per row of the Text column; a split 10 % or more shorter or longer than train is named (*Validation and test are about 14% shorter than train.*) |
| Class balance | With a Target: the largest classes' share per split, the train split's largest and smallest class, a class whose share moves 10 points or more between train and another split, and the Data Loader's balance mode |

Everything is counted with SQL on the whole tables; the data is not changed.
The train split's roles come from its profile, cached in the project's
`cache/query_results` like Explore > Profile's. When several columns read as
Text (a hash column does too), the longest one is used; set Text on another
column in Profile to choose it.

## Related

- [Prepare data with a recipe](prepare.md)
- [Profile a dataset](profile.md)
- [Split images and audio](split-images-and-audio.md)
- [Write the prepared dataset](publish-a-dataset.md)
