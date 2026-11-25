---
title: The Data Studio window
description: The parts of Data Studio (header, rail, jobs, In the graph strip, status bar) and how data enters the Engine through Data Input nodes.
type: reference
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - index.md
  - bring-in-data.md
  - explore.md
  - prepare.md
redirect_from:
  - overview.md
  - overview.md
---

Data Studio is the open graph's data work in one window: bringing data in,
exploring it with SQL, a profile and plots, preparing it, checking it,
getting it ready for training, writing the prepared dataset and reading a
model's results. Open it from the **Data Studio** tab (shown at first, to
the right of CyxWiz Studio) or its icon in the sidebar's **Data** group.

Everything in Data Studio is a node of the graph: what you change shows on
the canvas, and runs go through the Data Pipeline executor.

## The parts of the window

| Part | What it holds |
| --- | --- |
| Header | The graph it works on (and its recipe, when it has one), **Show graph** (back to CyxWiz Studio) and **Publish...** (comes with the Publish job; off until then) |
| Rail (left) | The data, in four lists (see the next table). In a narrow Data Studio (under about 670 pixels) it sits above the jobs instead, so the job keeps the window's width. |
| Jobs | **Bring in**, **Explore**, **Prepare**, **Validate**, **ML readiness**, **Publish**, **Results**, each with a line saying where it stands (*1 Data Input*, *1 recipe*, *4 of 4 pass*, *no leakage*, *1 output*) |
| In the graph | The nodes doing the open job, by kind (*Check x 4*, *Preparation Recipe tier1_paragraphs*, *step nodes Row Filter, ...*). Click one, or **Show on canvas**, to select and frame them in CyxWiz Studio; a collapsed recipe stands for its steps. When the line is too narrow, **Show on canvas** goes under the nodes. |
| Status bar | *Ready*, *Query running*, *Profiling*, *Comparing splits* or *Data Pipeline running*; the last run (*tier1_paragraphs - 15,349 rows - 4/4 checks - 3.1 s - at 12:39:04*, orange when a check failed, red when a step failed); *Runs through the Data Pipeline executor* |

### The rail

| List | Entries |
| --- | --- |
| **Data Inputs** | The graph's Data Input nodes with their rows, or *not loaded* (open the node's dialog and **Apply**, or run the graph) |
| **Node results** | What the last run made: a step's rows, a Row Count Check's *pass* or *fail*, *failed* with the error on hover |
| **Published** | The files the export nodes write, with the rows written or *not written* |
| **Other datasets** | Loaded tables no node of this graph made (a query saved as a dataset, another graph's results) |

Click an entry to show it in Explore; double-click it to show its node on
the canvas. The rail marks the table training reads with a cap (see
[What reaches training](explore.md#what-reaches-training)).

## The jobs

| Job | What you do there | Page |
| --- | --- | --- |
| **Bring in** | Add Data Input nodes and see what they loaded | [Bring data into Data Studio](bring-in-data.md) |
| **Explore** | Query, profile, plot and analyse a table, in 16 views (17 for images) | [Explore a dataset](explore.md) |
| **Prepare** | Edit the graph's Preparation Recipe step by step | [Prepare data with a recipe](prepare.md) |
| **Validate** | List, run and add the graph's checks | [Check a dataset before training](validate.md) |
| **ML readiness** | Compare the train, validation and test splits for leakage and balance | [Check a dataset before training](validate.md#compare-the-splits-ml-readiness) |
| **Publish** | Write the prepared dataset to a file with its lineage | [Write the prepared dataset](publish-a-dataset.md) |
| **Results** | A trained model's test, cross-validation, learning curve and generations | [Read test results](../evaluation/test-results.md) |

## Data enters through a Data Input

Data enters the Engine in one place: a **Data Input** node in CyxWiz
Studio. Every other tool reads what Data Input nodes loaded.

| Command | What it does |
| --- | --- |
| **Data > Import Dataset...**, **Data > Create Custom Dataset...** | Add a Data Input node and open its dialog |
| **Data > Dataset Statistics** | Opens Explore > Profile on the picked dataset |
| **Data > Explore** (**Missing Value Analysis**, **Outlier Detection**, **Correlation Matrix**) | Open the Missing, Outliers and Correlation views |
| **Tools > Statistics** | **Summary Statistics** and **Distribution Fitter** open Distribution, **Hypothesis Testing** opens Tests, **Linear / Polynomial Regression** opens Regression ([Statistics](statistics.md)) |
| **Tools > Evaluation** | Cross-Validation, Confusion Matrix, ROC Curve / AUC, Precision-Recall Curve and Learning Curves open the Results views ([Read test results](../evaluation/test-results.md)) |
| **Tools > Clustering and Projection** | Open the Groups view ([Clustering](clustering.md)) or the Projection view ([Dimensionality reduction](dimensionality-reduction.md)) |
| **Data > Transform** | Adds a scaling, log or power step after that dataset in Prepare ([scaling and transforms](../../usage/data_transforms.md)) |

These used to be separate windows; each now opens the matching Data Studio
view. The Data Explorer, which opened files directly, is gone: Data Studio
covers querying, profiling, exploring and plotting datasets.

### The Dataset list

Data Studio and **Tools > Evaluation > Feature Importance** pick their data
from the same list:

- datasets loaded by Data Input nodes, shown with the node's name
  (*Customer CSV · Data Input · 248 rows · 6 cols*);
- tables made by graph nodes from them when a pipeline runs, shown as
  *node result* with the node's name (for example *PCA (2 components)*).

Picking one reads it in the background, whether it is in memory or on disk.
The tool shows *Reading the dataset...* while it reads, or the reason when it
cannot read it.

Nothing appears until a Data Input node has loaded data. To fill the list,
open the node's dialog and apply it, or run the pipeline (**Execute
Pipeline**, or double-click a Plot node).

## Related

- [Bring data into Data Studio](bring-in-data.md)
- [The Data Input dialog](../../main-window/data-input-dialog.md)
- [What the Engine keeps loaded](../../concepts/loaded-data.md)
- [Data bigger than memory](../../concepts/data-bigger-than-memory.md)
- [Table Viewer](../../main-window/table-viewer.md)
