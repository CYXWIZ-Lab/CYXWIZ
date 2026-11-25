---
title: Node dialogs
description: The windows that set up nodes which read, preview or prepare data, the buttons they share, and where each dialog is described.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - properties.md
  - data-input-dialog.md
  - ../user-guide/graph-editor/node-properties.md
  - ../user-guide/graph-editor/compile-and-run.md
redirect_from:
  - node-dialogs.md
---

Nodes that read, preview or prepare data are set up in a dialog of their
own; the rest are set in the [Properties panel](properties.md). Open a
node's dialog by double-clicking the node, right-clicking it and choosing
**Configure...**, or clicking **Open Dialog...** at the top of Properties
(Data Studio has **Open dialog**). The node's settings are saved into the
graph when you press **Apply** or **OK**. Double-clicking any other node
only selects it.

## The dialogs

| Node | Dialog title | Described in |
| --- | --- | --- |
| Data Input | **Data Input** | [Data Input dialog](data-input-dialog.md), [Data Input formats](data-input-formats.md) |
| Data Split | **Data Split** | [Data Split and Data Loader dialogs](data-split-and-loader-dialogs.md) |
| Data Loader | **Data Loader** | [Data Split and Data Loader dialogs](data-split-and-loader-dialogs.md) |
| Tokenizer | **Text Tokenizer Configuration** | [Text Tokenizer and Embedding dialogs](text-node-dialogs.md) |
| Vocabulary, Padding (older graphs) | **Text Vocabulary Configuration**, **Text Padding Configuration** | [Text Tokenizer and Embedding dialogs](text-node-dialogs.md) |
| Embedding | **Embedding Configuration** | [Text Tokenizer and Embedding dialogs](text-node-dialogs.md) |
| Row Filter | **Row Filter** | [Row Filter dialog](row-filter-dialog.md) |
| Data Output | **Data Output** | [Data Output and Data Convert dialogs](data-output-and-convert-dialogs.md) |
| Data Convert | **Data Convert** | [Data Output and Data Convert dialogs](data-output-and-convert-dialogs.md) |
| SQL Query | **SQL Workbench** | [SQL Workbench](../user-guide/data-studio/sql-workbench.md) |
| Quality Analyzer | **Quality Analyzer** | [Quality Analyzer](../user-guide/data-studio/quality-analyzer.md) |
| Word Embeddings | **Word Embeddings** | [Word embeddings](../usage/word_embeddings.md) |

The Plot and Dashboard nodes open windows of their own instead (**Open Plot
window**, **Open Dashboard**); see
[Visualization nodes](../reference/nodes/categories/visualization.md).

## The buttons every dialog shares

| Button | Does |
| --- | --- |
| **Cancel** | Puts back the settings the node had when the dialog opened and closes it. Always available, also while the dialog is loading. |
| **Apply** | Saves the settings and keeps the dialog open. |
| **OK** | Saves and closes. |

While a dialog is loading or running something (a Data Input load, a Data
Convert run), Apply and OK are off; hover them to see why. Dialogs that
know when you changed something (Embedding, Word Embeddings) keep Apply off
until you do ("Nothing changed").

A file field is a path box with a browse button beside it (**Browse
file...**, **Browse folder...**, **Browse...** or **Save as...**); typing a
path works as well as browsing.

## Related

- [Properties panel](properties.md)
- [Set node properties](../user-guide/graph-editor/node-properties.md)
- [Compile and run a graph](../user-guide/graph-editor/compile-and-run.md)
