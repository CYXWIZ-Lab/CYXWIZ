---
title: Run a plugin's analytics
description: Answer questions about a dataset with the analytics installed plugins add, in Data Studio's Explore > Plugins.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - ../../usage/extension_nodes.md
  - ../../sdk/plugin-api-reference.md
---

**Explore > Plugins** in Data Studio runs the analytics that installed
plugins add: questions about a dataset that a plugin answers, for example
*Coefficient of variation* from the SDK's example plugin *Example Data
Formats*. Plugin analytics are not graph nodes: they answer a question and
do not change the data.

With no such plugin loaded, the view says so and offers **Open Plugin
Manager**.

## Run an analytic

1. In **Data Studio > Explore**, pick a dataset in the rail and click
   **Plugins**.
2. In **Analytic**, pick one. Every analytic of every loaded plugin is
   listed, grouped by the category the plugin gives, with the plugin and its
   version beside it and its description under it.
3. Check **Columns**. It starts with every column the analytic reads
   (numbers, text or any, up to the plugin's maximum). Columns it cannot
   read stay in the list, dimmed, with the reason (*text · this analytic
   reads numbers*).
4. Click **Run**. Nothing runs on its own, since the analytic is the
   plugin's code and may read every row. Beside Run: *reads 14 rows of 2
   columns*, or why Run is off (too few or too many columns, or *Example
   Data Formats is not loaded*, with **Open Plugin Manager**).

While it runs, the view shows the rows read so far and **Cancel**;
**Background Tasks** shows the run. The chosen columns go to the plugin in
batches, every row: a table kept on disk is read one row group at a time,
so it does not have to fit in memory. The plugin sees only those columns.

## Read the answer

- The plugin's single number (when it gives one) large, its values per label
  as bars with the value printed, its note, and a line such as *Computed by
  Example Data Formats 1.0.0 on 14 rows in 3 ms, 11:24:05*.
- **Values**: the same numbers as a table, with **Copy** and **Export
  CSV...**.
- Changing the analytic or the columns keeps the answer, dimmed, under
  *Choices changed since this answer*, with **Run again**. When the plugin
  is unloaded, the answer stays dimmed and Run is off.
- A plugin that fails, throws or crashes in its analytic shows its message;
  the Engine keeps running.

## Write your own analytic

A C++ plugin adds analytics through `IAnalyticsProvider`; see the
[plugin API reference](../../sdk/plugin-api-reference.md).

## Related

- [Explore a dataset](explore.md)
- [Your own nodes and plugins](../../usage/extension_nodes.md)
- [Plugin API reference](../../sdk/plugin-api-reference.md)
