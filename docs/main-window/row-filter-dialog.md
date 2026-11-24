---
title: Row Filter dialog
description: The Row Filter node's dialog, where you keep the rows that match conditions written as rows of column, operator and value.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - node-dialogs.md
  - ../user-guide/data-studio/prepare.md
redirect_from: []
---

The **Row Filter** node keeps the rows that match its conditions. Its
dialog writes the conditions as rows, so you do not write SQL. The buttons
every dialog shares are in [Node dialogs](node-dialogs.md).

## Conditions

**Keep the rows where** lists the conditions as rows: a column, an
operator in words and a value.

| Operator | Matches |
| --- | --- |
| *is*, *is not* | Equal or not equal to the value |
| *is more than*, *is less than*, *is at least*, *is at most* | Number comparisons |
| *is empty*, *is not empty* | A missing value, and for text also an empty string (a CSV's empty cell). These take no value. |

Rows after the first join with **and** or **or**; *and* binds first, so
`a and b or c` keeps rows matching both a and b, or c. **Add condition**
adds a row; **Remove** takes one away (a filter keeps at least one).

## Columns and values

Once the step's input is known (the pipeline ran it, or it is a loaded Data
Input), the column is picked by name from its columns, each shown with
what it holds (numbers, text, true/false), and the value follows the
column: a number box, a text box, or a true / false choice.

- Text and true/false columns compare only with *is* / *is not* (and the
  empty tests).
- Columns holding dates or lists are counted but cannot be compared.
- A column with no name (a CSV's index column) is listed as *(unnamed)*
  and cannot be picked.
- Before the first run the column is typed by name, and the value's type
  is read from what is typed (`5` is a number, `true` a true/false,
  anything else text).

## Runs as

**Runs as** shows the condition the step runs, for example
`"status" != 'Normal' AND "statement" IS NOT EMPTY`, and after a run how
many rows passed (*36,338 of 53,043 rows pass (last run)*, while the
condition is the one that ran). OK and Apply stay off, with the reason on
hover, until every row can run: a column picked, a number where the column
holds numbers, a column the input has.

Reopening the dialog shows the saved condition as rows again. A condition
written by hand with brackets in Properties does not fit rows: the dialog
shows it as text to edit, with **Start over with rows**.

## Related

- [Node dialogs](node-dialogs.md)
- [Prepare data with a recipe](../user-guide/data-studio/prepare.md)
- [SQL Workbench](../user-guide/data-studio/sql-workbench.md)
