---
title: Find missing values and outliers
description: See where a table has no value or values far from the rest in Explore > Missing and Outliers, and add a step that drops, fills, flags or clips them.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - profile.md
  - prepare.md
  - ../graph-editor/fix-missing-values.md
---

**Explore > Missing** and **Explore > Outliers** in Data Studio find the
cells of the picked table that have no value and the values far from the
rest, and add the step that deals with them to the graph. **Data > Explore
> Missing Value Analysis** and **Outlier Detection** open them.

## Find missing values

Missing counts as Profile does: empty cells, plus the texts marked as
missing for a column (**Treat as missing** in
[Profile](profile.md#count-texts-such-as-na-as-missing); the link under the
summary goes there).

| Section | What it shows |
| --- | --- |
| Summary | Missing cells and their share, complete rows and incomplete rows. Penguins (344 rows, 7 columns): 19 of 2,408 cells missing (0.8 %), 333 complete rows, 11 incomplete (3.2 %). |
| Columns | Every column with its missing count, share and a bar; sort by name or count with the headers. Columns with none are listed faint. |
| **MISSING TOGETHER** | Which columns are empty in the same rows, the most common mixes first. Penguins: 9 rows lack only sex; 2 rows lack bill length, bill depth, flipper length, body mass and sex together. |
| The picked column | Click a row: its count, advice for how much is missing, and its rows 50 at a time, numbered from 1 as in the Table Viewer (Previous / Next) |

The counts come from Profile, so switching views does not compute them
again. Complete rows, the mixes and the row pages are SQL on the dataset
(tables kept on disk work the same way) and show in **Background Tasks**.

### Drop or fill a column's missing values

1. In **Missing**, click the column's row.
2. Choose what to do:
   - **Drop these rows in Prepare** adds a guided step *Keep rows with
     <column>* after the dataset's node: it keeps rows where the column is
     not empty and is not one of its missing texts. It needs the table to
     be a Data Input or a node result of the open graph.
   - Pick what to fill with in **Fill with**, then click **Fill in
     Prepare**. It adds a Fill Missing Values step on the column: a number
     column takes its mean, median or mode, or
     a typed constant; any other column its mode or a constant. The step
     learns the statistic from the Train rows and fills the empty cells of
     every row. Texts marked as missing are values to it: drop those rows,
     or make them empty first.
   - **Open in Query** puts the SQL for these rows in Query and runs it.
3. Prepare lists the new step, and training reads its result. Ctrl+Z on the
   canvas undoes it.

**General recommendations** (folded) lists ways to fill values by kind of
data.

## Find outliers

Outliers are values far from the rest of a number column. Pick the
**Column** (searchable, for wide tables), the **Rule** and the **Factor**.
The count follows a change at once (the factor when you leave the field or
press - / +).

| Rule | Default factor | An outlier is | Score |
| --- | --- | --- | --- |
| **IQR** | 1.5 | Outside Q1 - k x IQR and Q3 + k x IQR | How many IQRs past the box |
| **Z-score** | 3 | Outside mean ± k x std | z |
| **Modified Z** | 3.5 | Outside median ± k x MAD / 0.6745 (robust to the outliers themselves) | 0.6745 x (x - median) / MAD |

A rule with no spread on the column (Q1 = Q3, all values equal, or MAD = 0)
says so instead of counting every value. Tables profiled approximately use
estimated quartiles and say so.

| Section | What it shows |
| --- | --- |
| Summary | How many outliers, low and high, and the bounds |
| Distribution | The column's histogram with the bounds dashed and the bins past them in orange; a note says when a bound lies beyond the values |
| Rows | Row number, score, low or high, the column checked, then the others; largest score first, 50 a page. **Open in Query** puts the SQL for these rows in Query. |
| Figures | The bounds, mean, median, std, MAD (Modified Z), Q1, Q3, values and missing, and what the rule does |

Penguins: no outliers at IQR 1.5 in any column. Z-score 2 finds 9 in
`body_mass_g` (all male Gentoo from Biscoe, 5,850 to 6,300 g); Modified Z 2
finds 8 in `flipper_length_mm` (MAD 11, above 229.6).

### Flag, clip or drop outliers

Under **In Prepare**, click one of:

| Button | Adds an Outlier Detector step that |
| --- | --- |
| **Flag rows** | Adds an `is_outlier` column, which training then reads as one more feature |
| **Clip values** | Pulls every value into the bounds |
| **Drop rows** | Removes the rows outside the bounds |

The step uses this rule and factor on the column. It learns the bounds from
the Train rows (every row without a Data Split) and applies them to every
row, so its count can differ from the one found here, which reads the whole
table. Training reads the step's result.

## Related

- [Profile a dataset](profile.md)
- [Prepare data with a recipe](prepare.md)
- [Fix missing values before training](../graph-editor/fix-missing-values.md)
- [Correlation and distribution](correlation-and-distribution.md)
