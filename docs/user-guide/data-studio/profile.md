---
title: Profile a dataset
description: Read a dataset's profile in Explore > Profile, set column roles, mark texts as missing, and measure an image dataset's channels and quality.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - missing-values-and-outliers.md
  - validate.md
  - quality-analyzer.md
---

**Explore > Profile** in Data Studio describes the picked table and decides
what each column is. It runs in the background as soon as a dataset is
picked, and again when the dataset is loaded again; **Profile again** runs
it by hand. **Data > Dataset Statistics** opens it.

The Table Viewer's **Stats** use the same profile: a dataset profiled in
either one opens in the other from the saved answers (MNIST: about a minute
the first time, about a second after that).

## Read the profile

The first line gives the rows, the columns, when the table was profiled and
how long it took, and *exact*, or *approximate* on very wide tables (beyond
64 columns), where distinct counts and quartiles are estimated.

The second line counts the columns per role, then **missing cells**,
**duplicate rows** and **data quality** (100 minus twice the share of
missing cells).

The table has a row per column: type, **role**, where the role comes from,
the values (range, dates from - to, or distinct values), missing (count and
share), mean, std, median and outliers (outside 1.5 times the interquartile
range).

## Set a column's role

Each column has one role: **ID**, **Group**, **Target**, **Numeric**,
**Category**, **Date**, **Text**, **File path**, **Weight** or **Ignore**.
**Group** marks rows that belong together and must stay in one split (a
family of passages, a document, a patient); it is never inferred, so set it
here.

1. In the profile table, click the column's role.
2. Pick another role, or **Back to inferred**.

The *from* column says who decided, in this order of precedence:

| From | Meaning |
| --- | --- |
| **contract** | The graph: the Data Input's label column is the Target. It cannot be changed here; change the label in the Data Input. |
| **you** | Set here. Saved with the project in `datasets/column_roles.json` for this file, and used by dashboards, plots and Visualize. |
| **inferred** | Guessed from the values: text unique per row, or a name ending in `_id`, is an ID; dates are Date; yes/no and numbers with up to 12 values are Category; long or mostly different text is Text; image or audio file names are File path; other numbers are Numeric. |

When the table is a Data Input of the graph, the roles you set are also
written into that node (`column_roles`; Ctrl+Z on the canvas undoes it), and
training keeps every ID, Group, Weight and Ignore column out of the
features. The line under the table names them. Group is not used to split,
and Weight is not used to weigh rows. A table that is not a Data Input says
its roles guide dashboards and plots only.

Setting a different role on the graph's target shows a warning: training
keeps the Data Input's label.

Example (Spotify, 8,582 rows): `track_id` and `album_id` are ID;
`album_type`, `explicit`, `artist_name` and `artist_genres` are Category;
`track_name` and `album_name` are Text; `album_release_date` is Date
(1952-09-12 to 2025-10-31); the rest are Numeric.

## Look at one column

Click a column for its distribution (a histogram for numbers, the most
frequent values otherwise) and its figures: non-missing, missing, distinct,
min, quartiles, max, mean, std, outliers, text length and type. The
strongest correlations between number columns (Pearson) are listed under
the details (Spotify: `artist_popularity` and `artist_followers` +0.64).

### Count texts such as N/A as missing

Some files write missing values as text (`N/A`, `-`, `none`).

1. In the column's details, type the texts in **Treat as missing**, comma
   separated. When a frequent value looks like one, the panel suggests it.
2. Click **Apply**. The profile counts them as missing (saved with the
   project for this file).
3. Optional: click **Make them empty in Prepare**. It adds an SQL step after
   the data in use that turns those texts into empty cells, so Missing's
   Fill and Drop steps and training see them as missing.

Until step 3, the cells still hold the texts: Explore and dashboards count
them as missing, the graph does not. Spotify's `artist_genres` has `N/A` in
3,361 rows (39.2 %); treated as missing, the table has 2.6 % missing cells.

## An image dataset's profile

Picking an image dataset (its Data Input in the rail) shows its own profile,
built from its [index table](bring-in-data.md#the-image-index); the Data
Input dialog's Profile tab shows the same. Nothing is decoded for it.

| Section | What it shows |
| --- | --- |
| Tiles | Images, classes, size on disk, unreadable files, and the colour (*RGB, all 25,000*, or *Mixed* with the count of each) |
| **Classes** | Each class's images and share, from the labels CSV or the class folders, and whether they are balanced (*Imbalanced: the largest class is 4.0x the smallest*, or *A class has no images*) |
| **Original size** | Width and height range and median, square images, and what the graph's Resize does with them (*exact: stretched to 64 x 64, aspect is not kept. 41 images are smaller than 64 x 64 on a side and are enlarged*); a graph without Resize says it needs one. Width and height histograms. |
| **Channel statistics** | The mean and standard deviation of each colour channel (see the next section) |
| **Quality** | The Quality Analyzer's measurements (see [Measure image quality](#measure-image-quality)) |
| **Prepared sets** | The sets in `cache/images` made from this folder (size, resize mode, the transforms done once, size on disk), with the one this graph trains from marked *used by this graph*. Sets made by older Engine versions are not listed; the first training run makes a new set. |

### Set Normalize from the channel statistics

**Channel statistics** are the mean and standard deviation of each colour
channel over every pixel of the prepared set this graph trains from (after
the Resize and the transforms done once), measured on the ArrayFire device,
with what the graph's Normalize makes of them (*Normalize uses mean 0.5, std
0.5: the model's inputs centre at -0.02 (R), -0.09 (G), -0.17 (B) with
spread 0.51, 0.50, 0.50*).

1. Click **Measure**. When training has not prepared the set yet, it is
   prepared first (17 s for the 25,000 cats vs dogs images); training then
   reuses it.
2. Click **Use in Normalize**. The graph's Normalize is set to Per channel
   with these numbers (Undo takes it back), and the line reads *Normalize
   uses these statistics: the model's inputs centre at 0.00 (R, G, B) with
   spread 1.00*.

The figures clear when the graph's image steps change. See
[Normalize per channel](../../usage/image_transforms.md#normalize-per-channel).

### Measure image quality

**Measure quality** runs the [Quality Analyzer](quality-analyzer.md)'s
measurements of every image at the Resize size (blur, brightness, contrast,
near duplicates) on the ArrayFire device, judges them with the default
checks (blur at least 700, brightness 50 to 220, contrast at least 0.12,
duplicates within 4 bits) and saves them, so they are found again next time.
The section then shows how many images fail, a bar per reason and the count
per class. For the 25,000 cats vs dogs images it takes under a minute.

Nothing is left out of training: only a Quality node on the canvas does
that. Once measured, the index gets the quality columns and Images its
quality filters.

Changed in 1.0.8: the image dataset's profile, channel statistics and
quality measurement.

## Related

- [Explore a dataset](explore.md)
- [Find missing values and outliers](missing-values-and-outliers.md)
- [Look at an image dataset](explore-images.md)
- [Quality Analyzer](quality-analyzer.md)
- [Table Viewer](../../main-window/table-viewer.md)
