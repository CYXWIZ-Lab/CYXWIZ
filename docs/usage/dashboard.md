# Dashboard node

A Dashboard shows the data at its place in the graph as a summary and a grid
of widgets that starts by itself and that you can change. It belongs to the
same family as the Plot node: every plot type is a widget.

```
Data Input (Spotify) --> Dashboard
```

Search **Dashboard** on the canvas (or the Visualization group), connect a
table to it, and double-click it (or right-click > Configure...). The node
shows its state underneath: *Not read yet*, *Reading ...*, *Dashboard · 12
widgets*, *Out of date*.

## Data

The line at the top says where the data comes from ("Data at Spotify ·
8,582 rows × 15 columns · read 15:53"), with **Refresh** (or **Cancel**
while reading). Only the nodes above the dashboard run, in the background
(Task View); a loaded Data Input is read directly. When a node above
changes, the dashboard says it is out of date until you refresh; a node that
runs only inside training says so instead of showing old charts.

## The automatic layout

The first time it opens, the dashboard profiles the data (the same profile
as Data Studio's Profile tab) and builds:

- the **target** (the Data Input's label column): a histogram for a number,
  bars for a category;
- a **bar chart per category column** with up to 30 values;
- a **histogram per number column**;
- **rows per year** for a date column (click a year to filter on it);
- a **correlation matrix** of the number columns;
- **missing values**: the share per column under the filters (texts marked
  as missing in Data Studio count), when there are any;
- the **target by its most correlated column** (a scatter, or a box per
  class for a category target).

**Regenerate** rebuilds these automatic widgets from the current data and
roles; widgets you added stay.

A column that only numbers the rows (a CSV's unnamed first column, read as
`C0`, `column0` or `Unnamed: 0`, or one named `index`) is an ID and gets no
widget.

## Text columns

A column with the Text role (long texts, or values that are mostly
different per row, such as reviews or statements) gets its own widgets:

- three KPIs in a row on top: **Words per text (median)**, **Vocabulary
  (words)** and **Empty or missing texts**;
- **Text length (words)**: a histogram, with the longest 1% of texts in the
  last bin so a few very long ones do not squash the rest;
- **Top words** and **Top 2-word phrases**: horizontal bars, common English
  words (the, and, ...) left out; **Keep common words** in the widget's
  settings counts them too;
- **Words by class**: each top word's share of each class's texts (a
  heatmap), when there is a class column with up to 20 values;
- **Sample texts**: the class, then the text.

Words are the text in lower case, split into letters, digits and
apostrophes. Click a word or phrase to keep the texts that have it (as a
whole word: "sleep" does not keep "sleeping"); click a length bin to keep
texts of that many words; the other widgets and the KPIs follow. A widget's
settings choose what it shows, the text column and the class column.

The words are split once per text column, in the background ("Dashboard:
words of <column>" in Task View; the text cards wait for it), and saved in
the project's `cache/dashboard_words` folder. Every card and filter reads
them, and a reopened dashboard loads them instead of splitting again; a
changed source file (new size or time) splits again. Deleting the folder is
safe.

Example: `p3_text.cyxgraph` (test project): the mental-health statements
(53,043 texts, 7 classes) show a median of 62 words and a vocabulary of
60,171 words; clicking "feel" keeps 15,335 texts.

## Sparse features

A Dashboard below a **Count Vectorizer** or **TF-IDF** node with **Output
format: sparse** shows the sparse matrix itself (it runs the vectorizer the
way training does, in the background; Task View shows it):

- KPIs: rows, features, non-zero values, density, memory, labels;
- **Labels**: rows per class (click a bar to keep that class's rows in every
  card; click it again for all rows);
- **Top features**: the largest total weights;
- **Features per row** and **Feature spread** (how many rows use each
  feature, log scale);
- **Top features by class**: the mean weight per row of each class;
- **Rows**: each row's label, how many features it uses and its strongest
  ones.

The figures are exact (read from the matrix, not sampled). A Plot node
cannot draw sparse features and says to connect a Dashboard.

The matrix comes from the same cache training uses (the project's
materialization cache): built the first time, then loaded, so a reopened
dashboard and a training run on the same graph share it.

Example: TF-IDF (2,000 features) on the statements: 1,839,166 non-zero
values, density 1.73%, 15.1 MB.

## Summary strip

Rows (filtered, *of* all rows), columns, missing cells, duplicate rows and
the target's mean (or most frequent value), with *all* for the unfiltered
figure when a filter is on.

## Filters (cross-filtering)

Click a bar, a pie slice or a histogram bin: every **other** widget shows
only those rows (the widget you clicked keeps showing all its values, so you
can pick another). The filter bar lists the conditions ("album_type =
single · 2,219 of 8,582 rows"); click a condition to remove it, or **Clear
all**. Clicking the same bar again removes its filter. Filters are saved in
the node.

Example (Spotify): click *single* in album_type: 2,219 of 8,582 rows, the
target mean 46.36 (all rows 52.36).

Filtered bar charts and histograms show all rows in grey behind the
filtered bars (on the same bins), and the widget you clicked keeps your
selection in full colour with the rest dimmed.

## Fields and roles

The left column lists the fields with their roles (ID, Target, Numeric,
Category, Date, Text, ...). Roles are edited in one place: **Edit roles in
Data Studio** opens Data Studio's Profile tab on this data. The contract
(the Data Input label) comes first, then roles set in Data Studio, then
inferred ones.

## Widgets

- **Add widget**: KPI, Table, Missing values, or any plot type (grouped as in the Plot
  window). A new widget starts with fitting columns.
- Click a widget's title to edit it on the right: title, plot type, the
  columns (X, Y, Colour by), or a KPI's measure (rows, sum, mean, median,
  min, max, distinct, missing) and field, or a table's columns and rows.
  **All settings in the Plot window** opens the full Plot window on the
  same data; changes there go back into the widget.
- **Size and place**: width (1 to 12 columns) and height, **Earlier** /
  **Later** to move it, **Remove widget**.
- **Re-query** runs every widget's query again.

Widgets read only the columns they need, through the shared query service,
and plots are prepared in the background; tables over 1,000,000 rows are
drawn from a reproducible sample of that many rows (the widget says
*sampled*).

## Export, SQL and Data Studio

- **Export PNG** (toolbar) saves the summary strip and widgets as they are
  on screen; scroll first if the widgets you want are below.
- **Open in Data Studio** (toolbar) opens the Query tab on the rows the
  dashboard shows, the dataset with the current filters, and runs it:

  ```sql
  SELECT * FROM "Spotify" WHERE CAST("album_type" AS VARCHAR) IN ('single')
  ```

  (2,219 rows). From there: Table Viewer, Plot, or Save as Dataset.
- **View SQL** (a widget's settings) shows the query that widget runs, with
  the other widgets' filters and the values written in; **Open in Query
  tab** runs it in Data Studio and **Copy** puts it on the clipboard. For
  the target histogram under the filter above:

  ```sql
  SELECT "track_popularity" FROM "Spotify" WHERE CAST("album_type" AS VARCHAR) IN ('single')
  ```

## Widgets from Data Studio

Data Studio's Visualize tab and Query results have **Add to Dashboard**
(see [Data Studio](data_studio.md)). A plot arrives as an ordinary widget.
A query arrives as a **query widget**: it runs its SQL over the rows the
dashboard shows, so the filters apply. For example

```sql
SELECT album_type, count(*) AS tracks, avg(track_popularity) AS popularity
FROM Spotify GROUP BY album_type
```

shows 5,856 / 2,219 / 507 tracks, and only *single* (2,219) when
album_type = single is filtered. Its settings show the query and **Edit in
Query tab**; its bars do not set filters (its columns are the query's).

## Loading and saved results

A card's query runs once per data and filter. Its result is saved in the
project (`cache/query_results`, Parquet, kept under 512 MB, least recently
used first), so reopening the dashboard on the same data shows every card
at once, and a filter you have used before comes back without a query. The
Task View marks such a result "(saved)". A result changes when the data does:
the key holds the query, its values and the content of every table it reads
(the file for a disk-backed dataset), so an edited file or a new load runs
again. Deleting the folder is safe.

Queries run a few at a time (three), cheap cards first: the summary strip
and numbers, then tables, then plots, then the text cards that read every
word. Changing a filter stops the queries it supersedes.

A text column's words are split once per data and saved next to the results
(`cache/dashboard_words`, one list of words per text); every text card and
the word filters read that list, so the first open of a text dataset pays
one pass over the texts and the cards after it work on words, not strings.

## When the data changes

- New values or rows: every widget updates; the layout stays.
- A column a widget uses is gone: the widget says so; when one new column of
  the same type appeared (a rename), **Rebind to** it, or **Remove**.
- A column became text where a widget needs numbers: the widget says so and
  suggests a Bar.
- New columns: existing widgets keep working; **Regenerate** adds cards for
  them.
