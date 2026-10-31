# Data Studio

Data Studio is the open graph's data work in one window: bringing data in,
exploring it with SQL, a profile and plots, preparing it, checking it,
getting it ready for training, publishing it and reading the results. Open
it from the Data Studio tab (or the sidebar). Everything in it is a node of
the graph: what you change shows on the canvas, and runs go through the
Data Pipeline executor.

## The window

- **Header**: the graph it works on (and its recipe, when it has one),
  **Show graph** (CyxWiz Studio) and **Publish...** (comes with the Publish
  job; disabled until then).
- **Rail** (left): the data. In a narrow Data Studio (under about 670
  pixels) it sits above the jobs instead, so the job keeps the window's
  width.
  - **Data Inputs**: the graph's Data Input nodes with their rows, or *not
    loaded* (open the node's dialog and Apply, or run the graph).
  - **Node results**: what the last run made: a step's rows, a Row Count
    Check's *pass* or *fail*, *failed* with the error on hover.
  - **Published**: the files the export nodes write, with the rows written
    or *not written*.
  - **Other datasets**: loaded tables no node of this graph made (a query
    saved as a dataset, another graph's results).

  Click an entry to show it in Explore; double-click to show its node on
  the canvas. Data enters the graph only through a Data Input.
- **Jobs**: **Bring in**, **Explore**, **Prepare**, **Validate**, **ML
  readiness**, **Publish**, **Results**, each with a line saying where it
  stands (*1 Data Input*, *1 recipe*, *4 of 4 pass*, *no leakage*,
  *1 output*).
- **In the graph**: the nodes doing the open job, by kind (*Check
  x 4*, *Preparation Recipe tier1_paragraphs*, *step nodes Row Filter,
  ...*). Click one, or **Show on canvas**, to select and frame them in
  CyxWiz Studio; a collapsed recipe stands for its steps. When the line is
  too narrow, **Show on canvas** goes under the nodes.
- **Status bar**: *Ready*, *Query running*, *Profiling*, *Comparing
  splits* or *Data Pipeline running*; the last run
  (*tier1_paragraphs - 15,349 rows - 4/4 checks - 3.1 s - at 12:39:04*,
  orange when a check failed, red when a step failed); *Runs through the
  Data Pipeline executor*.

## Bring in

**Add Data Input...** adds a Data Input node to the graph and opens its
dialog, where you pick the file, its format and the preview, then Apply.
The table lists the graph's Data Input nodes: the node, its file as written
in the graph, rows x columns once loaded, and its state (*loaded*, *not
loaded*, or *failed* with the error on hover). **Open dialog** opens the
node's dialog on the canvas; **Explore** opens a loaded one in Explore >
Profile. Where the five columns do not fit, each Data Input is listed as a
block with the same facts and links.

An image Data Input shows its folder, where its labels come from (*image
folder, labels from cat_dog.csv* or *image folder, class folders*) and the
image count (*25,000 images*).

### Image types

The Engine reads JPEG (`.jpg` `.jpeg` `.jpe`), PNG, BMP (`.bmp` `.dib`),
TIFF (`.tif` `.tiff`), WebP, GIF (its first frame), the Netpbm types
(`.ppm` `.pgm` `.pbm` `.pnm`), Sun raster (`.sr` `.ras`), Radiance HDR
(`.hdr` `.pic`) and PFM. The same list is used everywhere: the Data Input's
folder scan, the image index, training, Explore > Images and the asset
previews. Other files in an image folder are skipped. TGA, AVIF, HEIC,
JPEG 2000, JPEG XL and EXR are not read: convert them first.

### The image index

Loading an image dataset also builds its **index table**, one row per image,
named after the dataset with `_index` (`cat_dog_index`). It reads only each
file's header (JPEG, PNG and 16/24-bit BMP; other formats are decoded once),
on a background task, so the Data Input is loaded before the index is
ready: the 25,000 cats vs dogs images take 0.6 to 0.9 s when the files are in
the system's file cache and about 33 s from a cold disk. The rail lists it
under its Data Input (*indexing* while it builds). Its columns:

| Column | What |
| --- | --- |
| path | the image file |
| file | its file name |
| label | the class index (classes in sorted order, numbers as numbers) |
| label_name | the class name (the Data Input's class names when set) |
| width, height | the original size in pixels (empty when unreadable) |
| channels | what a decode gives: 1 grey, 3 colour, 4 with alpha |
| bytes | the file size |
| readable | false when the file cannot be read as an image |

Once quality is measured (Profile > Quality) the index has five more columns:
blur, brightness, contrast, verdict (empty when the image passes, else its
reasons, *blurry, low contrast*) and duplicate_of (the file it repeats). They
stay when the index is rebuilt (renamed classes).

It is an ordinary table: Query (`SELECT label_name, COUNT(*) FROM
"cat_dog_index" GROUP BY label_name`), the Explore views and the Table Viewer
all work on it. It is dropped when the images are unloaded and rebuilt when
they are loaded again.

## Explore

Explore has fourteen views: **Query**, **Profile**, **Missing**, **Outliers**,
**Correlation**, **Distribution**, **Tests**, **Regression**, **Groups**,
**Projection**, **Series**, **Forecast**, **Visualize** and **Plugins**, and a fifteenth, **Images**, when an image dataset is picked. A dashboard's "Open in Data Studio", "Edit roles", the Asset
Browser's "Open in Data Studio" and Data > Dataset Statistics open the
right one, as do Data > Explore (Missing Value Analysis, Outlier Detection,
Correlation Matrix) and Tools > Statistics (Summary Statistics and
Distribution Fitter open Distribution; Hypothesis Testing opens Tests; Linear /
Polynomial Regression opens Regression) and Tools > Clustering and Projection
(K-Means, DBSCAN, Hierarchical Clustering and Gaussian Mixture Models open
Groups on that method, Cluster Evaluation opens Groups, PCA / t-SNE / UMAP opens
Projection) and Tools > Time Series (Decomposition, ACF / PACF, Stationarity and
Seasonality open Series on that look; Forecasting opens Forecast). In Profile, a column's missing or outlier count opens
Missing or Outliers on it, "More about this column" opens Distribution, and
"See the matrix" opens Correlation. The views wrap to a second line in a
narrow dock.

### What reaches training

Training reads one table: the result of the node wired into the Data Split's
Training Dataset input (the Data Loader's input when there is no Split). Start
Training takes the canvas run's result of that node, else the prepared data
kept on disk, else it runs the steps before it through the Data Pipeline
executor first. Steps that learn (scalers, encoders, Fill Missing, Outlier
Detector) learn from the Train rows only. So:

- **A step feeds training.** Everything on the way to that node does: every
  Prepare step, Missing's **Drop these rows** and **Fill in Prepare**,
  Outliers' **Flag / Clip / Drop rows**, Groups', Projection's, Series' and
  Forecast's **Add as a step**, Query's **Turn into step**, Images' **Use in
  Normalize**. Their buttons say so on hover, and the message after adding one
  ends with *Training reads its result.*
- **A branch only shows.** **Keep as a node** (Tests, Regression, Series'
  Seasonality / Stationarity / ACF, Groups, Projection, Forecast), **Keep as
  Plot node** and dashboards read the data and keep their result with the
  graph; training does not read them. The message ends with *Training does not
  read a branch.*
- **Column roles reach training through the Data Input.** A role set in
  Profile on a Data Input's table is written into that node (`column_roles`),
  and training keeps ID, Group, Weight and Ignore columns out of the features
  (the compile report says which). Group is not used to split and Weight not
  to weigh rows yet. *Treat as missing* counts in Explore only until **Make
  them empty in Prepare** adds the step that turns the texts into empty cells.
- **Nothing else changes training.** Query's **Save as Dataset**, every
  **Export CSV...**, ML readiness and Validate's table change no step (a
  failing check stops the run).

The rail marks the table training reads with a cap, and Explore opens on it
when nothing is picked. One line above every view says what training makes of
the table in front of you: *Training reads this table through the Data Split.*,
*Training reads "Scale numbers", not this table.* (with **Show what training
reads**), *Training reads "Scale numbers", which has no table yet (run its steps
in Prepare); this is not it.* (with **Show the node**), or *Nothing feeds a Data
Split or a Data Loader yet, so training reads no table.*

## Query

Write SQL that names tables by their names, in double quotes when the name
has spaces or capitals:

```sql
SELECT album_type, count(*) AS tracks, round(avg(track_popularity), 1) AS popularity
FROM "Spotify"
GROUP BY album_type
ORDER BY tracks DESC
```

| album_type | tracks | popularity |
|---|---|---|
| album | 5856 | 55.7 |
| single | 2219 | 46.4 |
| compilation | 507 | 40.5 |

- The editor is the SQL Workbench's: highlighting, completion (type, or
  Ctrl+Space) with the picked dataset's columns, **Ctrl+F** find, and
  **Format** (keywords upper case). Until you type, it follows the dataset
  you pick.
- **Run** (or **Ctrl+Enter** while the editor has focus) runs the query in
  the background: the Engine stays responsive, Task View shows it, and
  **Cancel** stops it. One query runs at a time.
- Several tables in one query: `JOIN "Spotify" s ... JOIN "World countries" w`.
  Large datasets kept on disk (Parquet) can be queried too. How large tables are kept and read: [big_data.md](big_data.md).
- The line under the editor lists the tables you can name now.
- Only reading is allowed: one `SELECT` (or `WITH ... SELECT`). Changing data
  belongs in the graph (the SQL node, Filter Rows, ...). Queries cannot read
  files or reach the network.
- **Examples** fills in queries for the picked dataset, using its real
  columns: first rows, count, rows per value, average per value, a filter,
  columns and their types.
- **Clear** empties the editor.
- **Turn into SQL step** puts the last result's query into the graph: a new
  SQL step after the SQL step whose result it reads, in that step's recipe
  (a collapsed recipe opens). It takes over what read that step (its checks
  and exports read the new step), shows on the canvas (Ctrl+Z there undoes
  it) and fills on the next run. The query is kept as written; a first
  `WITH` part names the step's `input` as the table you queried. It is
  available when the query reads exactly one table and that table is an SQL
  step's result; otherwise the button says why.

The result line says **exact** (every row shown) or **first 100,000 rows**
(more rows than are shown), then rows, columns, the tables read (with their
rows) and the time. The rows are in the table grid: sort by a header, select
and copy cells, `null` for empty cells, a long value whole on hover (see
[Table Viewer and the table grid](table_viewer.md)). Decimals are numbers and
dates stay dates. Then:

- **Open in Table Viewer**: the result as a table you can sort, filter and
  plot from.
- **Plot**: opens the Plot window on the result (a category and a number
  open as bars, two numbers as a scatter, one number as a histogram; change
  the type there).
- **Add to Dashboard**: the query becomes a widget of a dashboard on the
  picked dataset (see Visualize below for the menu). The widget runs the
  query over the dashboard's rows, so its filters apply. A category and a
  number become bars, two numbers a scatter, one number a histogram. The
  query has to read the picked dataset.
- **Save as Dataset**: runs the query again over all rows and saves the
  result under a name; it then appears in the Dataset list and can be named
  in other queries.

Errors are shown under the buttons in plain words (for example "Only reading
is allowed here" or DuckDB's message for a typo); a long DuckDB message
shows its first line, and **Details** the rest.

## Profile

Profile (it replaced Analyze) describes the picked table and decides what
each column is. It runs in the background as soon as a dataset is picked,
and again when the dataset is re-loaded; **Profile again** runs it by hand.
The Table Viewer's Stats use the same profile: a dataset profiled in either
one opens in the other from the saved answers (MNIST: 62 s, then 1.3 s).

The first line: rows, columns, when it was profiled and how long it took,
*exact* (or *approximate* on very wide tables, beyond 64 columns, where
distinct counts and quartiles are estimated). The second line counts the
columns per role, then **missing cells**, **duplicate rows** and **data
quality** (100 minus twice the share of missing cells).

The table has a row per column: type, **role**, where the role comes from,
the values (range, dates from - to, or distinct values), missing (count and
share), mean, std, median and outliers (outside 1.5 times the interquartile
range).

### An image dataset's profile

Picking an image dataset (its Data Input in the rail) shows its own profile,
built from its index table; the Data Input dialog's Profile tab shows the
same. Nothing is decoded for it.

- The tiles: images, classes, size on disk, unreadable files, and the colour
  (*RGB, all 25,000*, or *Mixed* with the count of each).
- **Classes**: each class's images and share, from the labels CSV or the
  class folders, and whether they are balanced (*Imbalanced: the largest
  class is 4.0x the smallest*, or *A class has no images*).
- **Original size**: width and height range and median, square images, and
  what the graph's Resize does with them (*exact: stretched to 64 x 64,
  aspect is not kept. 41 images are smaller than 64 x 64 on a side and are
  enlarged*); a graph without Resize says it needs one.
- Width and height histograms.
- **Channel statistics**: the mean and standard deviation of each colour
  channel over every pixel of the prepared set this graph trains from (after
  the Resize and the transforms done once), measured on the ArrayFire device,
  and what the graph's Normalize makes of them (*Normalize uses mean 0.5, std
  0.5: the model's inputs centre at -0.02 (R), -0.09 (G), -0.17 (B) with
  spread 0.51, 0.50, 0.50*). **Measure** prepares the set first when training
  has not (17 s for the 25,000 cats vs dogs images); training then reuses it.
  The figures clear when the graph's image steps change. **Use in Normalize**
  sets the graph's Normalize to Per channel with these numbers (Undo takes it
  back); the line then reads *Normalize uses these statistics: the model's
  inputs centre at 0.00 (R, G, B) with spread 1.00*. See
  [Normalize per channel](image_transforms.md#normalize-per-channel).
- **Quality**: the Quality Analyzer's measurements of every image at the Resize
  size (blur, brightness, contrast, near duplicates), judged with the default
  checks (blur at least 700, brightness 50 to 220, contrast at least 0.12,
  duplicates within 4 bits): how many fail, a bar per reason, and the count per
  class. **Measure quality** runs it on the ArrayFire device (40 s for the 25,000
  cats vs dogs images here: 2,369 fail, 1,700 blurry, 864 low contrast, 157 dark,
  30 bright, 48 duplicates) and saves it, so it is found again next time.
  Nothing is left out of training: only a Quality node on the canvas does that.
  Once measured, the index gets the quality columns and Images its quality
  filters.
- **Prepared sets**: the sets in cache/images made from this folder (size,
  resize mode, the transforms done once, size on disk), with the one this
  graph trains from marked *used by this graph*. Sets made before
  2026-10-29 are not listed: their class order is the old one, and the
  first training run makes a new set.

### Column roles

Each column has one role: **ID**, **Group**, **Target**, **Numeric**,
**Category**, **Date**, **Text**, **File path**, **Weight** or **Ignore**.
**Group** marks rows that belong together and must stay in one split (a
family of passages, a document, a patient); it is never inferred, so set it
here. The *from* column says who decided:

- **contract**: the graph. The Data Input's label column is the Target. It
  cannot be changed here; change the label in the Data Input.
- **you**: set here. Click the role and pick another (or **Back to
  inferred**). Saved with the project in `datasets/column_roles.json` for
  this file, and used by dashboards, plots and Visualize. When the table is
  a Data Input of the graph, the roles are also written into that node
  (`column_roles`, Ctrl+Z on the canvas undoes it), and training keeps every
  ID, Group, Weight and Ignore column out of the features; the line under
  the table names them (Group is not used to split and Weight not to weigh
  rows yet). A table that is not a Data Input says its roles guide
  dashboards and plots only.
- **inferred**: guessed from the values: text unique per row is an ID, a
  name ending in `_id` is an ID, dates are Date, yes/no and numbers with up
  to 12 values are Category, long or mostly different text is Text, image or
  audio file names are File path, other numbers are Numeric.

Contract first, then yours, then inferred. Setting a different role on the
graph's target shows a warning: training keeps the Data Input label.

Example (Spotify, 8,582 rows): track_id and album_id are ID, album_type,
explicit, artist_name and artist_genres are Category, track_name and
album_name are Text, album_release_date is Date (1952-09-12 to 2025-10-31),
the rest Numeric. Profiled in 0.3 s.

### A column's details

Click a column for its distribution (a histogram for numbers, the most
frequent values otherwise) and its figures (non-missing, missing, distinct,
min, quartiles, max, mean, std, outliers, text length, type).

**Treat as missing**: some files write missing values as text (`N/A`, `-`,
`none`). Type them, comma separated, and **Apply**: the profile counts them
as missing (saved with the project for this file). When a frequent value
looks like one, the panel suggests it. Spotify's artist_genres has `N/A` in
3,361 rows (39.2%); treated as missing, the table has 2.6% missing cells.
The cells still hold the texts, so Explore and dashboards count them as
missing but the graph does not: **Make them empty in Prepare** adds an SQL
step after the data in use that turns those texts into empty cells, and
Missing's Fill and Drop steps and training then see them as missing.

The strongest correlations between number columns (Pearson) are listed
under the details (Spotify: artist_popularity and artist_followers +0.64).

## Images

For an image dataset (its Data Input in the rail), **Images** shows its
images as a **Grid**, a **List** or one at a time in the **Viewer** (the
first buttons of its toolbar). It works for any image dataset: the facts
come from the dataset's index table.

The chips filter by class (*All*, *Class 0*, *Class 1*, or the class names)
and show the unreadable files when there are any; once quality is measured,
also *Fails a check* and one chip per reason (*blurry*, *low contrast*,
*dark*, *bright*, *duplicate*). The filter holds in all three views. The
arrows at the right turn the pages (*1 to 60 of 25,000*).

**Grid.** The tiles fill the panel and scroll. **Tile** sets their size
(64 to 256 px, typed or with - / +); **Fill** crops each image to its
square, **Fit** shows it whole. **Show** chooses what a tile carries: its
**Class** (bottom left), **Name** (bottom right), **Size** in pixels (top
left) and **Flags**, the first failed check (top right, *blurry +1*).
**Sort** puts them in dataset order, smallest first, largest first (by pixel
count, unreadable files last) or, once measured, blurriest first. Hovering
shows the file name.

Clicking a tile picks it (a ring and a wash): the panel at the right shows
it larger with its row, label, original size and colour, size on disk and
path (and, when measured, its blur, brightness, contrast, verdict and the
image it duplicates), **Open in viewer**, and the links **Show in folder**
and **Open the index table** (Query on `<dataset>_index`). The arrow keys
move the pick (Up and Down a row), Enter or a double-click opens it in the
viewer. Thumbnails are read on two background threads, the ones on screen
first.

**List.** A row per image: thumbnail, file, class, size, channels, size on
disk and quality (*not measured* until it is). Click a column to sort by it
(File by name, Class, Size smallest or largest first, On disk largest first,
Quality blurriest first); a third click goes back to dataset order.
Double-click a row to open it in the viewer.

**Viewer.** One image at full size; nothing here changes the file.

- **Zoom**: **Fit** (the whole image, enlarging a small one), **100 %**, a
  typed percent, and - / + through steps from 6 % to 1600 %. The mouse wheel
  zooms around the cursor, a drag pans, a double-click goes between Fit and
  100 %.
- **Channel**: RGB, R, G or B alone, Luminance (as the Grayscale node:
  0.2989 R + 0.587 G + 0.114 B), and Alpha when the image has one.
- **View**: rotate left or right, **Flip** left to right, **Pixel grid**
  (lines between pixels from 800 %), **Checkerboard** behind transparent
  pixels.
- **Compare**: **Original**, **Model sees** (the image after this graph's
  fixed image steps, the size the model gets, *Gaussian Blur 3 x 3, sigma
  0.8 · 546 %*) or **Side by side**; each fitted to its half.
- Under the picture, the filmstrip (the images before and after it in the
  current order; click one to go to it) and the status bar: the pixel under
  the cursor (*x 189 y 190 · R 151 G 150 B 146 · #979692*), and the image's
  size, colour type, bits per channel and zoom.
- At the right: a **Histogram** per channel (red, green, blue, luminance,
  alpha when there is one; 32 bins of 0 to 255, with the mean and standard
  deviation; the chosen channel stays bright), **Around the cursor** (the
  12 x 8 pixels around the pointed one, one cell each), and the image's
  details.
- The arrows by the file name (or Left and Right) go to the previous and
  next image; Esc goes back to the grid or list.

The images are read with OpenCV, as training reads them: the pixels as
stored, without turning a photo by its EXIF orientation (a phone photo
taken sideways shows sideways, as the model gets it). 16-bit and float
images are scaled to 8 bits for the screen (the status bar names the file's
own depth).

**What the model sees**, folded under the grid (click it to open), runs the picked image through this
graph's image steps with the code training runs, on the ArrayFire device:
the Resize (*Resize 72 x 72, exact, once*), each transform done once with the
images (*Center Crop 64 x 64*, *Gaussian Blur 3 x 3, sigma 0.8*), then four
draws of the random transforms done per batch, each saying what it drew
(*flipped, rotated 7.5 degrees*). A graph without random transforms says every
epoch sees the same images. Normalize is left out so the pictures stay
viewable. It runs again when the graph's image steps change.

## Missing

Where the picked table has no value. It counts as Profile does: empty cells,
plus the texts marked as missing for a column (**Treat as missing** in
Profile; the link under the summary goes there).

- **Summary**: missing cells and their share, complete rows and incomplete
  rows. Penguins (344 rows, 7 columns): 19 of 2,408 cells missing (0.8%),
  333 complete rows, 11 incomplete (3.2%).
- **Columns**: every column with its missing count, share and a bar; sort by
  name or count with the headers. Columns with none are listed faint.
- **Missing together**: which columns are empty in the same rows, the most
  common mixes first. Penguins: 9 rows lack only sex; 2 rows lack bill
  length, bill depth, flipper length, body mass and sex together.
- **The picked column** (click a row): its count, advice for how much is
  missing, and its rows 50 at a time, numbered from 1 like the Table Viewer
  (Previous / Next).
  - **Drop these rows in Prepare** adds a guided step "Keep rows with
    <column>" after the dataset's node (Keep rows where the column is not
    empty and is not one of its missing texts). Prepare lists it; Ctrl+Z on
    the canvas undoes it. It needs the table to be a Data Input or a node
    result of the open graph.
  - **Open in Query** puts the SQL for these rows in Query and runs it.
  - **Fill with ... / Fill in Prepare** adds a Fill Missing Values step on
    the column: a number column takes its mean, median or mode, or a typed
    constant; any other column its mode or a constant. The step learns the
    statistic from the Train rows and fills the empty cells of every row.
    Texts marked as missing are values to it: drop those rows, or make
    them empty first.
  - **General recommendations** (folded) lists ways to fill values by kind
    of data.

The counts come from Profile's profile, so switching tabs does not compute
them again; complete rows, the mixes and the row pages are SQL on the
dataset (tables kept on disk work the same way) and show in Task View.

## Outliers

Values far from the rest of a number column.

- **Column** (searchable, for wide tables), **Rule** and **Factor**. The
  count follows a change at once (the factor when you leave the field or
  press - / +).
  - **IQR** (factor 1.5): outside Q1 - k x IQR and Q3 + k x IQR; the score
    is how many IQRs past the box.
  - **Z-score** (3): outside mean +- k x std; the score is z.
  - **Modified Z** (3.5): outside median +- k x MAD / 0.6745, robust to the
    outliers themselves; the score is 0.6745 x (x - median) / MAD.
  - A rule with no spread on the column (Q1 = Q3, all values equal, or
    MAD = 0) says so instead of counting every value.
- **Summary**: how many outliers, low and high, and the bounds.
- **In Prepare**: **Flag rows**, **Clip values** and **Drop rows** add an
  Outlier Detector step on the column with this rule and factor, and the
  action flag (an `is_outlier` column, which training then reads as one more
  feature), clip (every value pulled into the bounds) or remove (the rows
  outside dropped). The step learns the bounds
  from the Train rows (every row without a Data Split) and applies them to
  every row, so its count can differ from the one found here, which reads
  the whole table. Training reads the step's result.
- **Distribution**: the column's histogram with the bounds dashed and the
  bins past them in orange; a note says when a bound lies beyond the values.
- **Rows**: row number, score, low or high, the column checked, then the
  others; largest score first, 50 a page. **Open in Query** puts the SQL for
  these rows in Query.
- **Figures**: the bounds, mean, median, std, MAD (Modified Z), Q1, Q3,
  values and missing, and what the rule does.

Penguins: no outliers at IQR 1.5 in any column. Z-score 2 finds 9 in
body_mass_g (all male Gentoo from Biscoe, 5,850 to 6,300 g); Modified Z 2
finds 8 in flipper_length_mm (MAD 11, above 229.6). Tables profiled
approximately use estimated quartiles and say so.

## Correlation

How the number columns move together: Pearson r of every pair, over the rows
where both values are present.

- **The matrix**: the profile's first 30 number columns (more need picking in
  **Columns**, a searchable list with chips). Each cell shows r; hover for
  the pair, r, its strength in words and how many rows it used; click to
  pick it.
- **The picked pair** (right): r, a sentence ("very strong positive: when
  one is larger, so is the other"), and the rows it used.
- **Strongest pairs**: every pair by |r|, strongest first; click one to pick
  it.
- **Display**: cell size, values in the cells, the -1 .. +1 legend.
- **Export CSV...** saves the matrix (a header of the columns, one row each).

Penguins: flipper length and body mass +0.87; bill depth and flipper length
-0.58; every pair uses 342 of 344 rows (2 rows have no measurements).

## Distribution

How one number column is spread, and which distribution it follows.

- **Column**, **Bins** (5 to 100; the histogram follows when you leave the
  field) and **Export CSV...** (the figures, percentiles and fits).
- **Summary**: values, missing, skewness in words, and whether any
  distribution fits.
- **Histogram** with the chosen fit drawn over it (expected count per bin),
  a **box plot** (box Q1 to Q3, line the median, diamond the mean, whiskers
  to the last value within 1.5 IQR, dots beyond) and a **Q-Q plot** against
  the chosen fit (points on the line: the values follow it).
- **Figures**: count, mean, median, mode (and how often), std, variance,
  SEM, min, max, range, sum, Q1, Q3, IQR, skewness, kurtosis (excess) and
  CV; hover a name for what it means. **Percentiles** 5 to 95.
- **Fits**: Normal, Uniform, and for columns of positive values Exponential
  and Log-Normal, best AIC first, with BIC, the KS statistic and its
  p-value; a check mark when p is above 0.05. Click a fit to draw it; its
  parameters and log-likelihood show below.

The figures are SQL on the dataset (approximate quantiles on tables the
profile reads approximately). The fits use the column's values; a column of
more than 200,000 values is fitted on a 200,000-value sample and says so.

Penguins: bill_length_mm fits Normal (KS p 0.066); body_mass_g fits nothing
(best Log-Normal, p 0.018): the three species mixed give it more than one
hump. Its mode is 3,800 g (12 birds), skewness 0.47, excess kurtosis -0.72.

## Tests

A question about the data: two groups (Welch by default), paired columns,
several groups (with Tukey HSD), one column vs a value, two categories
(chi-square). The answer shows the groups and box plots, the figures with
effect sizes in words, a sentence, and its checks; **Keep as a node** adds a
Hypothesis Test node. See [statistics](statistics.md#tests).

## Regression

Predict a number column from others, or a curve (degree 1 to 6) in one:
coefficients with CIs, the equation, R², AIC, what the fit says, and the
residual plots; **Add to the graph** adds a Linear or Polynomial Regression
node. See [statistics](statistics.md#regression).

## Groups

Which rows group together. Pick the number columns (every number column, up
to eight, to start), **Scale** (Standardise turns each column into z-scores,
so a column in large units such as grams does not decide the groups alone; As
they are uses the raw values) and the **Method**, then its settings; every
number is typed:

| Method | Settings |
| --- | --- |
| K-Means | groups (2 to 50), seed |
| DBSCAN | eps, min samples (the row itself counts), metric (euclidean, manhattan, cosine) |
| Hierarchical | groups (where the tree is cut), linkage (ward, complete, average, single); metric when not ward |
| Gaussian mixture | components, seed, covariance (full, tied, diag, spherical) |

The same seed gives the same groups. Rows missing one of the columns are left
out and counted. The answer shows:

- the scores: silhouette (with its words: above 0.70 strong, 0.50 reasonable,
  0.25 weak structure), Davies-Bouldin, Calinski-Harabasz, and the rows used;
  what the method says about itself (K-Means inertia and iterations, the
  tree's cophenetic correlation, the mixture's BIC and AIC);
- the rows on their first two principal components, coloured by group (DBSCAN's
  noise faint), with each group's size;
- **What each group is**: the mean of each column per group, in the column's own
  units, each group's silhouette, and a sentence per group ("high
  flipper_length_mm, low bill_depth_mm": columns at least half a standard
  deviation from the mean of all rows);
- **Compare with a column** (any column with 2 to 50 values): the counts per
  group and value, the adjusted Rand index (1: the groups are the column's
  values, 0: chance) and what it means ("Group 3 is all Gentoo (123)").

**How many groups?** opens a chart for the method: K-Means inertia (the elbow)
and silhouette for 2 to 8 groups; Gaussian mixture BIC and AIC; the top 12
merges of the hierarchical tree, with the cut and the silhouette of each cut;
DBSCAN's sorted distance from each row to its (min samples − 1)-th neighbour,
with eps drawn and its knee (Kneedle). The suggestion comes in words, and
**Use 3** (or **Use eps 0.643**) sets it.

**Add as a step** puts the method's node after the data in use (the rest of the
graph gets a `cluster_id` column; a Gaussian mixture also adds
`probability_0` ..); **Keep as a node** adds it as a branch. The node gets the
tab's settings, so it gives the same groups ([clustering](clustering.md#the-nodes)).
**Export CSV...** saves the groups and a row per row used; **Copy** copies the
summary.

## Projection

What the rows look like in two dimensions: **PCA**, **t-SNE** or **UMAP** of the
chosen number columns (Scale as in Groups).

- PCA shows each component's share of the variation (a bar each), the loadings
  of the first two (those of 0.4 or more stand out) and what the components
  are in words ("Component 1: flipper_length_mm, body_mass_g and
  bill_length_mm together, against bill_depth_mm").
- t-SNE takes perplexity, learning rate and iterations; UMAP neighbours and
  min distance. Both read at most 5,000 rows: beyond that a seeded sample (the
  seed is typed) in table order, and the sentence says so. t-SNE shows its
  iteration, KL divergence and a progress bar; **Cancel** stops it. UMAP has no
  progress to show, so Cancel only stops waiting for it.

**Colour by** any column (text: a colour per value, the 17 most common beyond
18 values; numbers: four quartile bins) or by the groups of the Groups tab when
they were found on the same columns. **Add as a step** / **Keep as a node**
add the PCA, t-SNE or UMAP node, which keeps the input columns and adds
`pc_*`, `tsne_*` or `umap_*` beside them; the t-SNE and UMAP nodes work on
every row ([dimensionality reduction](dimensionality_reduction.md#the-nodes)).

## Series

One number column over time, in four looks: **Parts** (STL or classical
decomposition into trend, season and residual), **Seasonality** (the periods
of the detrended periodogram ranked by their share of the power, the season
by position), **Stationarity** (ADF and KPSS on the series and its
differences, the suggested differencing, rolling mean and spread) and
**Correlogram** (ACF / PACF with their bands, the seasonal lags marked, the
Ljung-Box test, the orders the cut-offs suggest). **Time** is the Data Input's
time column, else the one date column, else the row order: a date column, a
text that reads as dates, or numbers; the rows are sorted by it and the step
(monthly, daily, every 10 seconds) is read from it. **Period** is found from
the detrended periodogram and can be typed over. Each look says the series
("144 months, monthly, 1949-01 .. 1960-12; period found: 12") and puts its
node into the graph: **Add as a step** (Decomposition, a Differencing step at
the suggested order) or **Keep as a node** (Seasonality Detector, Stationarity
Test, ACF and PACF). The figures match statsmodels; the rules, the words and
the nodes are in [time series](time_series.md).

## Forecast

The same series forecast a typed horizon ahead by **Simple** exponential
smoothing, **Holt** (additive or damped trend), **Holt-Winters** (additive or
multiplicative season), **ARIMA** (p, d, q typed or chosen) or the
**Seasonal naive** baseline, with the model's prediction interval at the
typed level and the dates continued from the step. **Check on the last** H
rows (on by default: twice the period, else 12) holds them out, fits every
method on the rest and scores each against repeating last cycle (MAPE,
RMSE), in a table and a sentence; then the chosen method is fitted on
everything. The page shows the next value with its range, the AIC, the
series with the fit and the band, the forecast table and the held-out
table. **Add as a step** / **Keep as a node** add the ARIMA or Exp. Smoothing
node with these settings; **Stop** ends a long search ([time
series](time_series.md#forecast)).

## Visualize

Quick plots of the picked dataset, with the same plot types, plot view and
settings as the Dashboard.

- **New plot** lists every plot type, grouped (Basic, Distribution, Grid and
  density, ...). A new plot starts with fitting columns, the target first
  (Spotify: a Histogram opens on track_popularity).
- **PLOTS** on the left lists the plots made here with their types; click
  one to show and edit it. Each dataset keeps its own list for the session.
- The plot has its toolbar (Fit, Log Y, Legend, Export as PNG, SVG or CSV).
- On the right: title, plot type, the columns (with their roles), Colour by,
  **All settings in the Plot window**, and **View SQL** (the query the plot
  runs, Open in Query tab, Copy).
- **Keep as Plot node** adds the selected plot to the graph as a Plot node
  reading the node that made the data (a Data Input, a step, or a recipe's
  result for a step inside a recipe), placed below it; the Explore strip lists
  it and Ctrl+Z on the canvas undoes it. The Plot node is kept with the graph
  and reads the data again when it opens. Off, with the reason on hover, when
  no plot is selected or the data is not a table of this graph's nodes.
- **Remove** and **Clear all** remove the selected plot or all of them.
- The list of plot types scrolls inside the main window; a long note after
  an action goes on its own line.
- **Add to Dashboard** copies the plot into a Dashboard node on the same
  data: the dashboards of the open graph are listed (one on other data is
  greyed, with the reason), plus **New Dashboard node on Spotify**, which
  adds a Dashboard connected to the Data Input with the automatic layout
  and this plot. The dashboard opens with the plot selected.

## Plugins

The analytics installed plugins add: questions about a dataset that a plugin
answers (for example *Coefficient of variation* from the example plugin
*Example Data Formats*). The tab is there with no such plugin too; it then
says so and has **Open Plugin Manager**.

- **Analytic** lists every analytic of every loaded plugin, grouped by the
  category the plugin gives, with the plugin and its version beside it and
  its description under it.
- **Columns** starts with every column the analytic reads (numbers, text or
  any, up to the plugin's maximum). In the list, the columns it cannot read
  stay, dimmed, with the reason (*text · this analytic reads numbers*).
- **Run** starts it; nothing runs on its own, since the analytic is the
  plugin's code and may read every row. Beside it: *reads 14 rows of 2
  columns*, or why Run is off (too few or too many columns, or *Example Data
  Formats is not loaded*, with **Open Plugin Manager**).
- While it runs, the rows read so far and **Cancel**; Task View shows the run.
  The chosen columns go to the plugin in batches, every row: a table kept on
  disk is read one row group at a time, so it does not have to fit in memory.
  The plugin sees only those columns.
- The answer: the plugin's single number (when it gives one) large, its
  values per label as bars with the value printed, its note, and *Computed by
  Example Data Formats 1.0.0 on 14 rows in 3 ms, 11:24:05*. **Values** is
  the same numbers as a table, with **Copy** and **Export CSV...**.
- Changing the analytic or the columns keeps the answer, dimmed, under
  *Choices changed since this answer* with **Run again**. When the plugin is
  unloaded, the answer stays dimmed and Run is off.
- A plugin that fails, throws or crashes in its analytic shows its message;
  the Engine keeps running.

Plugin analytics are not graph nodes: they answer a question and do not
change the data. Writing one: [the plugin API
reference](../extending/plugin-api-reference.md) (IAnalyticsProvider).

## Prepare

The graph's Preparation Recipe, step by step. On the left, **RECIPE STEPS
(NODES)**: each step node in run order with its kind and its last run
(*SQL Query - 15,349 rows - 3.3 s*, *Check - pass*, *failed* in red
with the error on hover, *not run*). Click a step to show it; double-click to
show it on the canvas. A graph with several recipes lists them first; nodes
that prepare data outside a recipe are listed as *Steps outside a recipe*.

**An SQL step** opens in the SQL Workbench's editor (highlighting,
completion with its inputs' columns once they are loaded, find, Format).
The header names the step, *SQL Query node* and what it reads.

- **Run step** (Ctrl+Enter in the editor) saves the query to the step, then
  runs the Data Pipeline through it and its Check nodes. The rows it
  made show below (*15,349 rows - 13 columns - at 21:16:55*, the first 1,000
  in the grid) with the rows each part of the query keeps (*src 7 > parts
  17,083 > kept 15,385 > result 15,349*). The checks' *pass* / *fail* show in
  the step list.
- **Save to step** writes the query into the node without running it, also
  inside a collapsed recipe; Ctrl+Z on the canvas undoes it. *not saved*
  shows while the editor differs from the step; **Reset** goes back to it.
- **Open in Workbench** opens the step's SQL Workbench dialog (Plan,
  Parts, Guided mode); a collapsed recipe opens on the canvas first.
- **Add a step after** adds an **SQL step** or a **Guided step** after this
  one in the recipe (on the canvas, which opens a collapsed recipe).

**Guided | SQL** above the editor switches how the step is edited. A guided
step opens in Guided mode: its operations as cards (the rows after each one
once the step has run) and the selected operation's fields beside them, or
below them when the window is narrow. **Add an operation** lists the
operations by group (type to find one) and adds it after the selected card;
**Start from a template**, **Undo** and **Redo** work on the operations, as in
the Workbench (see [guided steps](sql-workbench.md#guided-steps)). **Save to
step** writes the operations and the SQL they make; **Run step** saves, then
runs. An operation marked in red must be fixed before the step saves or runs.

In SQL mode a guided step's query is read-only; **Edit as SQL** (it asks first)
turns it into an ordinary SQL step when you save. An SQL step in Guided mode
offers **Start a guided step instead**, which replaces the query when you save
(Reset brings it back until then).

**Scale, power and log steps** (Standard, Min-Max, Robust and Max-Abs
Scaler, Quantile, Power and Log Transform) show their settings as typed fields
and switches (a scaler's **Method** switch replaces the node with another
scaler, keeping its links and columns), the last run's before / after per
column with two histograms, a power step's λ, skewness, normality test and
likelihood curve, and **Compare methods**; see
[scaling and transforms](data_transforms.md#the-step-in-prepare).

**Other steps** (a Check, a step node) show their last outcome,
their settings as saved in the graph, **Run step**, **Open dialog** and
**Show on canvas**; settings are changed in the node's dialog.

**+ Add a step** under the step list (and **Add a step after** beside an SQL
step) adds after the selected step: an **SQL step**, a **Guided step**, a
**Row Filter**, **Joiner**, **GroupBy** or **Text Clean**, a step from
**Scale and transform** (Scale, Log transform, Power transform), or any other
preparation node from **More nodes**. The new step is picked. The node goes on the canvas to the
step's right, the nodes after it move right to make room (a collapsed recipe
opens first), and it is selected there; Data Studio stays in front, and the
step joins the list. Ctrl+Z on the canvas undoes it. A graph with no steps
yet offers **Add a step after** its Data Input. **Data > Transform** adds a
scaling, log or power step after the data in use and picks it.

**Checks.** A Check node checks one of: a row count (exactly, at
least, at most; all rows or the rows where a column is true or has a value),
**allowed values** of a column, **unique** columns, a column **not empty**, or a
numeric column in a **range**. A failing check stops the run with the count and
the first offending value; nothing after it runs, and its result from an
earlier run is dropped. A guided step's **Add checks** become these nodes when
the step is saved: in Prepare, in the SQL Workbench, or by switching the
operation on in Properties (they need to be its last operation that is on).
**Validate** lists them all and adds new ones.

### An image graph's recipe

A graph whose images go through a **Resize** shows its **Image recipe** in
Prepare instead of the table steps: the graph's image steps in the order
training runs them, each row being the node itself. The subtitle names the
data, the model input and the counts (*cat_dog to model input [64, 64, 3] ·
2 steps once, 2 per batch*); the tab caption counts the steps without the
Resize (*4 image steps*), and the **In the graph** strip has one *image
steps* chip with Show on canvas.

The rows come in four groups, the same split training makes:

- **First**: the Resize.
- **Once, before training**: steps done the same way to every image of every
  part when the images are prepared (they are kept in cache/images): Center
  Crop, Grayscale, Blur, Morphology and the OpenCV steps (CLAHE, Denoise,
  Sharpen, Edges, Smooth, Warp, Pyramid).
- **Per batch, training images only**: everything from the first random step
  on, drawn again for every batch on the GPU; validation and test images skip
  them. A Blur or Grayscale after a random step runs here too.
- **Last**: the Normalize, on the GPU with each batch.

Each row shows the step's picture on a sample image (the image after that
step; a per-batch row shows several draws, and Normalize shows the picture the
model gets), its settings in short and the shape after it. **Next sample**
takes the next image of the index; the pictures show once the images are
loaded and indexed. The first pictures of a step take a moment (its GPU code is
built once); the label says *Drawing the pictures of ...* meanwhile.

- **Settings** opens the node's settings in the row, the same typed fields as
  Properties; a change applies to the node.
- **Show on canvas**, **Open dialog** and **Remove** (once-only and per-batch
  rows) act on the node; Ctrl+Z on the canvas undoes a removal.
- **Drag a row by its handle** to move its step: over the upper half of a row
  it goes before that row, over the lower half after it, and a line on that
  side shows where it lands (no line: the order would not change). The node
  moves on the canvas with its links, and the steps between its old and new
  place swap places so the line stays in order. Resize stays
  first and Normalize last, and an OpenCV step can't go after a random one
  (*Image Contrast (CLAHE) is done once with the images: it must come before
  the first random step.*); a refused move shows why while you drag.
- **+ Add a step** offers the steps done once; **+ Add an augmentation** offers
  the random ones (Random Crop, Random Resized Crop, Horizontal / Vertical
  Flip, Rotate, Random Affine, Random Perspective, Color Jitter, Random
  Grayscale, Gaussian Noise, Advanced Augment, Augmentation Preset) and, under **The same for
  every image**, the device steps that are not random. Only image steps are
  offered, and the new node goes after the group's last row.

## Validate

The graph's checks in one table: **check (node)**, what it **reads** (the step
whose rows it checks, past any checks in front of it), its **rule** in words
(*exactly 15,349 rows*, *exactly 3,712 rows where source_id = pg41633*,
*exactly 0 rows where training_admitted is true*, *role is one of book,
chapter*, *paragraph_id is unique*, *text is never empty*, *word_count from 1
to 900*) and its **last run**: *pass - 15,349 rows* (a row count shows what it
counted), *pass*, *fail* with the message under the rule, *could not run* (for
example a column that is not there), or *not run* (not run yet in this
session, or after a check that failed). The line above says *4 of 4 checks
pass*. Double-click a check's name, or **Show**, to show its node on the
canvas.

- **Run checks** runs the Data Pipeline through every check and the steps
  before them (an export after the checks is not written).
- **Add a check**: **Row count**, **Allowed values**, **Unique**, **Not
  empty** or **Range** opens the form below the table. **After** picks the
  step it checks (the dataset chosen in the rail, else the step the other
  checks read); **Name** is shown when it fails (by default the rule). The
  rule's fields pick from the step's columns once it has run (type them
  before). *Rule:* shows the rule as it will read, and **Add check** stays off,
  with the reason on hover, until the form is complete. The check is added as
  a Check node right after the step (a collapsed recipe opens on
  the canvas; Data Studio stays in front); Ctrl+Z on the canvas undoes it.
- **Edit** in a check's row opens the same form on that node: change the kind
  or the rule; **Save check** writes the node (Ctrl+Z on the canvas undoes it).
- **Data Validator...** opens the validator's form: **Reads** picks the step,
  and **Required columns**, **No missing value** and **No repeated value** pick
  columns (checkboxes once the step has run). **Add validator** adds a Data
  Validator node reading the step, beside what already reads it; for a step
  inside a recipe it reads the recipe's result, since every recipe step must
  lead to the recipe's output. It reports the rows that break a rule as an
  issues table rather than stopping the run; its row shows its rules and *no
  issues* / *15,345 issues* (the table is its node result in the rail), and
  **Edit** opens the same form on it.
- When the graph cannot run, a red line under the summary says why and **Run
  checks** is off.
- In a narrow Data Studio, what a check reads goes under its rule and **Edit**
  / **Show** under its last run.

## ML readiness

The splits a model trains and is scored on, side by side, before training.

**Which splits.** A Data Split node's inputs, in pin order: train,
validation, test (a graph that feeds separate Data Inputs into them, like a
frozen family split). Without a Data Split, loaded datasets named
`<name>_train`, `<name>_validation` (or `_val`, `_valid`, `_dev`) and
`<name>_test` make a set called `<name>`. Several sets show as chips to
pick from. A Data Split that splits one input by ratio does that during
training; it is not previewed here. A Data Split input that is not loaded is
named in an orange line.

**Images.** For an image dataset the job shows the split training will
make, from the canvas graph's compile: a card per part, each class in each
part with its share, whether the shares match the whole set's, and what was
counted. Nothing is queried (see image_split.md).

**When.** Opening the job compares the splits in the background (Task View
shows it); so do a change to a split's data and **Check again**. The tab's
line then says what it found (*no leakage*, *leakage found*, *not
checked*).

**The header** names the set and its splits, the verdict (*No leakage
found*, *Leakage: 3 groups in two splits* in red, *Leakage not checked* when
no Group, ID or Text column is set, *One split only*), **Show the Data
Split** on the canvas and **Set roles in Profile** (Explore > Profile on the
train split).

**The cards:**

| Card | What it shows |
| --- | --- |
| Roles | The Text, ID, Group and Target columns, read from the train split's roles (yours first, then the graph's, then inferred). Group is never inferred: set it in Profile. |
| Split sizes | Rows per split, distinct groups and IDs; a split whose rows repeat an ID says how many. |
| Leakage across splits | Groups, IDs and identical texts found in two splits (exact matches), with the pairs when there are any. |
| Text length (words) | Median and mean words per row of the Text column; a split 10% or more shorter or longer than train is named (kjv80: *Validation and test are about 14% shorter than train.*). |
| Class balance | With a Target: the largest classes' share per split, the train split's largest and smallest class, and a class whose share moves 10 points or more between train and another split. The Data Loader's balance mode. |

Everything is counted with SQL on the whole tables; the data is not
changed. The train split's roles come from its profile, cached in the
project's `cache/query_results` like Explore > Profile's. When several
columns read as Text (a hash column does too), the longest one is used;
set Text on another column in Profile to choose it.

## Publish

The graph's outputs (Export CSV, Export JSON, Export Parquet, Data Output) in
one table: **output (node)**, what it **reads** (the step, past any checks in
front of it), the file it **writes** (relative to the project when inside it;
the full path on hover) and when it was **last written** (*15,349 rows -
15:19:58*, *not written yet*, *failed* with the error). Click a row to show its
lineage; **Show** selects the node on the canvas.

**Lineage beside the file.** Every time an output writes its file, the run
also writes `<file>.lineage.json` beside it, whether the run came from
Publish, Run step or Execute Pipeline. Publish shows the selected output's:

- **Made by**: the recipe and its revision (the first 12 hex digits of the
  SHA-256 of its steps' types and settings, so any change to a step changes
  it) and the number of steps;
- **Inputs**: each Data Input's file with its size and SHA-256. A file is
  hashed once (the digest is kept by path, size and write time in the
  project's `cache/ingestion/digests`). A CSV over the data budget's result
  threshold is hashed while it is converted, so it is not read again. A
  larger file never read in full (a Parquet file opened in place) shows *not
  hashed (large file; size and time identify it)*, and the lineage file holds
  its write time;
- **Result**: rows, columns, the file's size and SHA-256 (hashed while the
  file is written, so the export is not read back);
- **Checks**: *4 of 4 pass*, or the first one that did not;
- **Written**: when, from which graph file.

**Steps** opens the list of steps above the output. The file itself also holds
each step's settings and outcome, the schema, and the full digests.

**Publish...** (here or in the header) opens *Publish dataset*:

- **Output**: one of the graph's outputs (its name and file are set and it
  runs), or **New Parquet export**, which adds an Export Parquet node reading
  **Reads** (for a step inside a recipe, the recipe's result) and runs it.
- **Name** is the node's name; **Location** is the file, relative to the
  project unless it starts with a drive (suggested:
  `datasets/published/<step>.parquet`).
- **Publish** writes the file and its lineage; the table and the card update
  when the run ends. Ctrl+Z on the canvas undoes the node change.

**Add Data Output** adds a Data Output node reading the step chosen in the
rail (or the last step); its format and file are set in its dialog.

## Results

A model's results next to the data, in four views: **Test**,
**Cross-validation**, **Learning curve** and **Generations**. The job's line
counts what there is to show (*test: 72 rows · 2 preview runs*, and any
evaluation nodes in the graph). After a test, Data Studio opens Results > Test
by itself.

### Test

One view of a test (it replaced the Test Results window and the Confusion
Matrix, ROC / AUC and PR Curve windows). **Showing** picks what it shows:

- **Latest test**: the last Train > Test run (its rows are the
  `test_predictions` table);
- **Kept: name**: a test kept with Keep as node (`results/tests` in the
  project);
- **A table...**: any table read as a test. Pick the table, **Classes** or
  **Numbers**, the **True** and **Predicted** columns and, for the curves, the
  **Scores**: one `score_<class>` column per class, or one column for two
  classes (then say whose score it is). Columns named like `label` /
  `prediction` / `score_<class>` are picked for you. Rows missing a value are
  left out and counted. A model scored in a script or an imported CSV gets
  every section this way.

The headline (*Accuracy 94.12%*, *MAE 318.2*), the test's size and time, and
its figures as chips (loss, macro and weighted F1, AUC; RMSE and R² for
numbers). A table read as a test has no loss or time, so they are not shown.
Sections:

| Section | What it shows |
| --- | --- |
| Overview | macro precision / recall / F1, weighted F1; the best and worst class, classes never predicted; a regression's bias |
| Confusion matrix | counts, or the share of each true class; the largest confusion; a cell's tooltip says *x of y (z%)* |
| Classes | precision, recall, F1, found, false alarms, support per class; filter and sort |
| Curves | ROC and precision-recall (one line per class against the rest for three or more), AUC and AP; the precision-recall curve is drawn as steps, so its area is AP; a ring marks the threshold |
| Threshold | two classes, or one class against the rest: the **Positive class** by name, a typed **Threshold** (- / +) and **Best F1**; precision, recall, specificity, F1, balanced accuracy, MCC, accuracy and the 2 x 2 counts at that threshold |
| Plots | regression: predicted vs actual, residuals vs predicted, residual histogram |
| Predictions | every row with its dataset row (an image test: its file, the full path on hover), or the mistakes with the most confident first (regression: the largest errors first) |
| By column value | the error by the values (or ranges) of a column of the tested table |
| Comparison | the latest test beside the one before (Train > Test > Compare Test Results opens it) |

The figures at a threshold are read off the class's ROC curve, which holds the
true and false positives at every distinct score, so moving the threshold does
no new pass over the rows. They match scikit-learn
(`tests/fixtures/generate_evaluation_fixtures.py`).

**Export CSV...** / **Export JSON...** save what is shown. **Keep as node...**
keeps the latest test (below). **Clear** asks first, then forgets the latest
test and the one before (kept tests stay). **Run test again** tests again.

`test_predictions` has a `row_id` column (each test row's row in the tested
table, counted from 0; an image test's row in the dataset's index table),
a `file` column after it for an image test, and its metadata keeps the
tested table's name, the
class names, the loss and the time, so the results join back to the data and
a copy read back later gives the whole run.

**Keep as node...** keeps the latest test for later: testing again replaces
`test_predictions`, a kept test stays. Give it a name (the graph's name and
*test* to start with); **Keep**:

- saves the test's rows once as `results/tests/<name>_<date>_<time>.parquet`
  in the project (an existing file is never replaced), with its tested table,
  class names, loss and time and the name it was kept under;
- adds a **Data Input** named after it reading that file and a **Test
  Evaluation** node after it, in free space on the canvas (Ctrl+Z removes
  them; the file stays). Showing then lists it under *Kept*.

The **Test Evaluation** node (Analytics) reads any test's per-row table (a
kept test, or `test_predictions` itself) and gives Results > Test's figures as
four tables:

| Output | Columns |
| --- | --- |
| Metrics | `metric`, `value`: accuracy, macro precision / recall / F1, weighted F1, AUC, loss, rows (regression: MAE, RMSE, bias, loss, rows) |
| Confusion | `true`, `predicted`, `count` (a classifier) |
| Per class | `class`, `rows`, `precision`, `recall`, `f1` (a classifier) |
| Worst rows | `dataset_row` (counted from 1), `true`, `predicted`, `confidence`: the most confident mistakes; regression: `error`, the largest errors first |

*Worst rows* keeps 20 rows (Properties: **worst_rows**). A table that is not a
test's is refused (*not a test's predictions*).

### Cross-validation

Trains the open graph once per fold with fresh weights, each time on the
other folds, and scores the fold it did not see. **Folds** (typed, 2 to 20),
**Keep class shares** (stratified folds for classes) and **Seed**; **Run**
(or Train > Evaluate > Cross-validate).

Run goes through Train's own path: the graph is compiled and checked, data
that is not loaded is loaded, the memory question is asked when it applies,
and the graph's preparation steps run. Then, instead of one training, the
prepared table is trained once per fold:

- the rows are the table less the graph's own **test rows** (the ones Train >
  Test tests on), so those stay unseen;
- each fold trains for the graph's epochs with its batch size, optimiser and
  device; early stopping and the best-epoch restore are off, since they would
  pick the epoch by the fold being scored;
- each fold is scored by the test code: accuracy, macro F1 and loss (numbers:
  MAE, RMSE, R²).

It runs in the background (Task View shows *Cross-validation: fold 3 of 5*)
with the folds listed as they finish, and **Cancel** stops it. It does not
touch the trained model in use, its checkpoints or the Training Dashboard;
Train waits while it runs, and it waits while Train runs.

The answer: the mean with its spread (*Accuracy 91.9% ± 3.7 over 5 folds*),
the range, a sentence on what to expect on new rows (with the latest test
beside it: inside the folds' range, or a lucky or unlucky split), the bars per
fold with the mean, the fold table, and a fold that stands out (*Fold 5's loss
is high (2.663) though its accuracy is not low: a few mistakes made with great
confidence*). **Export CSV...** saves one row per fold.

The runner matches PyTorch training the same graph on the same folds
(`tests/computation_truth/reference/generate_cross_validation_fixtures.py`):
every fold's accuracy and loss fall inside the spread of 8 seeds.

### Learning curve

Trains the open graph on shares of each fold's training rows (**Training
rows**, typed: *10%, 25%, 50%, 75%, 100%*), with **Folds** and **Seed**, and
scores both the rows it trained on and the fold. **Score**: accuracy, macro F1
or loss (numbers: MAE, RMSE, R²). The plot shows training against validation
with one-standard-deviation bands; the table has the gap at each size; the
sentences say where validation levels off (or that it still improves, so more
rows would help) and what the gap at full size means (memorising, or a model
that needs more information rather than more rows). Loss and accuracy per
epoch of a run stay in the Training Dashboard.

### The Cross-Validation and Learning Curves nodes

They keep the settings in the graph, on the data path the way Data Split is
(**Add to the graph** in the view puts one after the Data Input). Data passes
through them unchanged, so Train trains as usual. When the graph has one, Run
(and Train > Evaluate) uses its settings, and the view says so.

| Node | Parameters |
| --- | --- |
| Cross-Validation | `folds` (2 to 20), `stratified` (true / false), `seed`, `results_table` |
| Learning Curves | `folds`, `seed`, `train_sizes` (*10%, 25%, 50%, 100%*), `results_table` |

A finished run is published as a table named by `results_table`
(`cross_validation_results`, `learning_curve_results`): one row per training
(`fold`, `share`, `train_rows`, `val_rows`, the validation scores, a learning
curve's training scores, `epochs`, `seconds`). It is the node's second output,
so a Plot or Export node can use it, and Data Studio lists it like any table.

### Generations

The generation previews a language model's training writes (set on the Data
Loader's Generation tab). **Run** lists every run of the project that wrote
previews, newest first; under it is the same Generations view as the Training
Dashboard's: the latest pass with one card per prompt, two passes compared, or
one prompt over time. See [Generation previews](generation_previews.md).
