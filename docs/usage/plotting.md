# Plotting in the Engine

What draws plots in the CyxWiz Engine today, and how to use each part. A
new plot system (one Plot window with a plot-type picker, shared with the
Plot node and, later, the Dashboard) is being built in TOFIX134; this page
describes what works now.

## Figures from Python (matplotlib)

`plt.show()` in a script, a notebook cell or the Console puts the figure in
the Engine instead of opening a separate window.

```python
import matplotlib.pyplot as plt

plt.plot([1, 3, 2, 4], label="loss")
plt.title("My figure")
plt.legend()
plt.show()
```

- **Script (Script Editor, F5):** the figure opens in the **Plot Output**
  window. Plot Output opens by itself if it was closed; it keeps every
  figure of the session (arrows or thumbnails to move between them).
- **Notebook cell:** the figure shows under the cell, in the notebook.
- **Console (Python REPL):** after a script has run in the session, the
  figure goes to Plot Output as well.

Plot Output toolbar: previous / next figure, zoom (or mouse wheel, drag to
pan), Fit, 100%, copy to the clipboard, save as PNG, close the figure,
clear all. Right-click a figure for Copy, Save as PNG and Close.

## The Plot window (Table Viewer)

Open a table in the Table Viewer (a dataset, a CSV, or a variable from the
Variable Explorer), then either right-click a column and choose **Plot**
(Histogram, Line Chart, Bar Chart, Box Plot, or Use as X-axis for Scatter)
or press **Plot** under the column's statistics. The Plot window opens:

- **Plot type** (left): Basic (line, scatter, bar, histogram, area, step,
  stem, pie, polar), Distribution (box, violin, KDE, parallel coordinates,
  error bars), Grid and density
  (heatmap, matrix, pair plot, 2D histogram, hexbin, contour, filled
  contour), Images (image), Vector
  fields (quiver, stream). Pick
  one; the columns that still fit are kept.
- **Rows** (top of the right panel): **All**, **First** N rows, a **Range**
  of rows (numbered from 1, as in the Table Viewer) or **Filter**:
  conditions *column = != < <= > >= contains value*, all of which must
  match ("class = 7" keeps 7,293 of MNIST's 70,000 rows). The label above
  the plot says which rows are drawn; exports and the Python script use the
  same rows. To filter for the whole graph, use a Filter Rows node.
- **Data**: the X column (or the row number for lines), one or more Y
  columns (one series each), **Colour by** a column, bins, median and mean
  lines, density, smoothing (moving average), the title. **Values** lists
  count, missing, min, max, mean and median of the plotted column.
- **Choosing columns in a wide table**: every column field opens a list
  with a search box. Each column shows its type (# number, Aa text), its
  range and how often it is not 0 ("0 to 255 · 66.3% not 0"); columns with
  one value (65 of MNIST's pixels) are hidden unless you untick that.
  Up / Down move, Enter picks, Esc closes. Y values: **Add all N matches**
  of the search, or a **Range** of columns ("pixel400" to "pixel409");
  the chosen columns show as chips (click one to remove it). The legend
  lists 12 series; hover shows all.
- **Colour by a number column**: on a scatter, a column with more than 12
  values colours each point on the theme's scale, with a colour bar named
  after the column (values on both sides of 0 use the two-sided scale);
  **Groups / Scale** switches. Other plot types split such a column into
  six equal ranges. A column with up to 12 values (class 0-9) gives one
  series per value.
- **Heatmap**: **Cell values** sums a column per cell; without it, rows are
  counted. Labels that are all numbers read in numeric order.
- **Matrix** (Grid and density): choose two or more number columns. Values:
  **Correlation (Pearson)**, **Correlation (Spearman, ranks)** (on a -1 to
  +1 two-sided scale, the value in each cell) or **The values** (the
  columns as a grid, one row per table row, up to 500 rows).
- **Contour** and **Filled contour**: X and Y numbers; **Z** is the density
  of rows (lightly smoothed) or the mean of a column per cell. **Grid cells
  across** and **Levels**. Hover gives the value at the mouse.
- **Hexbin**: like the 2D histogram with hexagons. **Hexagons across**;
  colour by the count (optionally its log) or the mean of a column. Hover
  gives the rows in the hexagon and their share.
- **KDE** (Distribution): a smooth histogram, one curve per Y column or per
  **Colour by** group. **Bandwidth** multiplies Silverman's rule (1.00).
- **Bar** with **Colour by**: one bar per group, **Grouped**, **Stacked** or
  **100%** (each category's shares). Hover lists every group.
- **Pie**: **Donut** shows the total in the middle.
- **Polar** (Basic): an **Angle** column and a **Radius** column, one line
  (or, with **Points instead of lines**, one set of points) per **Colour
  by** group. **Angle in**: Auto (text columns are categories, number
  columns degrees), Degrees, Radians, or Categories, which share the turn
  (in numeric order when they are numbers) and close each line. 0 is at
  the top, angles go clockwise. Example: months as the angle, passengers as
  the radius, colour by year draws one turn per year.
- **Quiver** (Vector fields): an arrow per row from **X**, **Y**. **Arrows
  from**: **u, v** (the arrow's x and y parts), or **Direction + length**
  (degrees clockwise from north and a length); tick **Wind** when the
  direction says where the wind comes from (weather data), so the arrows
  point where it blows. Arrows are coloured by length (colour bar) and
  scaled to fit their spacing. A dense grid shows **every Nth arrow** (0:
  about 600 at most); the label says so, and hover and exports use every
  row. Hover: the length, the direction it points to and u, v.
- **Image** (Images): any table whose number columns make a picture. Choose
  the **Pixel columns** (search "pixel" and **Add all matches**, or a
  range), and optionally a **Label** column. The shape is square when the
  count is a square (784 columns: 28 x 28), 3 channels when the count / 3 is
  a square (3,072: 32 x 32 RGB, as CIFAR-10; tick **Three planes** when the
  columns are all red, then green, then blue), else set **Width**.
  **Show**: **One row** (Previous / Next), a **Gallery** of the chosen rows
  (the Rows section picks them, e.g. class = 7; **Pictures at most**), or the
  **Mean per class** (the average picture of each label, with its count).
  **Values**: Auto (lowest to highest), 0 to 255 or 0 to 1; **Grey** or the
  theme scale; **Invert**. Hover gives the row, the label, the pixel and its
  value. Only the rows shown are read, so wide tables stay quick.
- **Pair plot** (Grid and density): 2 to 6 number columns; every pair as a
  scatter (sampled to 2,000 rows) and each column alone on the diagonal (a
  KDE per **Colour by** group, or **Histograms on the diagonal**).
- **Parallel coordinates** (Distribution): two or more number columns, each
  on its own vertical axis (its lowest and highest value at the ends), one
  line per row (1,000 at most) coloured by group. Hover highlights the
  nearest line and lists its values.
- **Stream**: the same columns as Quiver; lines that follow the field
  (rows of a grid are used as they are, other tables are averaged onto a
  40 x 40 grid), coloured by speed, with an arrowhead on each line.
  **Density** spaces the lines. Hover gives the speed and direction at the
  mouse.
- **Model results**: plots of a trained model's results, read from plain
  columns of any table (a predictions file, an evaluation node's output, a
  training history). Each shows its figures in the plot corner and under
  **RESULTS** in the Data panel.
  - **Confusion matrix**: **Actual** and **Predicted** label columns (text
    or numbers). Each cell shows the count and its share; **Colour and share
    by**: counts, share of actual (each row adds to 100%, the default),
    share of predicted, or share of all rows. Accuracy and rows in RESULTS.
    Hover: "Actual tested_positive · Predicted tested_negative", the rows,
    the share, right or wrong.
  - **ROC curve**: **Actual** and the **Score** of the positive class
    (a probability or any score; higher means positive). AUC in the corner,
    the chance line. **Positive class**: auto picks 1, true, yes or a label
    with "positive", else the last label; type another to change it.
    Hover gives the threshold and both rates.
  - **Precision-recall curve**: the same columns; precision by recall as a
    step, the average precision (AP, as scikit-learn) and the positive
    share as a dashed baseline.
  - **Calibration**: **Actual** and a **Probability** (0 to 1). The share
    of positives in each of the **Bins** by the mean predicted probability,
    the perfectly calibrated diagonal, rows per bin as bars along the
    bottom (right axis); Brier score and ECE.
  - **Residuals**: number **Actual** and **Predicted**; actual minus
    predicted by predicted, a line at 0; RMSE, MAE and R² (large tables are
    sampled like a scatter; the figures and exports use every row).
  - **Learning curve**: **X** (training rows or epoch) and one or more
    curves (train, validation); a **Spread** column per curve draws a band
    (± that value). The last curve's best point is ringed: **Auto** takes
    the lowest for a name with loss or error, else the highest.
  - **Feature importance**: a **Feature** column and its **Importance**,
    as horizontal bars, largest on top; **Show the top** N (20), an
    optional **Spread** for error bars. Hover gives the value and the rank.

  Example (diabetes classifier): Confusion matrix of actual by predicted
  shows 134, 16 / 36, 45 (accuracy 77.5%); ROC of actual by probability
  gives AUC 0.871; PR gives AP 0.767; Calibration gives Brier 0.143. A
  Spotify popularity regression gives RMSE 20.76 and R² 0.265 over 2,575
  rows.
- **Flows and hierarchies**:
  - **Sankey**: two or more **Steps** columns, left to right (text or
    numbers as categories). Each band joins a category of one step to one
    of the next; its width is the rows, or the sum of a **Value** column. A
    ready flow table works the same way: Steps = source, target and Value =
    the flow. **Largest per step** (8) keeps the biggest categories and
    joins the rest as *other*. Hover a node for its total and share, a band
    for its value and share of where it starts.
  - **Treemap**: one to four **Groups** columns (outer to inner) and a
    **Size** (summed; empty: rows). The rectangles' areas follow the size.
    **Colour** takes a number column (the mean per rectangle on the theme
    scale), or leave it empty to colour by the top group. Click a group to
    zoom into it; right-click, or click the path at the top ("All > Asia"),
    to go back. The 3,000 largest groups are drawn (the label says when
    there are more).
- **Maps** (built-in world map: Natural Earth 1:110m country outlines,
  public domain, shipped with the Engine, works offline):
  - **Map: points**: **Longitude** and **Latitude** columns (degrees), an
    optional **Size** column and **Colour by** (a number makes a scale, text
    makes groups). The map frames the points; pan and zoom like any plot.
    Hover gives the row's values and the country under it. A longitude
    outside -180..180 or a latitude outside -90..90 is refused with a hint
    (columns swapped?).
  - **Map: regions**: a **Country** column (names such as "France" or
    "United States", ISO codes FRA / FR, and common names like UK or Viet
    Nam) and a **Value**. Several rows for one country: **Sum** or **Mean**.
    **Colour by the log of the value** for values across orders of
    magnitude. Names that match no country are listed under the controls
    (and in the CSV export); countries with no rows stay grey. Hover gives
    the country, its ISO code, the value and the rows.

  Example: `world_countries.csv` (Natural Earth population and GDP)
  as a Treemap of continent > country sized by population and coloured by
  GDP per person, and as Map: regions of gdp_per_person (log); the USGS
  earthquake feed (`earthquakes_month.csv`) as Map: points sized by
  magnitude and coloured by depth.
- **Line and scatter**: **Show the y = x line** draws a reference line
  (chance on a ROC curve).
- **The plot**: hover for values (the nearest x of every series, the
  scatter point, the histogram bin and its share, the category, the pie
  slice, the box statistics, the heatmap cell). Drag to pan, wheel to zoom,
  double-click or **Fit** to fit. **Log Y**, **Legend**.
- The label next to the toolbar says what is drawn: *exact* (all values),
  *reduced* (long lines drawn with the lowest and highest value of each
  step; hover and exports use all points), *sampled* (an even sample of a
  large scatter; exports use all rows), or *first N rows* (a variable read
  with a row limit).
- **Export**: save image as PNG or SVG, copy image, save data as CSV, copy
  data, Plot with Python (copies a matplotlib script; paste it in the
  Script Editor and run it to get the figure in Plot Output).
- The window icon opens the plot in its own window; **Open in Visualizer**
  sends the column to the Visualizer.

Plot colours come from the theme (Preferences > Theme): the series
colours, the plot area (a shade of the window colour) and the text.

## The Plot node (CyxWiz Studio canvas)

Search **plot** on the canvas (or open Visualization in the node list) and
add a **Plot** node. Connect the table you want to look at to its **Data**
input, at any point of the graph:

```
Data Input (mnist_784.csv) --> Plot
```

- Under the node, two lines say what it shows: *Not connected*, *Not
  available yet* (with the reason), *Not read yet*, *Reading N%*, the plot
  and its column ("Bar · class") with the rows and the read time, *Out of
  date*, or *Could not read the data*.
- Double-click the node (or right-click > Configure, or **Open Dialog** in
  Properties) to open its Plot window. The window is the same Plot window
  as above, with a header: *Data at &lt;node&gt;*, rows × columns, the read
  time, and **Refresh** (**Cancel** while it runs).
- A loaded Data Input is read as it is. A node after it (a filter, a
  transform) is run on its own with only the nodes above it; progress and
  Cancel are in Task View. The first plot is picked for you: the counts of
  a label column (class, label, target or y), otherwise of a text column,
  otherwise a histogram of the first number column. The plot type, the
  columns and the labels you choose are saved in the node.
- When a node above changes, the plot keeps its data and says *Out of
  date*; press **Refresh** to read again. Changes beside the plot (for
  example a Dense layer) do not make it out of date.
- Nodes that run only inside training (Train/Val/Test Split, DataLoader,
  Normalize, layers) cannot be plotted yet: the window says so and offers
  **Plot its input (&lt;node&gt;)**, which connects the Plot node to the node
  before it.
- A Data Input without a file says "has no data yet: open it, choose a
  file and Apply".

### Model evaluation

A Plot node after an evaluation node opens on the fitting plot:

```
Data Input (actual, predicted, score) --> Confusion Matrix --> Plot
                                      \-> ROC Curve        --> Plot
```

- **Confusion Matrix** (actual_col, predicted_col): a heatmap of *Actual*
  by *Predicted* with the counts in the cells (or the shares, when the node
  normalizes); hover says "Actual 0 · Predicted 1 · value 38".
- **ROC Curve** (actual_col, score_col, positive_label): the true positive
  rate by the false positive rate, the AUC in the title ("ROC curve · AUC
  0.931") and the y = x chance line.
- **PR Curve**: precision by recall, the average precision in the title.

The same tables read from a file get the same first plot. The type and
columns can be changed like any plot.

The old per-type plot entries (Bar Chart, Line Plot, Box Plot and the other
"Coming Soon" plot nodes) are gone: the Plot window picks the type. A saved
graph that still has one of them does not load; the message names the node
so you can remove it and add a Plot node.

## Training Dashboard

Opens with **Train** on the CyxWiz Studio canvas; in the window list it is
**Training** (Training group), docked at the bottom right by default. It shows the run status, KPI cards (loss, accuracy, time
remaining), live loss and accuracy charts, custom metrics, data
preparation progress, execution truth and run comparison. Long runs stay
smooth: charts draw a reduced copy of each series (up to 4000 points, with
every spike and dip kept). Each chart uses the same plot view as the Plot
window: hover for the values of every curve at that epoch (from all
points), the data label, **Legend** and **Export** (PNG, SVG, copy image,
CSV with every point, copy data). Use the window icon on a chart to open it
in its own window; **Auto scale**, **Follow current epoch**, the epoch
window, **Log loss axis** and **Smooth** are in the chart window's toolbar
and the dashboard's controls.

Only canvas training reports to this window.

## RL Training Dashboard

A separate window for reinforcement learning (episode reward and length,
policy loss, value loss, explained variance). It opens with **Train RL** on
the canvas (graphs with RL nodes), or when a script reports RL metrics:

```python
import pycyxwiz

for episode in range(100):
    reward = run_episode()          # your code
    pycyxwiz.rl_update_metric("episode_reward", reward)
    pycyxwiz.rl_update_metric("episode_length", 200)
```

Metric names the window shows: `episode_reward`, `episode_length`,
`policy_loss`, `value_loss`, `explained_variance`.

## Not available yet

- The `cyxwiz_plotting` Python module cannot open Engine windows;
  `show_plot` raises an error saying so. Use matplotlib as above.
- The plot entries in the canvas node search (Line Plot, Box Plot, Quiver
  Plot, ...) are placeholders; they are replaced by one Plot node whose
  window picks the plot type (TOFIX134 P2).
- Bar Chart node: plots the raw file of the first Data Input upstream, not
  the data at its position in the graph.
