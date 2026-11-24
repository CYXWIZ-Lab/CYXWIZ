---
title: Explore a dataset
description: The Explore job's views, the commands that open each one, and which of your actions change what training reads.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - overview.md
  - query.md
  - profile.md
  - prepare.md
---

The **Explore** job of Data Studio looks at one table: the dataset picked in
the rail, or, when nothing is picked, the table training reads. Use it to
query the data, profile its columns, find gaps and outliers, test and model
it, and plot it, before you prepare it for training.

## Pick a table and a view

1. In **Data Studio**, click a dataset in the rail (a Data Input, a node
   result or another loaded table).
2. Click the **Explore** job, then a view in its strip. In a narrow dock the
   views wrap to a second line.

## The views

Explore has 16 views, and a 17th, **Images**, when an image dataset is
picked.

| View | What it answers | Page |
| --- | --- | --- |
| **Query** | Any read-only SQL over the loaded tables | [Query a dataset with SQL](query.md) |
| **Profile** | Each column's type, role and figures; an image dataset's classes, sizes and quality | [Profile a dataset](profile.md) |
| **Images** | An image dataset's pictures in a grid, a list or a viewer | [Look at an image dataset](explore-images.md) |
| **Missing** | Where values are missing, and dropping or filling them | [Find missing values and outliers](missing-values-and-outliers.md) |
| **Outliers** | Values far from the rest, and flagging, clipping or dropping them | [Find missing values and outliers](missing-values-and-outliers.md#find-outliers) |
| **Correlation** | How the number columns move together | [Correlation and distribution](correlation-and-distribution.md) |
| **Distribution** | How one number column is spread, and which distribution fits it | [Correlation and distribution](correlation-and-distribution.md#see-how-a-column-is-spread) |
| **Tests** | Hypothesis tests | [Statistics](statistics.md) |
| **Regression** | Linear and polynomial regression | [Statistics](statistics.md) |
| **Groups** | Clustering (K-Means, DBSCAN, hierarchical, Gaussian mixture) | [Clustering](clustering.md) |
| **Projection** | PCA, t-SNE and UMAP | [Dimensionality reduction](dimensionality-reduction.md) |
| **Series** | Decomposition, ACF / PACF, stationarity and seasonality | [Time series](time-series.md) |
| **Forecast** | Forecasting a series | [Time series](time-series.md) |
| **Signal** | Spectrum, spectrogram, filters and wavelets | [Signal processing](signal-processing.md) |
| **Text** | Tokens, words, terms, embeddings and sentiment | [Text](text.md) |
| **Visualize** | Quick plots of the dataset | [Plot a dataset](visualize.md) |
| **Plugins** | Analytics that installed plugins add | [Run a plugin's analytics](plugin-analyses.md) |

### Commands that open a view

| Command | Opens |
| --- | --- |
| A dashboard's **Open in Data Studio** or **Edit roles**, the Asset Browser's **Open in Data Studio** | The matching view on that data |
| **Data > Dataset Statistics** | Profile |
| **Data > Explore > Missing Value Analysis**, **Outlier Detection**, **Correlation Matrix** | Missing, Outliers, Correlation |
| **Tools > Statistics** | **Summary Statistics** and **Distribution Fitter**: Distribution; **Hypothesis Testing**: Tests; **Linear / Polynomial Regression**: Regression |
| **Tools > Clustering and Projection** | **K-Means**, **DBSCAN**, **Hierarchical Clustering**, **Gaussian Mixture Models**: Groups on that method; **Cluster Evaluation**: Groups; **PCA / t-SNE / UMAP**: Projection |
| **Tools > Time Series** | **Decomposition**, **ACF / PACF**, **Stationarity**, **Seasonality**: Series on that look; **Forecasting**: Forecast |
| **Tools > Signal Processing** | **FFT**: Signal on Spectrum; **Spectrogram**; **Filter Designer**: Filter; **Convolution Calculator**: Filter in Kernel mode; **Wavelet Transform**: Wavelet |
| **Tools > Text** | **Tokenization**: Text on Tokens; **Word Frequency**: Words; **TF-IDF Analysis**: Terms; **Word Embeddings**: Embeddings; **Sentiment Analysis**: Sentiment |
| **Data > Tokenization** | Text on Tokens |

Inside Profile, a column's missing or outlier count opens Missing or
Outliers on it, **More about this column** opens Distribution, and **See the
matrix** opens Correlation.

## What reaches training

Training reads one table: the result of the node wired into the Data Split's
**Training Dataset** input (the Data Loader's input when there is no Split).
Start Training takes the canvas run's result of that node, else the prepared
data kept on disk, else it runs the steps before it through the Data
Pipeline executor first. Steps that learn (scalers, encoders, Fill Missing,
Outlier Detector) learn from the Train rows only.

| Kind of action | Examples | Training |
| --- | --- | --- |
| **Adds a step** | Every Prepare step; Missing's **Drop these rows** and **Fill in Prepare**; Outliers' **Flag rows** / **Clip values** / **Drop rows**; **Add as a step** in Groups, Projection, Series, Forecast and Signal (Filter, Wavelet); Query's **Turn into SQL step**; Images' **Use in Normalize** | Reads the step's result. Each button says so on hover, and the message after adding one ends with *Training reads its result.* |
| **Adds a branch** | **Keep as a node** (Tests, Regression, Series' Seasonality / Stationarity / ACF, Groups, Projection, Forecast, Signal's Spectrum); **Keep as Plot node**; dashboards | Does not read it. The branch reads the data and keeps its result with the graph. The message ends with *Training does not read a branch.* |
| **Sets column roles** | A role set in Profile on a Data Input's table | The role is written into that node (`column_roles`), and training keeps ID, Group, Weight and Ignore columns out of the features (the compile report says which). Group is not used to split, and Weight is not used to weigh rows. |
| **Changes nothing** | Query's **Save as Dataset**, every **Export CSV...**, ML readiness, Validate's table; *Treat as missing* (until **Make them empty in Prepare** adds a step) | No change. A failing check stops the run. |

One line above every view says what training makes of the table in front of
you:

- *Training reads this table through the Data Split.*
- *Training reads "Scale numbers", not this table.*, with **Show what
  training reads**.
- *Training reads "Scale numbers", which has no table yet (run its steps in
  Prepare); this is not it.*, with **Show the node**.
- *Nothing feeds a Data Split or a Data Loader yet, so training reads no
  table.*

Changed in 1.0.8: the Missing, Outliers, Correlation, Distribution, Images
and Plugins views, and the line saying what training reads.

## Related

- [The Data Studio window](overview.md)
- [Prepare data with a recipe](prepare.md)
- [Compile and run a graph](../graph-editor/compile-and-run.md)
- [Table Viewer](../../main-window/table-viewer.md)
