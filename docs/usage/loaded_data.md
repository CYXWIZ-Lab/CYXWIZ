# Loaded data: what the Engine holds and when it lets go

Loading a dataset (a Data Input's Apply, a pipeline run) keeps it in the Engine
until something releases it. This page says how to see what is held. TOFIX160
adds the release rules as they are built.

## See what is held

**Tools > Monitoring > Memory Visualization > Loaded data** lists every dataset
the Engine holds:

- **Summary:**
  - how many datasets are held;
  - their memory, with each byte counted once;
  - their size on disk;
  - what Arrow holds now, with its peak on the line under it. The peak includes
    reading the file: a CSV load needs far more than the table it makes.
- **Table:** one row per dataset, the largest first. Columns:
  - **Name**;
  - **Kind:** a table in memory, a table on disk (Parquet), sparse features,
    images, audio, text or other;
  - **Rows**;
  - **In memory**;
  - **Shared** (see below);
  - **On disk**;
  - **Reads:** the file or folder it reads.
- **Details:** click a row to see that dataset's facts under the table.

Some datasets hold no memory of their own:

- **Tables on disk** are read a row group at a time.
- **Image, audio and text datasets** read their files as training goes, so
  they show 0 B in memory.

The tab updates when a dataset is added, replaced or removed, the same moment
the Console logs it.

## Shared memory

Several names can point at the same data, for example:

- a pipeline run's copy of its input (`ds_datainput_<id>`);
- a slice of a table;
- the same table registered twice.

That data is held once, so it is counted once in the total. Each of those rows
gets a **shared** mark, and the **Shared** column says how much of its memory
another row also holds. A row that should be a share but shows no shared part
is a real copy, which is worth looking into.

## Logs

The Console (Debug) logs each change with the dataset's size, then one total
for the batch:

```
Dataset catalog: added 'mnist_test' (in memory, 10000 rows, 7.49 MB in memory, generation 1)
Loaded data: 1 dataset, 7.49 MB in memory, 0.00 B on disk; Arrow holds 7.52 MB (peak 88.30 MB)
```

## Example

MNIST test, `mnist_test.csv` (17.4 MB, 10,000 x 785):

- Loading it takes 7.49 MB, because the pixel columns are packed to one byte
  each.
- The exit dialog's **Datasets in memory** shows the same total. Before
  TOFIX160 it said 0 B for every Arrow and Parquet dataset.

## When data is released

| What happens | What goes |
|---|---|
| Delete a Data Input | The dataset its dialog loaded, and everything a pipeline run made from it: its run copy (`ds_datainput_<id>`) and the in-memory copies of it, including the copies a Plot or Dashboard read |
| Delete any step | The results a run registered for it, and its extra outputs. A table another node still passes through stays. |
| Clear the graph, or open another graph | Everything the runs registered, except the outputs of Save, Deploy and Data Output nodes (you named them). Old copies can't be mistaken for the new graph's nodes, which start at id 1 again. |
| Delete a Dataset Input | Nothing: it only names a dataset Data Studio loaded |
| Close or switch the project | Every dataset |
| **Unload from memory** (the Data Input dialog's Memory tab) | The dataset from every place it is held, and the copies pipeline runs made of it; the node reads Not loaded until the next Apply or run |
| A Plot or Dashboard refresh finishes | Its in-memory copies of disk inputs (the plot keeps the rows it read) |
| A Plot or Dashboard is no longer connected (its input was deleted or unwired) | The rows it read; it reads "Not connected" |
| Delete a Plot of a trainer's model | The model's table (`ds_plot_model_<id>`), unless another plot shows it |
| A run that was replaced by a newer run finishes | What the older run registered, except what the newer run uses (it releases once the newer run has ended) |

If a run is still going when you delete or clear, its data goes as soon as the run ends.
While a Data Input's dialog is still loading, the graph can't be cleared or
replaced ("a Data Input is loading its data"). Otherwise the dataset would
arrive after its node was gone.

**Undo and redo** follow the data:

- A dataset no Data Input names any more is released.
- A Data Input that comes back reads **not loaded**, and its Apply loads it again.

**A Table Viewer tab of a removed dataset** stays open as a frozen copy of the
rows it shows. A line over the tab says so: "'mnist_test' was removed (its
Data Input or graph is gone). This tab is a frozen copy of the 10000 rows it
shows." The tab lets go of the dataset's own memory and keeps only its rows.

**A Table Viewer tab of a dataset that is loaded again** with new data (an
Apply with other options, for example) also stays as a frozen copy: "'mnist_test'
was loaded again with new data. This tab is a frozen copy of the 10000 rows it
showed before; open the dataset again for the new rows." If the dataset is
then removed, the line changes to the "was removed" notice.

**A dashboard on a removed dataset** keeps its plots as they were last read,
with a line saying so: "'<name>' is not loaded now (its Data Input or graph is
gone)." The plots update when the dataset is loaded again. A plot open in the
dashboard's plot editor lets go of its table and says where the plot was:
"Open the plot again from the dashboard when it is loaded."

**A dashboard on a sparse vectorizer** keeps its feature matrix in memory
until the project closes: closing the dashboard, deleting or unloading its
Data Input, or opening another graph does not free it. Hover the dashboard's
"Sparse features" line or its Memory tile for the same note.

Two Data Inputs reading files with the same name in different folders get
different dataset names (`mnist`, `mnist_2`). Two Data Inputs reading the
same file still share one table, so nothing is copied.

## Disk caches (Preferences > Storage)

The Engine keeps prepared copies of data on disk, so loading and preparing
are faster next time. **File > Preferences > Storage** (Ctrl+,) lists each
cache:

| Cache | What it holds | Limit |
|---|---|---|
| Temporary (this PC) | Parquet copies of files too large for memory | 30 days, 10 GB |
| Prepared inputs (this project) | Data Input copies, reused the next time a file loads | 30 days, 10 GB |
| Streamed preparations (this project) | The output of preparing data kept on disk, written again each run | 30 days, 10 GB |
| Prepared datasets (this project) | Training data after its preparation steps | 32 entries, 10 GB |
| Saved query answers (this project) | Data Studio and dashboard query results | 512 MB |

Each cache in the list shows:

- what it holds, its folder and **Show in Explorer**;
- its size and number of files;
- its limit;
- **Clear...**, which asks first.

Clearing never changes your files or graphs: what a cache held is made again
when it is needed (slower once). A file a loaded dataset reads stays.

When a project opens, the Engine applies the 30-day / 10 GB rule to that
project's two folders. The temporary cache gets the same rule when the
Engine starts.

Example (MNIST project): Prepared inputs, 1 file, 3 MB, **Clear...** then
**Clear** shows "Cleared 3 MB (1 file)." The next Apply of
`mnist_test.csv` prepares the file again.
