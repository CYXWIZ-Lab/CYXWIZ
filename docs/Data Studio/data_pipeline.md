# Data Pipeline — User Guide

This is the end-user guide to loading data into CyxWiz Studio and training on
it. It covers what happens when you click Apply in the Data Input dialog, how
the engine decides whether to keep your data in RAM or spill it to disk, and
what the Compile and Train buttons are checking for you.

For the contributor-facing version (class names, file paths, internals), see
the "Data Path Direction" section in the repository root `CLAUDE.md`.

## The 30-second version

1. Drop a **Data Input** node onto the canvas.
2. Open the dialog, pick your CSV, set the Label column.
3. Click **Apply**. A blue "Loading…" bar appears while the engine loads the
   data. You can keep using the dialog or close it while it loads.
4. When the bar turns green, your data is loaded. Click **OK** to close.
5. Click **Compile** to see a report. Fix any red errors it flags.
6. Click **Train**.

That's the whole loop. The rest of this doc explains what the engine does
behind each step, so you know what to expect and how to troubleshoot when
something looks off.

## The two backends: in-memory Arrow vs disk-backed Parquet

Every tabular dataset lives in exactly one of two places:

| Backend | When it's used | What it looks like | Speed |
|---|---|---|---|
| **In-memory Arrow** | File comfortably fits in RAM | Green "In Memory" in the Memory tab | Fastest — all rows sit in RAM |
| **Disk-backed Parquet** | File too big, or you ticked Force | Blue "Loaded via Parquet cache" | Slightly slower — reads pages on demand |

**The engine picks automatically.** On Apply, it compares the CSV file size
to your available RAM. The rule is:

- If the file is **smaller than 75%** of free RAM, it loads in memory.
- Otherwise, it converts the CSV into a Parquet cache on disk and trains
  from that, reading pages lazily as needed.

On a machine with 32 GB RAM free, that threshold is ~24 GB — enough for
almost every real dataset. You only ever hit the disk-backed path for truly
huge CSVs or when you flip the Force flag manually.

### How do I know which one I got?

Open the Data Input dialog and look at the **Memory** tab, "Current Status"
section.

- **"In Memory"** with a green dot → Arrow. The "size" shown is post-
  compaction RAM usage (the engine auto-downcasts int columns where it can,
  so MNIST's 420 MB on disk becomes ~52 MB in RAM).
- **"Loaded via Parquet cache"** with a blue dot → Parquet. The "size"
  shown is the on-disk cache size. Actual RAM usage during training is
  bounded by the OS page cache, not this number.

The Compile report also tells you: look for the `Backing:` line.

## The Force disk-backed checkbox

In the Memory tab, at the bottom, there's an **Advanced** section with a
"Force disk-backed cache" checkbox. What it does:

- **Off (default)**: engine picks automatically by file size.
- **On**: skip the RAM check, always convert to Parquet and train from disk.

When would you turn this on?

- **Testing/benchmarking.** You want to verify the disk-backed code path
  works on a small file before deploying against a giant one.
- **You plan to reopen the project later** and don't want the CSV re-read
  into RAM from scratch — the Parquet cache reload is much faster.
- **RAM pressure from other apps.** If the 75% rule would still leave you
  uncomfortably tight, Force puts the dataset on disk and frees RAM for
  the rest of your workflow.

The flag is saved in the project file, so your choice survives reopening
the project.

## Where does the cache live?

The Parquet cache sits at:

```
<system-temp>/cyxwiz/cache/
```

On Windows that's typically `C:\Users\<you>\AppData\Local\Temp\cyxwiz\cache\`.
On Linux/macOS it's under `/tmp/cyxwiz/cache/`. Each entry is named
`<csv-stem>_<hash>.parquet` where the hash is computed from the CSV's
absolute path, so two CSVs named `train.csv` in different folders get
separate cache files.

**The cache takes care of itself.** Every time the engine starts, and
again after every fresh conversion, it runs a housekeeping pass that:

1. Deletes any cache file older than **30 days** (mtime-based expiry).
2. If the total cache size still exceeds **10 GB**, deletes oldest files
   first until under the cap.

You don't need to clean it manually. If you want to anyway, you can just
delete any `.parquet` file in the cache directory; the next Apply will
rebuild whatever it needs. Files currently being trained on are mmap-
locked by the OS on Windows and won't be deleted — you'll see them
removed on the next prune after training finishes.

## Async load — why the dialog stays responsive

CSV loading runs on a worker thread. What this means for you:

- You can **keep scrolling the preview**, switch tabs, and interact with
  the dialog while data loads.
- You can **close the dialog** (Cancel or the X button) while the load is
  in progress. The load keeps running in the background and finishes into
  the registry. When you reopen the dialog, you'll see "In Memory" (or
  "Loaded via Parquet cache") ready to go.
- The **Apply and OK buttons grey out** during load — you can't
  accidentally kick off a second load while one is already running.
- **Cancel stays enabled** so you can always back out of the dialog.

If you're wondering whether your load actually finished when you come
back: check the Memory tab's Current Status. If it's green/blue, you're
good. If it's still gray "Not Loaded", click Apply again.

## The Compile button — your safety net

Before you click **Train**, click **Compile**. It runs all the same
checks Train will run, but without actually starting training. You get
a popup categorizing any findings into three tiers:

- **[ERR]** red — these block training. Fix them first.
- **[WARN]** yellow — training will run, but you should probably know
  about this. Examples: no validation split, no label column, split
  ratios that don't sum to 1.0.
- **[INFO]** blue — purely informational.

The popup also shows a "Configuration" section with your layers, dataset
name, backing store (in-memory vs disk-backed), split ratios, batch
size, optimizer, and more. This is the single best sanity check before
training — glance at it and make sure everything matches what you
expected.

### Common errors and how to fix them

| Error | What it means | Fix |
|---|---|---|
| `Graph is empty` | No nodes on the canvas | Add nodes |
| `Graph must have a DataInput or DatasetInput node` | Missing data source | Drag a Data Input node in |
| `Data is not loaded - open the node and click Apply` | You have the node but never clicked Apply, or the load failed | Open the Data Input dialog, click Apply, wait for green |
| `Dataset '...' is marked loaded but missing from registry - re-apply the DataInput node` | Project state is stale — the registry was cleared (e.g. closed/reopened project) but the node still thinks it has data | Reopen the Data Input dialog, click Apply |
| `Graph must have at least one model layer` | No Dense/Conv2D/etc. | Add a layer |
| `Graph must have a loss function` | No loss node | Drag in e.g. CrossEntropyLoss |
| `Graph must have an optimizer` | No SGD/Adam/AdamW | Drag in e.g. Adam |
| `Graph contains a cycle` | A link loops back on itself | Remove the circular link |
| `batch_size (N) is larger than the train split (M rows)` | Your batch is bigger than your training set | Lower batch size, or use more data |

### Common warnings

| Warning | What to do |
|---|---|
| `No label column selected` | Open Data Input and pick a Label column, or verify the last column really is your target |
| `Validation split is 0` | Add a DataSplit node and give it a non-zero val_ratio, or accept that training runs without validation |
| `DataSplit ratios sum to X (expected 1.0)` | Open the DataSplit node and rebalance train/val/test |
| `batch_size is more than half the train split` | You'll only get a couple of iterations per epoch — lower the batch size for more gradient updates |

## The Train button — what happens when you click it

1. The engine runs the same Compile pass as the Compile button.
2. If there are any red errors, it shows the same popup (with title
   **"Cannot Start Training"**) and refuses to start. No training
   launched.
3. If there are only warnings or no issues, it proceeds:
   - Loads the model architecture from your graph.
   - Sets up the batcher (Arrow or Parquet depending on backend).
   - Starts the training loop.
4. The training dashboard opens and starts plotting loss/accuracy.

The key guarantee: **you cannot start training on a broken graph or
missing data.** If Train fails silently on you, the compile gate is the
first place to look.

## Troubleshooting

**"I clicked Apply and nothing happened"** — Look at the Memory tab.
If you see a blue loading bar, the load is running; wait for it. If it
says "Not Loaded" with no bar, check the log (engine_log.txt in the
build output directory) for error messages — usually a file path issue
or a format mismatch.

**"Compile says OK but Train crashes the app"** — This shouldn't
happen anymore with the compile gate in place. If it does, it's a bug;
please grab the engine_log.txt and report it.

**"I switched to a new project and my old data is still there"** —
It shouldn't be. Closing or opening a project wipes every Arrow and
Parquet entry from the registry. If you see the old data persisting,
that's a bug.

**"I deleted the Data Input node and the Compile popup still shows
the old dataset name"** — It shouldn't. Deleting a Data Input node
unregisters its dataset. Try clicking Compile again; if the old name is
still there, the Clear All button on the node editor will scrub
everything.

**"The Parquet conversion is slow"** — The CSV-to-Parquet write runs
once per CSV file, on first load (or when the CSV changes on disk and
the cache is no longer fresh). On a 1 GB CSV, expect 15-30 seconds. On
subsequent loads of the same file, it reuses the cache and is near-
instant. The dialog stays responsive the whole time thanks to async.

**"How big can my dataset be?"** — With the disk-backed backend,
there's no hard cap. Training reads pages lazily from the memory-mapped
Parquet file. The practical limit is whatever fits on your disk minus
the 10 GB cache cap, minus the original CSV. For reference, MNIST
(70k x 785) compresses to ~20 MB of Parquet.

## Feedback

If the flow feels wrong, a compile check misses something, or the
cache defaults don't suit your workflow, file an issue with the
engine_log.txt attached. The 75% RAM threshold, the 10 GB cache cap,
and the 30-day expiry are all defaults we can tune.
