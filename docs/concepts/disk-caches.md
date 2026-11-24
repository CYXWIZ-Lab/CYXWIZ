---
title: Disk caches
description: The prepared copies of data the Engine keeps on disk, their limits, how to clear them, and how prepared training data is reused between sessions.
type: explanation
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - loaded-data.md
  - data-bigger-than-memory.md
  - steps-learn-from-train-rows.md
---

The Engine keeps prepared copies of data on disk, so loading and preparing
are faster next time. Clearing a cache never changes your files or graphs:
what a cache held is made again when it is needed (slower once).

## The caches

**File > Preferences > Storage** (Ctrl+,) lists each cache:

| Cache | What it holds | Limit |
| --- | --- | --- |
| Temporary (this PC) | Parquet copies of files too large for memory | 30 days, 10 GB |
| Prepared inputs (this project) | Data Input copies, reused the next time a file loads | 30 days, 10 GB |
| Streamed preparations (this project) | The output of preparing data kept on disk, written again each run | 30 days, 10 GB |
| Prepared datasets (this project) | Training data after its preparation steps | 32 entries, 10 GB |
| Saved query answers (this project) | Data Studio and dashboard query results | 512 MB |

Each cache shows what it holds, its folder and **Show in Explorer**; its
size and number of files; its limit; and **Clear...**, which asks first. A
file a loaded dataset reads stays.

When a project opens, the Engine applies the 30-day / 10 GB rule to that
project's two folders. The temporary cache gets the same rule when the
Engine starts. **Tools > Monitoring > Clear Caches...** lists the caches
with their sizes and clears the ones you tick.

Example (an MNIST project): Prepared inputs holds 1 file of 3 MB.
**Clear...**, then **Clear**, shows "Cleared 3 MB (1 file)." The next Apply
of `mnist_test.csv` prepares the file again.

## Prepared training data between sessions

**Prepared datasets** hold the table training reads after the pipeline run
(Execute Pipeline, or the run Train starts for SQL, Filter Rows and other
steps). After training on it, the Console says "'ds_filter_50' is kept on
disk for the next session". In a later session, Train with nothing changed
before the Data Split loads it and runs no data step: "training reads
'prepared_4ba1042cc5fa4c2b', prepared in an earlier session; the data steps
do not run".

### The prepared dataset's key

Each entry is found by a key: a hash of everything that decides the rows
training reads. Train computes it first and logs it (Debug, "prepared data
key b5979ba17bfce829"); the same key finds the entry, a different key makes
a new one. The key holds:

- **the steps before the Data Split**: every node from the Data Input to
  the step that feeds the Split, with each step's type and settings, and
  the links between them;
- **the Data Input**: its dataset name, its columns and their types, and,
  for data with no file behind it, its rows' content;
- **the source file**: its size and modified time (editing or replacing the
  file makes a new entry);
- **files a step reads**: a vocabulary, a lookup table or a Data Convert
  target named in a step's settings, by size and content;
- **the Data Split's rows**: its shares, seed, label column and Stratified,
  and the Data Loader's Shuffle, since the steps learn from the Train rows
  (see [Steps learn from the Train rows](steps-learn-from-train-rows.md));
- **the output kind**: table or sparse features (a vectorizer's Output
  format);
- **the entry format**: the Engine's own version of what an entry holds.
  When it changes, older entries no longer match and the data is prepared
  once again; the old entries go by the 30-day rule or **Clear...**.

Not in the key, so these keep the entry: the model and everything after the
Data Split, renaming a step, moving nodes on the canvas, training settings
(epochs, batch size, learning rate).

Not kept at all (each Train prepares again, and the Console says why):
paths with an SQL step (a query can read any file), or a scaler set to
**Save fitted state** (an entry would skip writing the file: "'Standard
Scaler' keeps fitted state"). A scaler that fits (the default) is kept: the
key holds every row it learns from. With **Use saved values** the values
file is in the key, by content.

A table kept on disk (a result larger than memory allows) is kept as its
file: the entry links to it, or copies it on another drive, and a later
session opens it on disk again instead of streaming the steps over the
data.

### Example

A graph of a Blobs training table, a Standard Scaler and a Data Split with
supplied Validation and Test inputs:

1. The first Train logs "'ds_operator_StandardScaler_5' is kept on disk for
   the next session" (key 12cbbcf99101b971).
2. After reopening the Engine, Train logs "the prepared data is on disk from
   an earlier session" and "training reads 'prepared_12cbbcf99101b971',
   prepared in an earlier session; the data steps do not run".
3. The batchers read the kept Train / Validation / Test marks: "partition
   '__partition__'=0 selected 384 / 480 rows", then 48 and 48.

With sparse features (3,000 reviews, TF-IDF with sparse output) it works the
same way: the first Train prepares the data ("stratified split assigned
2399 train / 299 val / 302 test rows"), and after a restart Train logs the
same key, runs no data step, and serves the same 2399 / 299 / 302 marked
rows. Setting the Split's train share to 0.7 makes another key.

## Related

- [Loaded data](loaded-data.md)
- [Data bigger than memory](data-bigger-than-memory.md)
- [Steps learn from the Train rows](steps-learn-from-train-rows.md)
