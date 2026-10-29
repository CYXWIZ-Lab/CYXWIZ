# Node dialogs

Double-click a node (or right-click it and choose **Configure...**; Data Studio
has **Open dialog**) to open its dialog.
The node's settings are saved into the graph when you press **Apply** or
**OK**.

## The buttons every dialog shares

- **Cancel** puts back the settings the node had when the dialog opened and
  closes it. It is always available, also while the dialog is loading.
- **Apply** saves the settings and keeps the dialog open.
- **OK** saves and closes.

While a dialog is loading or running something (a Data Input load, a Data
Convert run), Apply and OK are off; hover them to see why. Dialogs that know
when you changed something (Embedding, Word Embeddings) keep Apply off until
you do ("Nothing changed").

A file field is a path box with **Browse...** beside it; typing a path works
as well as browsing.

## Data Input

Data Input reads a file or a folder: a table (CSV, TSV, JSON Lines, Parquet,
Feather, Arrow / IPC, a ZIP of text documents; HDF5 files can be inspected),
text, a token-tagging table, a time series, a folder of images or a folder of
audio. Databases come in through Data Studio's SQL Workbench, not through Data
Input.

### Table format options

- **Quote character** (default `"`): a value inside it may hold the
  delimiter and line breaks; a doubled quote inside is one quote. Leave it
  empty to read quotes as ordinary text. Preview, Apply and pipeline runs use
  the same character.
- Files are read as UTF-8; a byte order mark at the start is skipped.
- **Limit Rows** skips the first rows and caps how many are loaded. To keep
  only some rows, filter them in Data Studio's Prepare step or with a Filter
  or SQL node.

### Text needs a node that turns it into numbers

Data Input loads text and its label; a node after it turns the text into
numbers: a **Text Tokenizer** (word ids), a **TF-IDF** or **Count**
vectorizer, or **Word Embeddings**. Compile refuses a text input that reaches
the model without one (*CW-C-0203*, "gives text, and nothing turns it into
numbers") and offers **Add a Text Tokenizer**: it adds the node after the
Data Input, takes over its links, fills in the text and label columns and
opens its dialog.

Data Input's Text options set only the layout and the text and label
columns. Under **Preprocessing (done by nodes)** the dialog names the nodes
after it that turn the text into numbers (*Tokenized by Tokenizer (word),
vocabulary sentiment_analysis_vocab.txt, 64 ids per row*), with **Open it**
and **Show on canvas**, or offers **Add a Text Tokenizer**. A new Text
Tokenizer takes its text and label columns from the Data Input before it.
Apply reads the texts (an empty text is not a sample) and builds no
vocabulary: the sentiment CSV loads in about 3 s (52,681 samples, 7
classes, 30.4 MB).

### Images need a Resize, audio a feature node

Data Input only loads images and audio. A **Resize** after the Data Loader
sets the size every image is decoded at; Compile refuses an image graph
without one (*CW-C-0204*) and offers **Add a Resize**. Audio features come
from a **Mel Spectrogram**, **Spectrogram** or **MFCC** node after the Data
Loader; Compile refuses an audio graph without one (*CW-C-0205*) and offers
the three. The added node goes after the Data Loader's data output (the
labels keep their link) and opens its settings.

Data Input's Image options set only the layout, the folder and (flat
layout) the labels CSV; its Audio options the layout, folder, labels CSV,
sample rate and length ("seconds, padded or cut"). Audio is always mixed to
mono. Under **Preprocessing (done by nodes)** the dialog names the Resize
(*Resized by Resize 64x64, 64 x 64*) and Normalize nodes, or the audio
feature node, with **Show on canvas** (and **Open it** for nodes with a
dialog); when one is missing it offers **Add a Resize**, **Add Normalize**
(after the Resize) or **Add Mel Spectrogram / Spectrogram / MFCC**. An image
folder holds nothing in memory: when training starts the images are decoded
once at the Resize size into cache/images (see image_transforms.md).

### The data strip and the tabs

Above Data Input's tabs a strip says what the node loads and how far it got,
on every tab: the file or folder, its state (*Not loaded*, *Loading: ...*
with the seconds so far, *Loaded in 1.79 s as sentiment_mental_health
(text)*, or a red *Not loaded* with the reason and what to do), the counts
(*52,681 samples, 7 classes, 30.0 MB on disk*) and how the data is held
(*30.4 MB in memory*, *kept on disk (Parquet ...)*, *decoded once at
the Resize size into cache/images when training starts*). When an Apply fails while an earlier load stays, the
strip keeps the earlier load and says the last Apply failed.

The tabs are **Source**, **Preview**, **Profile**, **Rows and columns**
(which columns to load, then the rows to skip and the row limit; not shown
for an image or audio folder, which has no rows to read), **Memory** and
**Audit**. For an image folder, **Preview** shows its first 24 images with
their class once it is loaded (Explore > Images in Data Studio shows them
all), and **Profile** shows the image profile (counts, classes, sizes,
prepared sets; see Data Studio > Profile).

### Source: the file or folder, then how its rows are used

**1. Source** has one field, **File or folder**, with **Browse file...**
and **Browse folder...**. Under it, **Detected** says what it is:

- a file: its format and columns (*CSV table, 30.0 MB, 3 columns: (unnamed),
  statement, status*), with **Format options** (format, delimiter, header
  row, decimal separator, quote character, missing values; a ZIP's member
  list). Change the format only when the extension is wrong.
- a folder: what its files are (*Image folder: flat folder, 25,000 images*,
  or *2 class folders (cat, dog)*), with **Scan again**.

**2. Use the rows as** offers only the uses that fit the source. A table
file can be used as **Table** (a label column; every other column is an
input feature), **Text** (a text column and a label column), **Token
tagging** (token, tag, POS and sentence columns) or **Time series** (a
label column and a **Time column**: the rows are put in that column's order
on load, missing times last, so windows and a chronological split follow
time; a column named date, time, timestamp, datetime or ds is offered with
**Use it**; with none the rows keep the file's order; the strip then reads
*as sales (time series by date)*; a table kept on disk cannot be put in
order and the load says so). A folder of images is used as images (**Labels from**: class folders, or a
flat folder and a labels CSV), a folder of audio as audio (labels, sample
rate, length), a folder of text files as text. Training, validation and
test roles come from the graph's Data Split, not from this node.

## Text Tokenizer

The tabs are named by what you decide:

- **Words**: the text and label columns, **Split into** (Spaces, Words,
  Characters, WordPiece, Byte BPE; Byte BPE adds its starting units and
  space handling) and **Lowercase first**. A vocabulary card shows the file
  the node uses (*sentiment_analysis_vocab.txt, 5000 tokens, min frequency
  5*), warns when entries are broken bytes from before 2026-10-16 (rebuild
  it), and has **Build and save vocabulary** and **Inspect file**.
- **Vocabulary**: source CSV, text column, minimum frequency, maximum size,
  the file and "build and save it when the file is missing".
- **Length**: the output (padded token columns, causal document windows,
  decode token IDs, round-trip check), maximum length, pad value, pad and
  cut.
- **Try it**: type a sample; its tokens show as you type ([UNK] in orange,
  hover for the id) with the count against the maximum length, and the ids
  and round trip under a disclosure.

Help for each field is the hover text on its label.

## Data Split

A bar shows the three parts with their rows from the graph's data (*train 80%
about 42,145, val 5,268, test 5,268*), above the Train / Validation / Test
sliders and their sum (Apply scales them to 1.00). **Seed** sets the
shuffled order the parts are cut from; **Stratified** keeps each class's
share in every part. *From the graph* names the dataset and its samples, with
**Refresh from graph**; **Details** holds the resolved manifest (origins,
schemas, policy, source ids, leakage checks). An old graph using Data
Boundary v1 shows a caution and **Migrate graph to Dataset v2**.

## Data Loader

Six tabs: **Training** (epochs, validate every, batch size with about how
many batches an epoch has, seed, model seed, shuffle, drop the last short
batch), **Batches** (gradient accumulation, prefetch), **Balance** (class
balancing for training batches: mode, target per class, seed),
**Checkpoints** (best validation checkpoint, early stop patience, checkpoint
folder, start from an earlier checkpoint), **Generation** (samples the model
writes during training: prompts, chat template, stop texts, decoding, when;
see [Generation previews](generation_previews.md)) and **Advanced** (worker
threads, pin memory, report interval). The Data Loader is the one place
batching is set.

## Row Filter

**Keep the rows where** lists the conditions as rows: a column, an operator
in words (*is*, *is not*, *is more than*, *is less than*, *is at least*, *is
at most*, *is empty*, *is not empty*) and a value. *is empty* matches a
missing value, and for text also an empty string (a CSV's empty cell); the
two empty tests take no value. Rows after the first join with **and** or **or**
(*and* binds first: `a and b or c` keeps rows matching both a and b, or c).
**Add condition** adds a row; **Remove** takes one away (a filter keeps at
least one).

Once the step's input is known (the pipeline ran it, or it is a loaded Data
Input), the column is picked by name from its columns, each shown with what it holds (numbers, text, true/false),
and the value follows the column: a number box, a text box, or a true / false
choice. Text and true/false columns compare only with *is* / *is not* (and the
empty tests). Columns holding dates or lists are counted but cannot be
compared; a column with no name (a CSV's index column) is listed as
*(unnamed)* and cannot be picked. Before the
first run the column is typed by name, and the value's type is read from what
is typed (`5` is a number, `true` a true/false, anything else text).

**Runs as** shows the condition the step runs, for example
`"status" != 'Normal' AND "statement" IS NOT EMPTY`, and after a run how
many rows passed (*36,338 of 53,043 rows pass (last run)*, while the
condition is the one that ran); OK and Apply stay off, with the reason on hover,
until every row can run (a column picked, a number where the column holds
numbers, a column the input has). Reopening the dialog shows the saved
condition as rows again. A condition written by hand with brackets in
Properties does not fit rows: the dialog shows it as text to edit, with
**Start over with rows**.

## Data Output

**File** is the file the step writes, with **Browse...**; left empty, the
step writes to the project's exports folder, named after the table it reads.
**Format** is CSV or Parquet; switching it also switches the file's
extension. A relative file path shows **Relative to**: the project folder
(like Data Input paths) or the exports folder. **Writes to** shows the full
path the run will write, and after a run **Last run** shows the file it wrote
and when, or why it failed.

## Data Convert

Converts a table file to another format once, with an optional sidecar
manifest. Six tabs:

- **Source**: the input file with **Browse...** and its format (*Auto* reads
  it from the extension). An XLSX input adds the worksheet (blank is the
  first) and the start column; Plain Text notes that it gives one column
  named `text`.
- **Output**: the output file with **Save as...**, **Use the input's name
  with this format's extension**, the format (picking one switches the
  file's extension; a mismatch is flagged), and **When writing**: overwrite
  an existing file, create missing folders, write a conversion manifest
  beside it.
- **Options**: *Reading the input* (delimiter, detected or typed; decimal
  separator; header; line breaks inside quoted values; skip rows) and
  *Writing Parquet* (compression, row group size). Settings that do not apply
  to the chosen formats are dimmed with the reason.
- **Preview**: **Preview schema** reads the input in the background and shows
  its rows, columns, each column's type and whether it can be missing, and
  sample rows.
- **Run**: **Run conversion** (the tab's main button, with its progress), then
  the output path, rows written, columns, size and manifest. The write
  finishes even if the dialog closes; the footer's Apply saves the settings.
- **Log**: the messages of this dialog session, with **Clear**.
