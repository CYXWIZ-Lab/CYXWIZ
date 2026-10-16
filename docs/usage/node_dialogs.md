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
folder holds nothing in memory: images are decoded per batch at the Resize
size.

### The data strip and the tabs

Above Data Input's tabs a strip says what the node loads and how far it got,
on every tab: the file or folder, its state (*Not loaded*, *Loading: ...*
with the seconds so far, *Loaded in 1.79 s as sentiment_mental_health
(text)*, or a red *Not loaded* with the reason and what to do), the counts
(*52,681 samples, 7 classes, 30.0 MB on disk*) and how the data is held
(*30.4 MB in memory*, *kept on disk (Parquet ...)*, *decoded per batch at
the Resize size*). When an Apply fails while an earlier load stays, the
strip keeps the earlier load and says the last Apply failed.

The tabs are **Source**, **Preview**, **Profile**, **Rows and columns**
(which columns to load, then the rows to skip and the row limit),
**Memory** and **Audit**.

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
tagging** (token, tag, POS and sentence columns) or **Time series**. A
folder of images is used as images (**Labels from**: class folders, or a
flat folder and a labels CSV), a folder of audio as audio (labels, sample
rate, length), a folder of text files as text. Training, validation and
test roles come from the graph's Data Split, not from this node.
