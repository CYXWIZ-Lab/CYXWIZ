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
