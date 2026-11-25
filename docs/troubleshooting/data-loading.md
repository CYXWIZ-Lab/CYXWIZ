---
title: Data loading problems
description: Fix Data Input loads that fail or are refused, data that is not loaded at Train, and the memory guard and data budget messages.
type: troubleshooting
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - ../main-window/data-input-dialog.md
  - ../concepts/data-bigger-than-memory.md
  - ../concepts/loaded-data.md
  - compile-and-training.md
---

A Data Input node loads its file when you click **Apply** in its dialog.
The dialog's status line shows the outcome; the Console has the detail.
How the Engine decides to keep a table in memory or on disk is explained in
[Data bigger than memory](../concepts/data-bigger-than-memory.md).

## "Data is not loaded - open the node and click Apply"

Shown in the compile report with code `[CW-C-0201]`.

**Cause:** the Data Input has its settings but no loaded data. A reopened
project keeps each Data Input's settings, not its data.

**Fix:** select **Open its dialog** on the finding, then **Apply**. Train
also loads the Data Inputs that feed the Data Loader by itself before the
first batch.

## "Dataset '...' is marked loaded but missing from registry - re-apply the DataInput node"

**Cause:** the node says its data is loaded, but the Engine no longer holds
it, for example after **Unload from memory** or a project change.

**Fix:** open the Data Input and click **Apply** again.

## "Failed to load data"

The dialog's status line shows this or the loader's own reason.

**Cause:** the file could not be read in the chosen format: a wrong
delimiter or encoding, a damaged file, or a folder layout the image or audio
loader does not recognise.

**Fix:**

1. Read the reason in the status line and the **Error** line in the Console.
2. Check the source settings in the dialog (format, delimiter, header row,
   label column).
3. Click **Apply** again.

The formats and their options are in
[Data Input formats](../main-window/data-input-formats.md).

## "... Previous loaded dataset remains active."

Not a second error. A new **Apply** failed, and the node keeps the data it
had loaded before. Fix the cause of the first part of the message and apply
again.

## "Apply refused by dataset audit. ..."

**Cause:** the dataset loaded, but its audit found errors (for example
broken values or a label that cannot be read), so the Engine did not accept
it. The list of findings is in the dialog.

**Fix:** correct the file or the dialog's settings the findings name, then
click **Apply**. If a dataset was loaded before, it stays active.

## "No label column selected - apply the Data Input node so the compiler can inspect its schema"

**Cause:** the compiler cannot find the label column of a tabular dataset.

**Fix:** open the Data Input, apply it, and choose the label column. When a
data step makes or renames the label, pick it in the Data Split's **Label
Column** in Properties.

## "Selected label column '...' is not present in registered dataset '...'"

**Cause:** the label column you chose is not in the loaded table, for
example after you loaded another file.

**Fix:** choose a column that exists, then compile again.

## "Train waits while a Data Input is loading its data. It can start when that ends."

Shown when you hover a greyed **Train**. Wait for the load to finish; the
Data Input dialog shows its progress.

## "Memory risk before training"

**Cause:** preparing the data before training (a step before the Data
Split) may use most of the free memory. The window shows the step, the
estimate, the free memory and the safe budget. "Nothing has started yet."

**Fix:** select **Cancel** (the default) and reduce the data, or use a
disk-backed Data Input (its **Memory** tab). **Continue anyway** goes on, but
the system may page to disk and slow everything down.

## "the memory guard stopped '...': the Engine grew by ... while it ran, the limit is ..."

**Cause:** a data step made the Engine grow past the safe memory limit, so
the Engine stopped it before memory ran out.

**Fix:** use fewer rows or columns (add a Sample Rows or Select Columns
step), then run again.

## "...: its input (N rows) is on disk and needs about ... of memory to read, over the data budget of ..."

**Cause:** the step's input is kept on disk because it is larger than the
data budget allows in memory, and this step must read it into memory.

**Fix:** select fewer columns in the Data Input, or raise **Data budget
(MB)** in **File > Preferences... > General**, under **Data**.

## "Cross-validation reads the prepared table from memory, and '...' is kept on disk."

**Cause:** cross-validation needs the prepared table in memory, and this
one is larger than the data budget allows.

**Fix:** raise the data budget in **Preferences > General**, or use fewer
rows (a Sample Rows step).

## "'...' gives text, and nothing turns it into numbers before the model"

Code `[CW-C-0203]`. Local Debug says "Text reaches the model untokenized".

**Cause:** a text column goes straight into the model.

**Fix:** add a Text Tokenizer, or a TF-IDF or Count vectorizer, after the
Data Input. The compile report's **Add a Text Tokenizer** button does it.

## "'...' gives images, and no Resize sets their size"

Code `[CW-C-0204]`. The Data Input dialog says "No Resize yet: images of
different sizes cannot share a batch."

**Cause:** an image dataset has images of different sizes and no Resize
node.

**Fix:** add a Resize after the Data Loader. 64 x 64 trains fast; 224 x 224
keeps more detail. The compile report's **Add a Resize** does it.

## Related

- [The Data Input dialog](../main-window/data-input-dialog.md)
- [Data bigger than memory](../concepts/data-bigger-than-memory.md)
- [What the Engine keeps loaded](../concepts/loaded-data.md)
- [Fix missing values before training](../user-guide/graph-editor/fix-missing-values.md)
