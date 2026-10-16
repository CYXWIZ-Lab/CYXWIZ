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
