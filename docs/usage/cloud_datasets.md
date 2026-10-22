# Datasets that live in CyxCloud

The project stays on your computer; the data can live in CyxCloud. A Data
Input then names a file in a CyxCloud dataset instead of a file on disk, and a
node you train on streams that file from the gateway. Nothing is copied to
your computer for that (owner decision, 2026-10-22).

## Pointing a Data Input at a cloud file

Open the Data Input's dialog. Under **1. Source** choose **CyxCloud** (the
other choice, **This computer**, is the file or folder field as before).

- The line beside the choice says how the connection stands: **Connected to
  <gateway> as <account>**, **Connecting...**, **Gateway unreachable at ...**
  (with **Try again**), or **Not connected: sign in to CyxWiz first**.
  The Engine connects through the Cloud Browser (sidebar, Network group) once
  you are signed in; open it once per session if the dialog says it is not
  connected.
- **Dataset**: your datasets and the ones shared with you, with their trust
  word, file count and size. **Refresh** after an upload.
- **File in the dataset**: the files with their format and size. Only Arrow
  IPC (`.arrow`, `.feather`, `.ipc`) and Parquet files can be trained from on a
  node; others are greyed with "not for remote training".
- **Detected** says what you picked ("Parquet table, 149.6 MB, file 2f1c7f8e
  in dataset 3b9d2c1e") and what happens with it.

**Apply** saves the choice on the node. The strip shows the file as **Not
loaded** with "149.6 MB in CyxCloud, no local copy". Preview, Profile and the
columns come with the sample read (next step of TOFIX164); a node trains from
the file as it is.

Without a connection the dialog shows **What the graph remembers** (dataset,
ids, size, gateway): the job can still be sent to a node once you are signed
in, but the Engine cannot list datasets or mint the node's access token.

## From the Cloud Browser

Right-click a dataset, **Use for Training**: a Data Input appears on the
canvas on the dataset's largest Arrow IPC or Parquet file, with its dialog
open so you can pick another file. A dataset without such a file says so on
the browser's error line.

## Training on a node

See `server-connection.md`, "Datasets that live in CyxCloud": when the job is
sent, the Engine mints one access token per cloud dataset and the node
streams the file from the gateway. The gateway address comes from the config
(`servers.cloud_gateway`; empty means the central server's host on port
50060).

## Parameters the node carries

`source_type = cloud`, `cloud_dataset_id`, `cloud_dataset_name`,
`cloud_file_id`, `cloud_file_name`, `cloud_size_bytes`, `file_type`
(`parquet` or `arrow`), `dataset_name`. A Data Input with these and no
`file_path` is a cloud input; the job plan ships nothing for it.
