---
title: Train from a CyxCloud dataset
description: Point a Data Input at a file in a CyxCloud dataset so a server node streams it from the gateway instead of from your computer.
type: how-to
area: remote
audience: user
status: experimental
since: 1.0.8
owner: remote
last_reviewed: 2026-11-25
related:
  - data-shipping.md
  - run-a-remote-job.md
  - ../server-and-cloud/cyxcloud/index.md
  - ../server-and-cloud/cyxcloud/cli.md
redirect_from:
  - usage/cloud_datasets.md
---

Use this when your project is on your computer but its dataset lives in
CyxCloud (uploaded with the Cloud Browser or `cyxcloud dataset put`). A Data
Input then names a file in a CyxCloud dataset instead of a file on disk, and
the node you train on streams that file from the gateway. Nothing is
downloaded to your computer.

A cloud Data Input is for **remote** training only in this build: preview,
profiling and training on this computer cannot read a cloud file yet. To
work on the data locally, download it (`cyxcloud dataset get`) and point a
Data Input at the file.

## Before you start

- You are signed in (**Account**, top right) to the server whose CyxCloud
  holds the dataset.
- The dataset has an Arrow IPC (`.arrow`, `.feather`, `.ipc`) or Parquet
  file. Other formats cannot be trained from on a node.

## Point a Data Input at a cloud file

1. Double-click the Data Input to open its dialog.
2. Under **1. Source**, choose **CyxCloud** (the other choice, **This
   computer**, is the file or folder field).
3. Read the line beside the choice: **Connected to <gateway> as <account>**,
   **Connecting...**, **Gateway unreachable at ...** (with **Try again**), or
   **Not connected: sign in to CyxWiz first**. The Engine connects through
   the Cloud Browser (sidebar, **Network** group) once you are signed in;
   open it once per session if the dialog says it is not connected.
4. In **Dataset**, pick one of your datasets or one shared with you (with
   its trust word, file count and size). **Refresh** after an upload.
5. In **File in the dataset**, pick the file. Files that are not Arrow IPC
   or Parquet are greyed with "not for remote training".
6. Read **Detected** ("Parquet table, 149.6 MB, file 2f1c7f8e in dataset
   3b9d2c1e") and what happens with it.
7. Click **Apply**. The node keeps the choice ("Source saved: CyxCloud file
   ... A node trains from it") and the Data Input shows **Not loaded**: there
   is no local copy.

Without a connection the dialog shows **What the graph remembers**
(dataset, ids, size, gateway). The job can still be sent to a node once you
are signed in, but the Engine cannot list datasets or get the node's access
token until it is connected.

## From the Cloud Browser

Right-click a dataset and choose **Use for Training**. A Data Input appears
on the canvas on the dataset's largest Arrow IPC or Parquet file, with its
dialog open so you can pick another file. A dataset without such a file
says so on the browser's error line.

## Train on a node

Reserve a node and start training as in
[Run a remote job](run-a-remote-job.md). When the job is sent, the Engine
gets one access token per cloud dataset (scoped to that dataset, valid until
the reservation ends plus a few minutes) and the node streams the file from
the gateway. The gateway address comes from the
[server address](../server-and-cloud/server-address.md); with no server
address, from `servers.cloud_gateway` in `engine_config.json` (empty: the
central server's host on port 50060).

## What the Data Input keeps

`source_type = cloud`, `cloud_dataset_id`, `cloud_dataset_name`,
`cloud_file_id`, `cloud_file_name`, `cloud_size_bytes`, `file_type`
(`parquet` or `arrow`) and `dataset_name`. A Data Input with these and no
`file_path` is a cloud input; the Engine sends nothing for it.
