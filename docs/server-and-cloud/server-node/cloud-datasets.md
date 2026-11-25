---
title: Datasets from CyxCloud on a node
description: How a server node fetches a remote job's CyxCloud dataset itself, what it checks, and where the file goes.
type: explanation
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - ../../distributed-training/data-shipping.md
  - ../../distributed-training/cloud-datasets.md
  - ../cyxcloud/datasets.md
redirect_from:
  - cyxwiz-server-node/docs/usage/cloud-datasets.md
---

A remote job's Data Inputs reach the node from one of two places:

- **From the Engine.** The Engine sends the whole file over the job's
  connection when the node asks for it. Nothing to set up on the node.
- **From CyxCloud.** A Data Input whose dataset lives in CyxCloud (its
  parameters carry `cloud_dataset_id`, `cloud_file_id` and
  `cloud_file_name`) is not sent by the Engine. The job carries, per such
  dataset, the gateway's address, the file id and a dataset access token the
  Engine obtained. The node streams the file from the gateway itself and
  never asks the Engine for it.

The user's side is in
[Train from a CyxCloud dataset](../../distributed-training/cloud-datasets.md).

## What the node checks

- **Whole or nothing.** The gateway's last chunk carries the SHA-256 of
  every streamed byte and the byte count; both must match what arrived. A
  gateway that cannot serve a chunk ends the stream with `DATA_LOSS` naming
  the chunk, and the job fails with that reason.
- **No access, no fetch.** A job that names a CyxCloud dataset without its
  access token ("carries no access for it") is refused before any fetch.
- **No silent stall.** A gateway that sends nothing for 60 seconds ends the
  fetch ("no data for 60 s"), as does stopping the job.
- **Nothing half-written.** The file is written as
  `<dataset>.<format>.part` and renamed only after it verifies.
- **Two formats.** The file keeps its format from its name in CyxCloud:
  `.parquet` stays Parquet; Arrow IPC names (`.arrow`, `.feather`, `.ipc`)
  become `.arrow`. Remote training accepts only these two.

## Network

The node connects to the gateway address the job names: the gateway of the
Engine's [server address](../server-address.md). An `https://` address is
dialled over TLS; a plain `host:port` (a LAN) is plain gRPC. The access
token is scoped to one dataset and expires with the reservation plus a few
minutes, so a leaked job does not open the account's other data.

## Where the files go

`<job data folder>/<dataset name>.<arrow|parquet>`, next to the files the
Engine sends (the job data folder is `CYXWIZ_NODE_DATA_DIR`, else
`<temp>/cyxwiz-node`). The job's graph is pointed at those paths before
training starts, exactly as for Engine-sent files.
