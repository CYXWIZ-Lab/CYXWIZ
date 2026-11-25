---
title: Limits of remote training
description: What remote training cannot do in this build: safety, one node per job, the graphs and data a node takes, devices, and payments.
type: reference
area: remote
audience: user
status: experimental
since: 1.0.8
owner: remote
last_reviewed: 2026-11-25
related:
  - index.md
  - run-a-remote-job.md
  - data-shipping.md
  - ../server-and-cloud/security.md
---

Remote training is experimental. This page lists what it cannot do yet, so
you know before you reserve a node. Training on your own computer has none
of these limits.

## Safety

- **Not for strangers yet.** A server node accepts job assignments on its
  node service without checking who sent them, and it runs every job inside
  its own process with no sandbox. Use remote training only between
  machines and people you trust, on a network you control. The operator's
  view is in [Security notes](../server-and-cloud/security.md).
- What *is* checked: the Engine's connection to a node carries a token the
  central server signed for your reservation, and the node ends the session
  at the reservation's end on its own clock.
- Traffic between the Engine and a node is not encrypted. Keep it on a
  private network or a VPN.

## One node per job

A job trains on one server node. Training one model on several nodes at
once, or splitting a large model across nodes, is not in this build.

## Graphs

A node trains the graph with the same training code as the Engine, but it
cannot run every node:

- **Data preparation steps are refused** ("node '...' is a data preparation
  step"): nodes in the data transform, preprocessing, text processing, time
  series, audio, database, cloud storage, JSON/XML and big data families,
  and every node the Engine runs through its data preparation pipeline.
  Prepare the data in the Engine (for example in Data Studio), save the
  result as a Parquet or Arrow file, and train from that.
- Image transforms (resize, crops, flips, augmentation) do run on the node.
- **Extension nodes are refused** ("This graph uses extension node ...
  Remote nodes do not run extension nodes yet"): C++ plugin nodes and Python
  nodes run only on your computer.
- The graph must compile on the node against the shipped data, as it does
  in the Engine.

## Data

- Only tables: Arrow IPC (`.arrow`, `.feather`, `.ipc`) and Parquet. Other
  files are refused ("files are not supported by this host (use Parquet or
  Arrow IPC)").
- Each Data Input must be applied in this session, or point at a CyxCloud
  file. An empty table is not sent.
- A supplied Test input stays on your computer; test the model in the
  Engine.
- Datasets are sent whole before training starts, so a large dataset takes
  time to reach the node at the start of each job.
- A CyxCloud Data Input is for remote training only: this computer cannot
  preview or train from it yet. See
  [Train from a CyxCloud dataset](cloud-datasets.md).

## The node's devices

- A node trains only on device routes verified on that machine; a job is
  refused "Device problem" when the route is not verified, and "Out of
  memory" when the job does not fit. **Will this job fit?** estimates this
  before you start.
- Training runs in 32-bit floating point, as in the Engine.

## Reservations and payment

- A reservation lasts 10 minutes to 8 hours and can be extended. The clock
  runs whether you are connected or not, and time left is not returned.
- At the end a running job is checkpointed on the node and stopped; it
  resumes when sent again in a new reservation, and its datasets are sent
  again.
- **Payments are not live.** The window shows prices and a quote, your
  balance is "Not tracked yet", and receipts say "Nothing charged".

## The server node itself

- Server nodes are not in the release packages; they are built from source.
  See [Build and install a server node](../server-and-cloud/server-node/install.md).
- On macOS a node reports temperatures and power as 0.
