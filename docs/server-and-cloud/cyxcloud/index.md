---
title: CyxCloud
description: The CyxWiz storage network for datasets: what it is, who joins it, and the pages that run, use and secure it.
type: landing
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - architecture.md
  - storage-node.md
  - cli.md
  - datasets.md
  - ../../distributed-training/cloud-datasets.md
---

CyxCloud stores datasets for CyxWiz. Your project stays on your computer;
its datasets can live in CyxCloud, and a server node you train on streams
them from there instead of from your Engine.

A **gateway** is the manager and the only part clients talk to. It cuts
every file into 4 MB chunks, encodes each chunk into 10 data and 4 parity
shards, and places the shards on **storage nodes**: machines whose owners
lend disk space. Any 10 shards of a chunk rebuild it, and every shard is
checked against its hash when it is read back.

> **Experimental.** CyxCloud is early: with fewer than four storage nodes a
> single node loss makes files unreadable until the node returns, and the
> traffic to storage nodes is not encrypted yet. See
> [How CyxCloud fits together](architecture.md#known-limits).

## Who does what

| You want to | Use | Page |
| --- | --- | --- |
| Train from a dataset in CyxCloud | The Engine: Data Input > **CyxCloud**, or the Cloud Browser | [Train from a CyxCloud dataset](../../distributed-training/cloud-datasets.md) |
| Upload, list, download or share datasets from a terminal | The `cyxcloud` CLI | [The cyxcloud CLI](cli.md) |
| Lend disk space | A storage node (`cyxcloud node init`, `start`) | [Run a storage node](storage-node.md) |
| Run your own CyxCloud | The gateway, Postgres, Redis, storage nodes | [Run CyxCloud](run-cyxcloud.md) |
| Put one address with TLS in front of it | The front door (`cyxwiz-front`) | [The front door](front-door.md), [TLS](tls.md) |
| Understand the pieces | | [How CyxCloud fits together](architecture.md) |
| Call the dataset services yourself | gRPC `DataStreamService`, `DataService` | [Dataset services](datasets.md) |

CyxCloud's source is its own repository, `CYXWIZ-Lab/cyxcloud` on GitHub,
with release downloads of `cyxcloud-gateway`, `cyxcloud-node` and the
`cyxcloud` CLI for Linux, macOS and Windows.
