---
title: Dataset services
description: The CyxCloud gRPC calls behind datasets: who may call them, uploading and making a dataset, access tokens, and streaming a file to a node.
type: reference
area: server-and-cloud
audience: plugin-developer
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - architecture.md
  - cli.md
  - ../server-node/cloud-datasets.md
  - ../../distributed-training/cloud-datasets.md
---

How a CyxWiz account puts files into CyxCloud, makes them a dataset, and
lets a training node stream them. The Engine and the `cyxcloud` CLI use
these calls; the `.proto` files are `datastream.proto` (package
`cyxcloud.datastream`) and `data.proto` (package `cyxcloud.data`) in the
CyxCloud repository. The Engine compiles verbatim copies of both.

## Who the caller is

A client signs in to the CyxWiz API and presents that token to the gateway
as `authorization: Bearer <token>`. The gateway maps the account to one
stable user id and creates the user on first sight, so the same account owns
the same buckets and datasets whether it arrives from the website, the
Engine or a node.

Only two methods run without a sign-in:

| Method | Why |
| --- | --- |
| `cyxcloud.node.NodeService/RegisterNode` | A storage node's first call (signed with the node's key instead) |
| `cyxcloud.data.DataService/StreamData` | Its credential is the dataset access token in the request |

Everything else fails with `UNAUTHENTICATED` without a valid token.

## Upload, then make a dataset

1. `DataStreamService.UploadFile` (client stream). The first message carries
   `UploadFileMetadata`: `filename` (one plain file name), `size_bytes`,
   optional `content_type`, optional 32-byte BLAKE3 `content_hash`. Every
   following message carries a `chunk`. The gateway accepts the file only
   when the bytes add up to exactly `size_bytes` (and match the hash when
   given); at most 256 MB per file. The file lands in the caller's own
   bucket and the reply carries its `file_id`.
   - `UNAVAILABLE` "no storage node is online to hold the file": start a
     storage node; the gateway stores shards on nodes, not on itself.
   - `FAILED_PRECONDITION` when the gateway runs `--memory-only`: memory
     objects cannot become datasets.
2. `DataStreamService.CreateDataset` with those `file_ids` makes the
   dataset.
3. `DataStreamService.CreateAccessToken` (scope `stream`) makes the token a
   training node presents to `DataService.StreamData`. It lives at most 24
   hours.

`DataStreamService.DeleteDataset` removes a dataset the caller owns, with
its tokens and shares; the files stay in the bucket.

## Stream a file to a node

`DataService.StreamData` sends one stored file, in order, in chunks:

- **A token is required.** `StreamDataRequest.access_token` holds a dataset
  access token. The gateway checks that it exists, is not revoked or
  expired, carries the `stream` or `read` scope, and that its dataset holds
  the requested file. No token: `UNAUTHENTICATED`; a token for another
  dataset: `PERMISSION_DENIED`.
- **No silent skips.** A chunk with no storage location, or one no storage
  node can serve, ends the stream with `DATA_LOSS` naming the chunk ("chunk
  3 of 7 could not be served: ..."). Nothing is marked complete.
- **The bytes are checked twice.** The gateway compares the BLAKE3 hash of
  what it assembled with the file's stored hash before the last chunk; a
  mismatch is `DATA_LOSS`. The last chunk (`is_last = true`) carries
  `stream_sha256` (hex) and `total_bytes`, which the node checks before it
  uses the file.
- `shuffle` is refused (`INVALID_ARGUMENT`): a file stream is ordered.
  `batch_size` (KB) only sets how many bytes travel per message.

What the server node does with the stream is on
[Datasets from CyxCloud on a node](../server-node/cloud-datasets.md).
