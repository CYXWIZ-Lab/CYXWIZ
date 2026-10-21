# Datasets from CyxCloud

A remote job's Data Inputs come from one of two places (TOFIX164):

- **From the Engine.** The Engine ships the whole file over the training
  stream when the node asks (`remote://`, TOFIX118). Nothing to set up on the
  node.
- **From CyxCloud.** A Data Input whose dataset lives in CyxCloud (the graph's
  Data Input carries `cloud_dataset_id`, `cloud_file_id` and `cloud_file_name`)
  is not shipped. The job carries, per such dataset, the gateway's gRPC address,
  the file id and a dataset access token the Engine obtained; the node streams
  the file itself with `DataService.StreamData` and never talks to the Engine
  for it.

## What the node checks

- The stream is accepted only whole: the gateway's last chunk carries the
  SHA-256 of every streamed byte and the byte count; both must match what
  arrived. A gateway that cannot serve a chunk ends the stream with `DATA_LOSS`
  naming the chunk, and the job fails with that reason.
- No token, no bytes: a job that names a CyxCloud dataset without its access
  (`"carries no access for it"`) is refused before any fetch.
- A gateway that goes quiet for 60 s ends the fetch (`"no data for 60 s"`), as
  does stopping the job. Nothing half-written stays in the job's data folder:
  the file is written as `<dataset>.<format>.part` and renamed only after it
  verifies.
- The file keeps its format from its name in CyxCloud: `.parquet` stays
  Parquet, Arrow IPC names (`.arrow`, `.feather`, `.ipc`) become `.arrow`. The
  Engine's plan admits only those two formats for remote training.

## Network

The node connects to the gateway address the job names (the Engine's
`servers.cloud_gateway` setting, by default the central server's host on port
50060). The connection is plain gRPC on the LAN for now; TLS between machines
is TOFIX163 D4. The access token is scoped to one dataset and expires with the
reservation (plus a few minutes), so a leaked job config does not open the
account's other data.

## Where the files go

`<job data dir>/<dataset name>.<arrow|parquet>` next to the Engine-shipped
files; the job's graph is bound to those paths before training starts
(`BindGraphJobDatasets`), exactly as for shipped files.
