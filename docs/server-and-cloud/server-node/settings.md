---
title: Server node settings
description: The server node GUI's pages, the Settings tabs, and every key of server_config.yaml with its default.
type: reference
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - run-a-server-node.md
  - sign-in.md
  - ../server-address.md
  - ../ports.md
---

The node reads its settings from `server_config.yaml`; the GUI's
**Settings** page edits the same file. Command-line options override the
file (see [Run a server node](run-a-server-node.md#options)).

## The GUI's pages

The sidebar on the left of `cyxwiz-server-gui`:

| Page | What it shows |
| --- | --- |
| **Dashboard** | The system overview: devices, load, jobs, connection |
| **Hardware** | The detected CPU, GPUs, memory and disks |
| **Allocation** | What to lend: devices, memory and cores. **Apply** registers the node and starts its heartbeat; **Retry** sends Apply again after a failure. |
| **Analytics** | Metrics over time |
| **Jobs** | Training jobs on this node |
| **Models** | Model files in the models folder |
| **Fine-tuning** | Fine-tuning jobs on a local model |
| **Deploy** | Deploy a model on this node for the [HTTP API](http-api.md) |
| **API Keys** | Keys for the node's API |
| **Settings** | The tabs below |
| **Logs** | The daemon's log |
| **Account** | Who is signed in; sign in or out |

## Settings tabs

**General**: **Node Name** and **Region**, shown in the Engine's node list.

**Network**:

- **CyxWiz server**: **Server address**, its examples, and whether central
  and sign-in answer (**Check again**). See
  [The server address](../server-address.md). Datasets come from the
  gateway address in each job, so there is no gateway field.
- **Certificate authority (optional)**, for an `https://` server whose
  certificate comes from a private CA: **Browse...**, **Clear**, and the
  CA's SHA-256 fingerprint.
- **Separate addresses (development)**: **Central server** and **Sign-in**,
  used only while the server address is empty.
- **This node's own API**: **HTTP API port**, and **Serve this node's API
  over TLS** (writes `network.enable_tls`; see
  [Security notes](../security.md#what-is-not-protected-yet) for what it covers).

**Remote Connection**: connect this GUI to a daemon on another machine
(**Daemon Address**, **TLS Security Settings**). Prefer an ssh tunnel to the daemon's IPC
port; leave **Skip Certificate Verification** off.

**Training**: **Enable Training**, **Max Concurrent Jobs**, **GPU
Allocation**. **API**: **Require Authentication**, **Default Rate Limit**.
These five are saved to the file, but this build of the daemon does not act
on them: what a node lends is set on **Allocation**, and the HTTP API does
not check keys (see [Security notes](../security.md)).

**Save changes** writes the config file the node loaded and names it
("Saved to ..."); **Discard** drops the edits. Saving a new server address
signs the node out, because a sign-in belongs to the server that issued it.
The daemon uses new server settings after it restarts.

## server_config.yaml

The daemon looks for the file in this order: `--config=PATH`, then
`./config/server_config.yaml`, `./server_config.yaml`,
`../config/server_config.yaml`, then `%APPDATA%\CyxWiz\server_config.yaml`
on Windows (`~/.config/cyxwiz/server_config.yaml` elsewhere).

```yaml
node:
  name: CyxWiz Server Node
  region: us-west-2
network:
  cyxwiz_server: https://cloud.example.com   # the one server address; empty: the two keys below
  server_ca_file: ""                          # private CA (PEM) for an https:// server
  central_server: localhost:50051
  auth_api_url: http://127.0.0.1:3002/api
  p2p_secret_file: p2p.secret                 # relative to this file's folder
  ipc_address: localhost:50054
  p2p_address: 0.0.0.0:50052
  terminal_address: 0.0.0.0:50053
  node_service_address: 0.0.0.0:50055
  deployment_address: 0.0.0.0:50056
  inference_address: 0.0.0.0:50057
  http_api_port: 8082
  enable_tls: false
  cert_path: ""
  key_path: ""
  ca_path: ""
deployment:
  enabled: true
  models_directory: ./models
  max_loaded_models: 3
logging:
  level: info
  file: ./logs/server-node.log
```

| Key | Default | Meaning |
| --- | --- | --- |
| `network.cyxwiz_server` | empty | The [server address](../server-address.md). When set it decides `central_server` and `auth_api_url`. |
| `network.server_ca_file` | empty | A private CA's certificate; empty trusts the system's roots |
| `network.central_server`, `network.auth_api_url` | `localhost:50051`, `http://127.0.0.1:3002/api` | Used only while `cyxwiz_server` is empty |
| `network.p2p_secret_file` | empty (required) | The file holding the P2P secret. A secret written into the config as `network.p2p_secret` is not read; the daemon logs an error. |
| `network.*_address`, `network.http_api_port` | as above | The services the daemon opens; see [Ports](../ports.md) |
| `network.enable_tls`, `cert_path`, `key_path`, `ca_path` | off | TLS for the node service (port 50055) |
| `deployment.models_directory` | `./models` | Where the **Models** page looks for model files |
| `logging.level`, `logging.file` | `info`, `./logs/server-node.log` | The log level and file |

The file also holds `training` and `api` sections for the tabs above.

## Where else the node keeps things

| What | Where |
| --- | --- |
| Session and node registration | `%APPDATA%\CyxWiz\server-node\auth.json` on Windows |
| Dataset files and resume checkpoints of remote jobs | `CYXWIZ_NODE_DATA_DIR`, else `<temp>/cyxwiz-node` |
| Device verification results | Shared with the Engine on the same machine |
