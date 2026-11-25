---
title: Run a server node
description: Start the server node daemon and GUI, put the machine on the network, set its options, and follow how a reserved job runs.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - install.md
  - sign-in.md
  - settings.md
  - ../ports.md
  - ../security.md
redirect_from:
  - cyxwiz-server-node/docs/usage/running.md
---

Use this when the daemon and the GUI are built and `--doctor` passes (see
[Build and install a server node](install.md)), and you want the machine
listed for others to reserve.

> **Experimental.** Run nodes only on machines and networks you control:
> some of the node's services accept calls without a sign-in, and jobs run
> inside the daemon with no sandbox. See [Security notes](../security.md).

## What the node needs

- **The server address**: where the central server and sign-in are. Set it
  with `--server=ADDR`, or as `network.cyxwiz_server` in the config file, or
  in the GUI under **Settings > Network**. The forms are on
  [The server address](../server-address.md). `--central-server=ADDR` sets
  the central server alone (default `localhost:50051`).
- **The P2P secret file** (`--p2p-secret-file=PATH`, or
  `network.p2p_secret_file` in the config). The file holds the secret the
  central server signs the Engine's tokens with; the node checks each Engine
  connection against it. The daemon does not start without it. Outside
  Windows the file must be readable by its owner only. A secret on the
  command line (`--p2p-secret=`) is refused, because every user of the
  machine can read the process list.
- **A CyxWiz account**: you sign in to it in the GUI, and the node is
  registered to it.

## Start the node

1. Save the secret in a file (once):

   ```powershell
   Set-Content -NoNewline -Path p2p.secret -Value '<P2P secret>'
   ```

   ```bash
   printf '%s' '<P2P secret>' > p2p.secret && chmod 600 p2p.secret
   ```

2. Start the daemon:

   ```powershell
   .\cyxwiz-server-daemon.exe --server=https://cloud.example.com --p2p-secret-file=p2p.secret
   ```

   It logs the address of each service it opens and ends with
   "Server Daemon is ready!", then "not connected - waiting for user
   allocation".
3. Start the GUI in a second window: `.\cyxwiz-server-gui.exe`.
4. In the GUI, sign in with your CyxWiz account. See [Sign in](sign-in.md).
5. On the **Allocation** page, choose the devices, memory and cores to lend
   and press **Apply**. The node registers with the central server and
   starts its heartbeat; from then on it is listed in the Engine's **Server
   Connection** window.

On another machine of your network, start the daemon there with the same
server address and its own copy of the secret file.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--server=ADDR` | none | The one server address (sets central and sign-in) |
| `--central-server=ADDR` | `localhost:50051` | The central server alone; overrides `--server` for central |
| `--p2p-secret-file=PATH` | `network.p2p_secret_file`, relative to the config file | The P2P secret; required |
| `--config=PATH` | the first that exists of `./config/server_config.yaml`, `./server_config.yaml`, `../config/server_config.yaml`, `%APPDATA%\CyxWiz\server_config.yaml` (`~/.config/cyxwiz/` outside Windows) | The config file; the options here override it. A `--config` file that does not exist stops the daemon. |
| `--ipc-address=ADDR` | `localhost:50054` | Where the GUI connects |
| `--http-port=PORT` | `8082` | The HTTP API for deployed models |
| `--inference-addr=ADDR` | `0.0.0.0:50057` | The gRPC inference service |
| `--tls`, `--tls-cert=`, `--tls-key=`, `--tls-ca=`, `--tls-auto` | off | TLS for the node service (port 50055) only; see [Security notes](../security.md#what-is-not-protected-yet) |
| `--doctor` | | Check whether the node can take jobs, then exit |
| `--benchmark` | | Measure training throughput on each verified route, save it, exit |
| `--help` | | Print the options |

The ports the daemon opens are listed on [Ports](../ports.md); the config
file keys on [Server node settings](settings.md).

## GUI and terminal client options

`cyxwiz-server-gui` is the control client: it connects to the daemon and
shows its jobs, models and settings. It also runs in a terminal, which suits
a machine with no desktop, such as a headless Linux node.

| Option | Default | Meaning |
| --- | --- | --- |
| `--tui`, or `--mode=tui` | | The terminal interface |
| `--gui`, or `--mode=gui` | the GUI | The graphical interface |
| `--daemon=ADDR` | `network.ipc_address` from the config file, else `localhost:50054` | The daemon to connect to; match the daemon's `--ipc-address` |
| `--help`, `-h` | | Print the options |

```bash
./cyxwiz-server-gui --tui
```

The terminal interface has six views: **Dashboard**, **Jobs**, **Models**,
**Deploy**, **Settings** and **Logs**. Press `1` to `6` (or `Tab`) to
switch views, `r` to refresh, `h` for help and `q` to quit. It has no
sign-in or **Allocation** page: sign in and lend the machine from the GUI.

The terminal interface is part of every build of `cyxwiz-server-gui` on
Windows, Linux and macOS while the CMake option `CYXWIZ_BUILD_TUI` is on
(the default). A client built with only one of the two interfaces starts
that one, whichever is asked for.

## How a reserved job runs

1. A user reserves the node in the Engine (**Deploy > Connect to
   Server...**). The central server marks the node busy and gives the Engine
   a token naming the reservation, this node and when the reservation ends.
2. The Engine connects to the node's P2P port (50052) with that token. The
   node checks the token's signature with the P2P secret and refuses an
   invalid or expired one ("Invalid or expired auth token").
3. The Engine sends the graph. The node fetches the job's dataset files,
   whole, from the Engine or from CyxCloud, and trains them with the same
   training code as the Engine. See
   [How data reaches the node](../../distributed-training/data-shipping.md).
4. Jobs run until the reservation ends. When time runs out mid-job, the node
   saves a checkpoint, stops the job and tells the Engine "Reservation
   ended". The checkpoint stays on the node (under the job data folder), so
   the job resumes when it is sent again in a new reservation. An extension
   bought in the Engine moves the node's deadline at once.
5. The node reports each job's result and the end of the reservation to the
   central server.

A Parquet file larger than a quarter of the node's data budget stays on
disk and is read one row group at a time, as in the Engine. See
[Data bigger than memory](../../concepts/data-bigger-than-memory.md).

## If something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| Engine: "Invalid or expired auth token" | The node's secret differs from the central server's, or the reservation is over. Check the secret file; if the node registered under another id, restart the GUI and **Apply** again. |
| The node is not in the Engine's list | Not registered yet (**Apply** in the GUI), or the node cannot reach the central server. |
| `--doctor` FAIL on **Verified routes** | Verify this machine's devices (Engine **Preferences > Devices**, or the installer). |
| A job is refused "Device problem" or "Out of memory" | The node's admission check: the job does not fit, or its route is not verified. |
| A deployed model is not loaded: "the Server Node cannot run extension nodes yet" | The model uses a plugin or Python node the node does not have. Serve it from the Engine instead. |
| The daemon exits: "No P2P secret", or "... can be read by other users; run chmod 600 on it" | Give it the secret file, readable by its owner only. |
| The daemon exits: "--p2p-secret= puts the secret where every user ... can read it" | Write the secret to a file and pass `--p2p-secret-file=PATH`. |
| The daemon exits: "Failed to start NodeService gRPC server on 0.0.0.0:50055; is the port in use (another cyxwiz-server-daemon already running)?" | Another daemon owns the node's ports. One daemon per machine: stop the old one by its process id, or keep using it. |
| **Apply** fails with "Failed to connect to Central Server", the daemon log says "No route to host" | The machine cannot reach the server side (often right after waking from sleep). Check the network, then press **Retry**: it sends Apply again with your allocation and sign-in. |
