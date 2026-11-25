---
title: What CyxWiz sends over the network
description: Exactly when the Engine, the installer and the server node contact a server, what they send, and what stays on your computer.
type: reference
area: about
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - ../security/index.md
  - ../security/server-node-hardening.md
  - support-and-versioning.md
redirect_from: []
---

**The Engine sends nothing on its own.** It has no telemetry, no usage
statistics, no update check and no crash upload. It contacts a server only
when you sign in, use a cloud or remote feature, install something, or use a
plugin that downloads. This page lists every such contact found in the
source code.

## The Engine

| When | Contacts | Sends |
| --- | --- | --- |
| You sign in (Account) | The sign-in address in your server settings | Your email and password, once, to `/auth/login`. The password is not saved. |
| You stay signed in | The same address | Every 5 minutes the Engine checks the token's expiry on your computer. When less than 2 hours remain it sends the token to `/auth/refresh` for a new one. |
| You open the server settings in **Preferences** | The server address shown there | A request to each CyxWiz service at that address, to say whether it answers. |
| You open the Cloud Browser or the cloud dataset manager while signed in, or load a cloud dataset in Data Input | The CyxCloud gateway | Your sign-in token, then the listings, uploads and downloads you ask for. |
| You use **Server Connection**, reserve a node or train remotely | The central server, then the reserved server node | Your sign-in token and the reservation; the job (the graph and its settings) and its dataset to the node, or the dataset to CyxCloud. Metrics and the trained model come back. |
| You deploy a model to a server | The central server and the deployment server you choose | The deployment request: the model and the settings you chose. |
| You run `pip install` in the console or Python settings | The Python package index pip is set to use | What pip sends. |
| You run a Python script | Whatever the script contacts | Whatever the script sends. |
| You click a Help link, **Create an account** or a download link | Your web browser opens the page | Nothing from the Engine itself. |

The token is stored on your computer in `session.json`:
`%APPDATA%\CyxWiz\Engine` on Windows,
`~/Library/Application Support/CyxWiz/Engine` on macOS and
`~/.config/cyxwiz/engine` on Linux.

### Plugins that ship with CyxWiz

| Plugin | When | Contacts | Sends |
| --- | --- | --- | --- |
| MuJoCo | You download a model from the model list | GitHub (`api.github.com`, `raw.githubusercontent.com`) | Requests for the model's files, with the user agent `CyxWiz-Engine/1.0` |
| Assistant | You ask it a question | A model runtime on this computer only (`localhost` or `127.0.0.1`; other addresses are refused) | Your question and the context it gathered |

A plugin you install yourself can do what its code does; see
[Plugins, Python nodes and opened files](../security/untrusted-files-and-plugins.md).

## The setup and the installer

- **cyxwiz-setup**, when you run it, downloads the CyxWiz Installer from the
  GitHub release and checks its signature.
- **The installer** reads the signed package list (the catalog) it came with.
  It downloads a newer catalog only when you click **Refresh catalog** or
  **Check for updates**, and downloads GPU packs only when you install or
  update them. These requests use the user agent
  `CyxWiz Backend Pack Service/1.0`.

## The server node

When you sign in on a server node and press **Apply**, the node sends the
central server:

- your sign-in (email and password once; then the token, renewed when less
  than 2 hours remain);
- its registration: a node name, the number of CPU cores, total and free
  memory, its local network address, each device's model and memory, the
  shares you lend, and its measured benchmark;
- heartbeats with its status and free memory;
- the result of each job and the end of each reservation.

Jobs and their data come to the node from the Engine or from CyxCloud and
are kept in the node's data folder. To serve ONNX models the node uses
Microsoft's ONNX Runtime; Microsoft documents that its official Windows
builds can emit trace events to the Windows diagnostic system, which sends
them to Microsoft according to your Windows diagnostic data settings. The
node does not turn these events off today.

**Work offline** on the node's sign-in card keeps the node local: it does not
register with the central server, is not listed and cannot be reserved.

## What stays on your computer

- Projects, graphs, datasets, trained models and checkpoints, unless you
  upload them or train remotely.
- Logs, the Engine's settings and preferences, plugin permission choices and
  Python node approvals.
- Crash dumps that Windows writes, and diagnostic bundles you build in the
  Engine: they are files for you to share, never uploaded.

## Related

- [How CyxWiz handles security](../security/index.md)
- [Harden a server node](../security/server-node-hardening.md)
- [Support and versions](support-and-versioning.md)
