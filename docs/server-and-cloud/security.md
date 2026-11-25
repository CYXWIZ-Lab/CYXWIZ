---
title: Security notes for server operators
description: What protects a server node, CyxCloud and their clients today, what does not yet, and how to run a node safely meanwhile.
type: explanation
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - ports.md
  - server-node/run-a-server-node.md
  - cyxcloud/tls.md
  - ../distributed-training/limits.md
---

Remote training and the server node are experimental. This page says what
the current build protects and what it does not, so you can decide where to
run a node. Until the gaps below are closed, run server nodes only on
machines and networks you control, for people you trust.

## What is protected

- **Reserved training sessions.** The Engine connects to a node's P2P port
  with a token the central server signed for that reservation. The node
  checks the signature with the P2P secret and refuses a missing, invalid or
  expired token. The session ends at the reservation's end time on the
  node's own clock: the node checkpoints and stops a running job, and an
  extension only moves that time when a new signed token says so.
- **The P2P secret** is read only from a file kept to its owner; the daemon
  refuses a secret on the command line or in the config file.
- **No user code runs on a node.** A graph arrives as node types and
  parameters. A graph or model with an extension node (a C++ plugin node or
  a Python node) is refused, and data preparation steps the node cannot run
  are refused before training.
- **Dataset access.** A CyxCloud dataset reaches a node only with an access
  token scoped to that one dataset and valid until the reservation ends plus
  a few minutes. Streamed files are checked whole (byte count and SHA-256)
  before use.
- **Clients and the server side.** An `https://` [server address](server-address.md)
  is always checked against trusted certificates (or the CA you give);
  plain `http://` is refused to anything but this machine or a private
  network. No client has a switch that skips the check.

## What is not protected yet

| Gap | What it means for you |
| --- | --- |
| **Job assignments are not authenticated.** The node service (port 50055) accepts a job from any caller, whatever node id it names. | Anyone who can reach port 50055 can queue a training job that names data on the node's disk. Allow 50055 only from the central server. |
| **The terminal service (port 50053) does not check who calls.** On Linux and macOS it opens a shell session as the daemon's user. | Never open 50053 to other machines. |
| **Deployment, inference and the HTTP API do not check callers.** Ports 50056, 50057 and 8082 answer anyone, without TLS; **Require Authentication** in the GUI is not applied. | Open them only to clients you trust, or keep them closed. |
| **No sandbox.** A job runs inside the daemon process, with its user's rights, files and network. The Docker sandbox code is not used. | Run the daemon as an ordinary user with no access to data you would not share, on a machine set aside for it. Never run it as root or administrator. |
| **TLS only on the node service.** `--tls` and **Serve this node's API over TLS** cover port 50055 only; the P2P, terminal, deployment, inference and HTTP ports are plain. | Keep node traffic on a private network or a VPN. |
| **No limits on what a job asks for** beyond the admission check (memory, verified device). | A large or malformed graph can tie up the node until the reservation ends. |
| **CyxCloud storage nodes**: their shard port is plain on the network; the shared secret that guards it travels in clear. | Keep storage nodes and the gateway on a private network. See [CyxCloud architecture](cyxcloud/architecture.md#who-a-node-is). |
| **Payments are not live.** Reservations show a price but nothing is charged or paid. | Do not offer a node to strangers expecting to be paid. |

## Run a node safely meanwhile

1. Use a machine (or a virtual machine) that holds nothing private.
2. Create an ordinary user for the daemon and start it as that user.
3. In the firewall, allow 50052 from the machines whose Engines will train
   here and 50055 from the central server; block 50053, 50056, 50057 and
   8082 from other machines. See [Ports](ports.md).
4. Keep the P2P secret file readable by that user only.
5. Prefer a private network or a VPN between the Engine, the node and the
   server side, and an `https://` server address.
6. Stop the daemon when you are not lending the machine.

## Report a problem

Report security problems as the repository's `SECURITY.md` asks, not in a
public issue.
