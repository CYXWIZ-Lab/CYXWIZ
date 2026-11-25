---
title: Run a storage node
description: Lend disk space to CyxCloud with the cyxcloud CLI: set up, start, watch, keep running at boot, and what the node keeps.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - cli.md
  - architecture.md
  - tls.md
  - ../ports.md
---

Use this to offer disk space to CyxCloud from a Linux, macOS or Windows
machine. The node holds shards of other people's files, never a whole file,
and the gateway checks every shard it reads back against its hash. Shards
are not encrypted on the client side yet, so a node holds readable pieces.

## Before you start

- The `cyxcloud` CLI and `cyxcloud-node` from a CyxCloud release, side by
  side or on the `PATH` (or `CYXCLOUD_NODE_BIN` naming the node binary).
- A CyxWiz account, and the server address of the CyxCloud you join.
- Port 50051 (or the one you choose) open in the firewall to the gateway.

## Steps

1. Sign in once; the node is registered to this account:

   ```text
   cyxcloud login --server https://cloud.example.com
   ```

   For a server with a private CA, add `--ca-cert ca.crt`; `node init`
   copies it into the node's config. Plain `http://` is accepted only to
   this machine or a private network.
2. Set the node up. In a terminal it asks three things; otherwise pass the
   flags:

   ```text
   cyxcloud node init --data-dir /srv/cyxcloud --offer 500GB --name storage-1
   ```

3. Start it in the background. It waits until the node has registered (up
   to 2 minutes; opening a large store takes a while):

   ```text
   cyxcloud node start
   ```

4. Check it with `cyxcloud node status`, and `cyxcloud doctor` when
   something is wrong.
5. Optional: start the node with the machine,
   `cyxcloud node service install` (below).

## node init

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `~/.cyxcloud/node/data` | The folder for the shards |
| `--offer` | 90 % of free space, at most 500 GB | For example `700GB`, `1TB`; more than the free space is refused |
| `--name` | the machine's host name | Shown in lists |
| `--port` | 50051 | The port the gateway dials for shards |
| `--status-port` | 9090 | The local port `node status` reads (this machine only) |
| `--public-address` | this machine's address towards the server | What the gateway dials back |
| `--force` | | Set up again: the key and node id are kept; the folder and offer are asked again |

It writes, all owner-only (mode 600 on Linux and macOS; on Windows an ACL
for the user, SYSTEM and Administrators only):

| File | What |
| --- | --- |
| `~/.cyxcloud/node/config.toml` | The node's configuration |
| `~/.cyxcloud/node/key` | The Ed25519 key the node id is derived from |
| `~/.cyxcloud/node/access_secret` | What the gateway presents on shard traffic |
| `~/.cyxcloud/node_credentials.json` | The account the node belongs to |

## What the node runs on

The account sign-in is needed once, to set the machine up. After that the
node runs on credentials that do not expire: its key signs every
registration with the gateway (whose 24-hour node token is renewed by
registering again), and a machine key from the CyxWiz API carries its
heartbeats there.

When the API no longer knows the machine, the node logs once:
"Registering this machine needs a current sign-in: run `cyxcloud login`,
then `cyxcloud node sign-in`, then restart the node". `cyxcloud node
sign-in` gives the node the CLI's current sign-in and changes nothing else;
it refuses when the node belongs to another account (moving a node to
another account is `node init --force`).

## start, stop, status, logs

- `start` runs the node in the background and writes `node.pid` and
  `node.log` under `~/.cyxcloud/node/`. `--foreground` runs it in the
  terminal.
- `stop` stops the process in `node.pid` after checking it is a
  `cyxcloud-node`.
- `status` reads `http://127.0.0.1:<status-port>/status`: name, id, state,
  machine, storage used and offered, shards held, server and last
  heartbeat, listening address, version. `--watch` refreshes every 2
  seconds; `--json` prints the raw report. Exit code 1 when the node is not
  running.
- `logs [-n 50] [-f]` prints the log without colour codes and without any
  line that mentions a password.

## Start with the machine

`cyxcloud node service install` (and `uninstall`) needs no administrator
rights:

| OS | What is installed | Starts | Restarts on failure |
| --- | --- | --- | --- |
| Linux | systemd user unit `~/.config/systemd/user/cyxcloud-node.service` | At boot when lingering is on (`loginctl enable-linger`, tried for you), else at login | Yes, after 10 s |
| macOS | launchd agent `~/Library/LaunchAgents/ai.cyxwiz.cyxcloud-node.plist` | At login | Yes |
| Windows | Run key value `CyxCloud Node` under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` | At logon | No (run `cyxcloud node start` again) |

With a systemd unit or launchd agent, `node start` and `stop` go through
it, and `node status` says how the node starts. On Windows, install prints
the `netsh` line that opens the node's port to other machines, to run as
administrator.

## Not there yet

- `cyxcloud node upgrade`.
- Reputation, audits and earnings in `node status`.
