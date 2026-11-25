---
title: Harden a server node
description: What a CyxWiz server node exposes on the network, which services check who is calling, the limits today, and the steps that protect a node now.
type: explanation
area: security
audience: operator
status: stable
since: 1.0.7
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - index.md
  - report-a-vulnerability.md
  - ../about/privacy.md
redirect_from: []
---

A server node runs training jobs that other people send to it. This page
lists what the node daemon (`cyxwiz-server-daemon`) opens on the network,
what each service checks, and what you must do to keep the machine safe
today. Read it before you lend a machine outside your own network.

## What the daemon listens on

Defaults, from the daemon's built-in configuration. A config file or the
command line can change each address.

| Port | Service | Bound to | Checks the caller |
| --- | --- | --- | --- |
| 50052 | P2P job service: the Engine connects here to send a job | `0.0.0.0` | Yes: a reservation token signed with the P2P secret |
| 50053 | Terminal service | `0.0.0.0` | **No** |
| 50054 | IPC for the node's own GUI | `localhost` | No (local only) |
| 50055 | Node service: job assignment | `0.0.0.0` | **No** |
| 50056 | Deployment service | `0.0.0.0` | **No** |
| 50057 | Inference service (gRPC) | `0.0.0.0` | **No** |
| 8082 | Inference REST API | `0.0.0.0` | **No** |

The node also connects out: to the central server (registration,
heartbeats, job results), to the sign-in service (token renewal) and, for
jobs that use cloud datasets, to the CyxCloud gateway.

## What is protected

**Engine connections (port 50052).** Every connection must carry a token the
central server issued for one reservation, naming this node and the end of
the reservation. The node checks its signature with the **P2P secret**, its
expiry, the job and the node id, and refuses anything else ("Invalid or
expired auth token"). With no validator configured, the node refuses every
token. Commands sent later on the connection, such as a reservation
extension, are checked the same way.

**The P2P secret.** The daemon does not start without it. It reads the
secret from a file (`--p2p-secret-file=` or `network.p2p_secret_file`), never
from the config itself, and outside Windows refuses a file that other users
can read. Passing the secret on the command line is refused, because every
user of the machine could read it in the process list.

**Sign-in.** The node's GUI signs in with your CyxWiz account. The password
is never saved; the session token is kept in the node's session file
(`%APPDATA%\CyxWiz\server-node\auth.json` on Windows) and renewed in the
background.

**One daemon per machine.** The daemon takes its ports exclusively and stops
if another daemon holds them.

## Limits today

State these to anyone who sends jobs to your node:

- **The terminal service (50053) opens a shell for any client that reaches
  it**, with the daemon's user rights, without a sign-in check (on Linux and
  macOS; on Windows the shell is not implemented yet).
- **The node service (50055), deployment service (50056) and inference
  services (50057, 8082) do not check who is calling.** A client that can
  reach the node service can submit a job; the node accepts an assignment
  even when the node id in it does not match.
- **The REST API's API keys are not enforced.** The setting
  `api.require_authentication` and the **Require Authentication** box in the
  node's settings are saved but not applied, and the API answers requests
  from any web origin (`Access-Control-Allow-Origin: *`).
- **TLS covers the node service only.** `--tls` (or `network.enable_tls`)
  encrypts port 50055. The P2P, terminal, deployment and inference services
  always run without TLS, so jobs, data and tokens to and from the node travel
  unencrypted. If the certificate cannot be loaded, the daemon falls back to
  no TLS and logs "Running gRPC servers WITHOUT TLS encryption".
- **No sandbox.** Jobs run inside the daemon's process with the daemon's
  user rights. Jobs are graph descriptions and data, not code: a node cannot
  run C++ plugin or Python nodes from a job ("the Server Node cannot run
  extension nodes yet").
- **The audit log records TLS events only**, and only when an audit file is
  set.

## Harden the node now

1. **Run the daemon as its own, unprivileged user.** Never as an
   administrator or root. Give that user only the node's folders.
2. **Firewall everything but the P2P port.** Allow inbound 50052 only from the
   networks your Engine users connect from. Block 50053, 50055, 50056, 50057
   and 8082 from outside the machine, unless you serve models to a network
   you control, and then only to that network.
3. **Keep the IPC port local.** Leave `--ipc-address` at `localhost:50054`.
4. **Protect the secret file.** Readable by the daemon's user only
   (`chmod 600` on Linux and macOS; on Windows, restrict the file's
   permissions to that user). Rotate it with the central server's operator
   if it may have leaked.
5. **Optional: turn on TLS** for the node service with `--tls`,
   `--tls-cert=` and `--tls-key=` (and `--tls-ca=` for mutual TLS). Check the
   log for "TLS initialized successfully".
6. **Check the machine** with `cyxwiz-server-daemon --doctor` before you lend
   it, and keep the node on the newest release (see
   [Support and versions](../about/support-and-versioning.md)).
7. **Watch the daemon's log** for refused tokens ("Invalid or expired auth
   token", "SendJob rejected") and for job assignments you did not expect.

The deployment and terminal services start with every daemon; the
`deployment.enabled` setting does not stop them. Block their ports instead
(step 2).

## Related

- [How CyxWiz handles security](index.md)
- [What CyxWiz sends over the network](../about/privacy.md)
- [Report a vulnerability](report-a-vulnerability.md)
