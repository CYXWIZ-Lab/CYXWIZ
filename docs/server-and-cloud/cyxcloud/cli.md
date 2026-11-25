---
title: The cyxcloud CLI
description: Every cyxcloud command: server and sign-in, datasets, access tokens, health, exit codes, and the storage node commands.
type: reference
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - storage-node.md
  - datasets.md
  - tls.md
  - ../server-address.md
---

One binary, one server address, one sign-in.

## Server and sign-in

```text
cyxcloud login --server HOST [--email E] [--password-file F] [--ca-cert ca.crt]
cyxcloud register
cyxcloud whoami
cyxcloud logout
cyxcloud config show
cyxcloud config set server HOST
```

- `--server` (or `config set server`) sets every service address at once,
  remembered in `~/.cyxcloud/config.toml` under `[server] host`:
  - `https://HOST[:PORT]`: a [front door](front-door.md) with TLS, every
    service at that one URL;
  - `http://HOST:PORT`: a front door without TLS (local or LAN; a local one
    is `http://HOST:8000`);
  - `HOST` (or `http://HOST` without a port): no front door, the services
    on their own ports (gateway HTTP `:8180`, gateway gRPC `:50060`,
    sign-in `:3002`).

  The separate keys `gateway.http_url`, `gateway.grpc_url` and
  `auth.api_url` can still be set one by one.
- Plain `http://` and a bare `HOST` are allowed only to this machine or a
  private network (the same rule as the Engine:
  [The server address](../server-address.md)). Anything else is refused
  with "use the server's https:// address". Setting a plain LAN server
  prints a one-line note.
- **A server with a private CA**: `--ca-cert ca.crt` saves the CA with the
  server (`[server] ca_cert`, an absolute path); every command then trusts
  only that CA for the server, for gRPC, the gateway and sign-in, and
  `cyxcloud node init` passes it to the storage node. `cyxcloud config set
  server.ca_cert PATH` changes it, an empty value clears it, and changing
  the server clears it too. `--ca-cert` (or `CYXCLOUD_CA_CERT`) on any
  command overrides it for that command. There is no switch that skips
  certificate checks.
- Changing the server drops the old sign-in: a sign-in belongs to the
  server that issued it.
- The password is asked for, or read from the first line of
  `--password-file` (for scripts). It never goes on the command line.
- `~/.cyxcloud/credentials.json` is owner-only on every operating system;
  it expires with the token.
- `register` creates the account through the sign-in API, then signs in.
  `logout` forgets the local sign-in.

## Datasets

```text
cyxcloud dataset put <FILE|FOLDER>... --name NAME [--description D]
cyxcloud dataset ls [--shared] [--public] [--filter TEXT]
cyxcloud dataset info NAME|ID
cyxcloud dataset get NAME|ID [-o DIR]
cyxcloud dataset rm NAME|ID [--force]
cyxcloud dataset verify ID
cyxcloud dataset share ID --with EMAIL
cyxcloud token create NAME|ID [--ttl 1h]
```

- These are the same datasets the Engine shows under Data Input >
  **CyxCloud**: both use the gateway's dataset service.
- `put` uploads each file (every file directly in a folder) in 1 MB pieces
  with its BLAKE3 hash; the gateway checks the bytes against it, then the
  files become one dataset.
- `get` streams each file and checks it against the gateway's SHA-256
  before moving it into place (`*.partial` until then). A file name from the
  server is cut to its last part, so it cannot write outside the folder.
- A dataset is named by its name or its id; the first 8 characters of the id
  are enough when they are unique.
- `rm` asks first in a terminal; scripts pass `--force`.
- `token create` prints only the token on stdout (the rest on stderr), for
  a training node; lifetime 1 minute to 30 days.
- `verify` and `share` use the gateway's REST API; `share` takes an email
  address.

## Health

`cyxcloud doctor` checks the path from this machine to the server and the
storage node on this machine, and says what to do. It checks the version,
the server address, the gateway and sign-in (reachable, TLS or not, clock
skew), the sign-in (confirmed with the server, time left) and a local node
(running, registered, its port accepting connections, offer against free
space).

## Exit codes

Every command uses the same codes, so a script can tell who has to act:

| Code | Meaning |
| --- | --- |
| 0 | Done (also `--help` and `--version`; `doctor` with warnings only) |
| 1 | You: a wrong argument, not signed in, no such dataset, refused by the server; `node status` when the node is not running |
| 2 | Network: the server or the node could not be reached in time |
| 3 | Server: the server failed (an internal error, a lost shard), or the local node stopped right after `node start` |

The message on stderr says what happened and what to do next.

## Storage node

```text
cyxcloud node init | start | stop | status | logs | sign-in
cyxcloud node service install | uninstall
```

See [Run a storage node](storage-node.md).
