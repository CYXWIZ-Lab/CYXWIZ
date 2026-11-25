---
title: The server address
description: Point the Engine, a server node and the cyxcloud CLI at the CyxWiz server side with one address, and trust a private certificate authority.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - ports.md
  - cyxcloud/tls.md
  - cyxcloud/cli.md
  - server-node/settings.md
  - ../main-window/preferences.md
redirect_from:
  - server-address.md
  - ../usage/server-connection.md
---

The Engine and the server node reach the whole CyxWiz server side (the
central server, the CyxCloud gateway and sign-in) through one address. The
`cyxcloud` CLI reads it with the same rules.

## What you can give

| You give | Means |
| --- | --- |
| `https://cloud.example.com` | A front door with TLS: everything at that one origin (gRPC on 443, sign-in at `/api`) |
| `http://server.lan:8000` | A front door without TLS (this machine or a private network; a local front door listens on 8000) |
| `server.lan` | No front door: central on `:50051`, the gateway on `:50060`, sign-in at `http://server.lan:3002/api` |

`host:port` without a scheme is refused, because it could mean either. A
path after the host is refused too.

Plain `http://` (and a bare host) is accepted only for this machine or a
private network: loopback, the private IPv4 ranges (10/8, 172.16/12,
192.168/16), link-local, 100.64/10, IPv6 unique-local and link-local, and
LAN names (one label, or `.local`, `.lan`, `.internal`, `.home.arpa`).
Anything else is refused with "use the server's https:// address", because
a sign-in would cross the internet unencrypted.

## Where it is set

- **Engine**: `servers.cyxwiz` in `engine_config.json` (`%APPDATA%\CyxWiz\`
  on Windows, `~/.cyxwiz/` elsewhere). When it is set it decides central,
  the gateway and sign-in; `servers.central`, `servers.cloud_gateway` and
  `auth.api_url` apply only while it is empty (development).
- **Server node**: `network.cyxwiz_server` in `server_config.yaml`, or
  `cyxwiz-server-daemon --server=ADDR`. When set it decides
  `central_server` and `auth_api_url`; `--central-server` still overrides
  central alone.
- **`cyxcloud` CLI**: `cyxcloud login --server ADDR`, saved in
  `~/.cyxcloud/config.toml`. See [The cyxcloud CLI](cyxcloud/cli.md).

## Set it in the Engine

1. Open **Preferences > Servers**.
2. In **Server address**, type the address, or click an example (**This
   PC** `http://localhost:8000`, **Another PC**, **No front door**,
   **CyxWiz**). The line under the field says what kind of address it is,
   or why it cannot be used.
3. Click **Use this server**. When you are signed in it asks first:
   changing the server signs you out, because a sign-in belongs to the
   server that issued it. The Engine drops its connections to the old
   server; central reconnects in the background, and CyxCloud when a panel
   needs it.
4. Read **What the Engine connects to**: central, the CyxCloud gateway and
   sign-in, each with its address and whether it answers. A front door is
   asked once at its `/health`; without one, central and the gateway are
   asked over gRPC and sign-in at its `/health`. **Check again** asks again.

Applying an empty address goes back to the **Separate addresses
(development)** fields: central, gateway and sign-in, saved when you leave
a field. **Account** shows who is signed in, and where.

**Server Connection**, **Deploy Model** and the **Account** popup show the
server in use with **Change in Preferences**; they have no address fields
of their own.

## Set it on a server node

- **Settings > Network** has the same field, examples and rows (central and
  sign-in only: a node gets the gateway address in each job), the separate
  addresses under a disclosure, and this node's own API. **Save changes**
  writes the config file the node loaded and names it; saving a new server
  signs the node out. The daemon uses the new server after it restarts.
- On the sign-in card, **Change** opens the **CyxWiz server** dialog with
  the same field; **Use this server** saves `network.cyxwiz_server`.

See [Server node settings](server-node/settings.md).

## TLS

An `https://` address is dialled over TLS for every service (gRPC with TLS,
sign-in over HTTPS) and checked against the system's trusted certificates,
or against the server's own certificate authority when you set one (below).
Nothing turns certificate checks off. The gateway address the Engine puts
into a training job keeps its scheme, so the node dials it the same way.

### Trust a private certificate authority

A test or LAN server whose certificate comes from a private CA (see
[TLS for CyxCloud and the front door](cyxcloud/tls.md)) is trusted by giving
the client that CA's `ca.crt`. The client then trusts **only** that CA for
the server.

- **Engine**: **Preferences > Servers > Certificate authority (optional)**,
  shown under an `https://` address. Type the path or click **Browse...**;
  it applies when you leave the field (open connections are made again;
  you stay signed in). **Clear** goes back to the system's roots, or to the
  `CYXWIZ_CA_FILE` environment variable when that is set. Under the field:
  what it trusts, and the CA's SHA-256 fingerprint (hover for the full one)
  to compare with the one the server's operator gives you. Saved as
  `servers.ca_file` in `engine_config.json`.
- **Server node**: **Settings > Network**, the same row, saved with **Save
  changes** as `network.server_ca_file` in `server_config.yaml`.
- Leave it empty for a server with a public certificate.

A file that is not a certificate is named under the field, and connections
to the `https://` server fail until it is fixed; the Engine and node logs
say so at start.

## If a check says Not reachable

The Engine log has the reason at Debug level (`GET ... failed: ...`, with
the certificate problem when there is one). Check the address form above,
the [ports](ports.md) the server side opens, and the CA.
