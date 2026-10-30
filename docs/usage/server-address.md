# The server address

The Engine and the server node reach the CyxWiz server side (central server,
CyxCloud gateway, sign-in) through one address (TOFIX168), with the same
rules as the `cyxcloud` CLI:

| You give | Means |
| --- | --- |
| `https://cloud.cyxwiz.ai` | a front door with TLS: everything at that one origin (gRPC on 443, sign-in at `/api`) |
| `http://192.168.1.216:8000` | a front door without TLS (local, LAN; `deploy/local` runs one on 8000) |
| `192.168.1.216` | no front door: central `:50051`, the gateway `:50060`, sign-in `http://…:3002/api` |

`host:port` without a scheme is refused (it could mean either).

## Where it is set

- Engine: `servers.cyxwiz` in `engine_config.json` (`%APPDATA%/CyxWiz/` on
  Windows, `~/.cyxwiz/` elsewhere). When it is set it decides central, the
  gateway and sign-in; `servers.central`, `servers.cloud_gateway` and
  `auth.api_url` apply only when it is empty (development).
- Server node: `network.cyxwiz_server` in `server_config.yaml`, or
  `cyxwiz-server-daemon --server=ADDR`. When set it decides `central_server`
  and `auth_api_url` (`--central-server` still overrides central alone).

## Setting it on screen (TOFIX168 step 3b)

**Engine: Preferences > Servers.**
- **Server address**: type it, or click an example (This PC
  `http://localhost:8000`, Another PC, No front door, CyxWiz). The line under
  the field says what kind of address it is, or why it can't be used.
- **Use this server** saves it. When you are signed in it asks first:
  changing the server signs you out, because a sign-in belongs to the server
  that issued it. The Engine then drops its connections to the old server;
  central reconnects in the background and CyxCloud when a panel needs it.
  Applying an empty address goes back to the separate addresses.
- **What the Engine connects to**: central, the CyxCloud gateway and sign-in
  with their addresses and whether each answers. A front door is asked once
  at its `/health` (it reports every service); without one, central and the
  gateway are asked over gRPC and the API at its `/health`. **Check again**
  asks again.
- **Account**: who is signed in, and where.
- **Separate addresses (development)**: central, gateway and sign-in fields,
  used only while the server address is empty. They save when you leave a
  field.

Server Connection, Deploy Model (step 1) and the Account popup show the
server in use with **Change in Preferences**; they no longer have address
fields of their own.

**Server node: Settings > Network** has the same address field, examples and
service rows (central and sign-in: a node gets the gateway address in each
job), the separate addresses under a disclosure, and this node's own API
(HTTP API port, and "Serve this node's API over TLS", which was "Enable TLS"
and is about the node's API, not the connection to CyxWiz). **Save changes**
writes the config file the node loaded and says which; saving a new server
signs the node out. The daemon uses the new server after it restarts.

**Server node sign-in card: Change** opens the "CyxWiz server" dialog with
the same field, examples and rows; **Use this server** saves
`network.cyxwiz_server`.

The words and rows come from `cyxwiz-protocol/common/server_address_view`
(tested by `test_server_address_view`); both apps render that one model.

## TLS

An `https://` address is dialled over TLS for every service (gRPC with SSL
credentials, sign-in over HTTPS), checked against the system's trusted
certificates. The old `connection.use_tls` switch is gone: the scheme
decides. The gateway address the Engine puts into a training job keeps its
scheme, so the node dials it the same way.

## Fixed along the way

- The Engine signed in at the built-in `http://127.0.0.1:3002/api` whatever
  `auth.api_url` said; it now uses the configured address.
- The server node daemon renewed its sign-in at the built-in address whatever
  the yaml said; it now uses the configured one.
- The server node's profile and node-id calls used plain HTTP for an
  `https://` sign-in service (and the node-id call turned certificate checks
  off); both now use HTTPS with checks.
