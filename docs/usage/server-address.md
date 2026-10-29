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

There is no field for it in Preferences yet (TOFIX168 step 3b).

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
