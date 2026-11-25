---
title: TLS for CyxCloud and the front door
description: How traffic between CyxWiz machines is encrypted, the front door's certificate, private certificate authorities, and every client's CA setting.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - front-door.md
  - ../server-address.md
  - ../security.md
  - cli.md
---

Use this to serve the CyxWiz server side over TLS and to make every client
trust it, with a public certificate or a private certificate authority (CA).

## Which hops are encrypted

| Hop | Who proves what | TLS today |
| --- | --- | --- |
| Clients to the front door (Engine, `cyxcloud` CLI, storage and server nodes, browsers) | The server proves its name with the front door's certificate | **Yes**, when the door has a certificate |
| Front door to central, the gateway and the sign-in API, on one host | Nobody: one host, a private network | Plain, by design; the services are not published |
| Gateway and rebalancer to storage nodes | The gateway presents the node's access secret | Plain |
| The Engine and the central server to server nodes | The Engine presents a signed reservation token | Plain |

Plain text is meant only for this machine or a private network: every
client refuses `http://` to a public address.

## Give the front door a certificate

The door ([front-door.md](front-door.md)) speaks TLS on `listen` when
`front.toml` names a PEM certificate chain and its key:

```toml
listen = "0.0.0.0:443"
tls_cert = "/etc/cyxwiz/certs/fullchain.pem"
tls_key  = "/etc/cyxwiz/certs/privkey.pem"
internal_listen = "0.0.0.0:8000"
```

- The TLS port answers TLS only; a plain request there gets no answer.
- `internal_listen` is a plain listener for the server's own services. It
  is refused without TLS. **Publish only `listen`**: anything that reaches
  the internal port skips TLS.
- The services learn which listener a request came through from
  `X-Forwarded-Proto`; the audit line says `tls=true` or `false`.
- Keep the key file mode 600 and out of every repository and shared backup.

Check from another machine:

```bash
curl --cacert ca.crt https://server.lan/health
curl http://server.lan:443/health
```

The first answers `{"front":"ok",...}`; the second gets no HTTP answer. On
Windows, add `--ssl-no-revoke` to the first: curl there uses the system TLS,
which wants revocation information a private CA does not publish.

## Which certificate

- **The server has a public domain name**: use a public certificate, from
  an ACME client such as certbot (Let's Encrypt), or a Cloudflare origin
  certificate behind Cloudflare. Clients need nothing: the system's roots
  trust it. Restart the door after each renewal.
- **A LAN or test server without a domain**: use a private CA. It signs the
  door's certificate, and every client is given the CA's public
  certificate, `ca.crt`.

### Rules for a private CA

- Make the CA once per server and keep its private key (`ca.key`) in one
  owner-only folder on the server. Anyone with it can make certificates
  every client trusts; never copy it into a repository, a container or a
  backup others can read.
- The door's certificate must list every name and address clients use to
  reach the server (host names as DNS names, addresses as IP names). A
  client that connects by a name not listed refuses the server.
- Renew the door's certificate before it expires and restart the door.
  Clients need nothing new: they trust the CA, not the certificate.
- Before trusting a `ca.crt` that came by email or chat, compare its
  fingerprint with the one on the server:

  ```bash
  openssl x509 -in ca.crt -noout -fingerprint -sha256
  ```

For development only, the CyxCloud repository's
`scripts/generate-dev-certs.sh` makes a CA and certificates for a gateway,
three nodes and the CLI. Never use those for anything others reach.

## Give clients the CA

A client given a CA trusts **only** that CA for the server; without one it
uses the system's trusted roots. No client can skip certificate checks.

| Client | Setting |
| --- | --- |
| `cyxcloud` CLI | `cyxcloud login --server https://HOST --ca-cert ca.crt` (saved as `[server] ca_cert`; `config set server.ca_cert PATH`, empty clears; `CYXCLOUD_CA_CERT` for one run). `cyxcloud doctor` shows which CA is in use. |
| Storage node | `[network] tls_ca_cert` in its config (written by `cyxcloud node init` from the CLI's CA), or `TLS_CA_CERT`. Used for the gateway, the dataset service and sign-in. |
| Engine | **Preferences > Servers > Certificate authority (optional)** (saved as `servers.ca_file`; `CYXWIZ_CA_FILE` when empty). It shows the CA's SHA-256 to compare. |
| Server node | **Settings > Network**, the same row (saved as `network.server_ca_file`). |
| A browser on a test machine | Import `ca.crt` as a trusted authority (Windows: `certutil -addstore -f Root ca.crt` as administrator; macOS: Keychain Access, System, Always Trust; Linux: copy to `/usr/local/share/ca-certificates/` and run `update-ca-certificates`). Only on machines you control; remove it when the server is gone. |

## Switch a running server to TLS

1. Back up the server's deploy folder, with `front.toml` and its
   environment file.
2. Make the CA (once) and the door's certificate with every name clients
   dial.
3. In `front.toml`, replace `listen = "0.0.0.0:8000"` with the four TLS
   lines above, and publish port 443 instead of 8000.
4. Restart the door. Check from another machine: `https://HOST/health`
   answers with every service `up` when given the CA and fails without it;
   `http://HOST:8000/health` cannot connect; `http://HOST:443/health` gets
   no HTTP answer.
5. Give every client the CA and the new address (`https://HOST` instead of
   `http://HOST:8000`) and check each: `cyxcloud doctor`, a dataset `put`
   and `get`, the Engine's **Preferences > Servers**, a server node's
   **Settings > Network**, the website. The door's audit lines show
   `tls=true` for every outside client.

To roll back, restore `front.toml` from the backup, publish 8000 again and
restart the door; clients go back to `http://HOST:8000`.

## The gateway's own TLS

The gateway can also serve TLS itself, without a front door: `TLS_CERT`,
`TLS_KEY`, `TLS_CA_CERT`, and `TLS_REQUIRE_CLIENT_CERT` to ask clients for
a certificate (`TLS_CLIENT_CERT`, `TLS_CLIENT_KEY` on the client side).

## If something goes wrong

| Problem | What to do |
| --- | --- |
| Clients fail with "certificate expired" | Issue a new door certificate with the same names and restart the door. |
| A client says the name does not match | It used a name or address that is not in the certificate. Issue one with the missing name and restart the door. |
| The door's private key leaked | Issue a new certificate (with a new key) at once and restart the door. A private CA has no revocation list, so the old certificate stays valid until it expires: replace the CA too if the leak is serious. |
| `ca.key` leaked, or the CA is lost | Replace the CA: move the old one aside, make a new CA and door certificate, restart the door, give every client the new `ca.crt` and remove the old one, then delete the old CA. Until a client has the new CA it refuses the server. |
