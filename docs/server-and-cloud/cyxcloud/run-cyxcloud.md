---
title: Run CyxCloud
description: Build and run your own CyxCloud gateway and storage nodes, with Postgres and Redis or in Docker, and check them.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - architecture.md
  - storage-node.md
  - front-door.md
  - ../ports.md
---

Use this to run a private CyxCloud: one gateway, its database and cache,
and storage nodes. To lend disk space to an existing CyxCloud, see
[Run a storage node](storage-node.md) instead.

| Part | Binary | Default ports |
| --- | --- | --- |
| Gateway | `cyxcloud-gateway` | HTTP `0.0.0.0:8180`, gRPC `0.0.0.0:50052` |
| Storage node | `cyxcloud-node` | gRPC 50051, metrics 9090 |
| CLI | `cyxcloud` | none |
| Rebalancer | inside the gateway | none |

On a machine that also runs the CyxWiz central server (50051) or a server
node (50052), give the gateway another gRPC port: the CyxWiz addresses
expect it on 50060 (`--grpc-addr 0.0.0.0:50060`). Give storage nodes on the
same machine their own port (`--port 50061`).

## Before you start

- A clone of the CyxCloud repository (`CYXWIZ-Lab/cyxcloud`), or its
  release binaries.
- To build: Rust (the version in `rust-toolchain.toml`) and `protoc` on
  `PATH` or in `PROTOC`.
- Postgres and Redis (containers are easiest). Without Postgres the gateway
  can run with `--memory-only`; its data is lost when it stops, and memory
  objects cannot become datasets.
- The CyxWiz sign-in API's JWT secret: the gateway verifies users' tokens
  with it.

## Build

```bash
cargo build --release -p cyxcloud-gateway -p cyxcloud-node -p cyxcloud-cli
cargo test -p cyxcloud-gateway
```

**CPUs without AVX2.** On Linux and macOS x86-64 the erasure-coding crate
compiles its SIMD code for Haswell, so a binary built on a recent machine
stops with "illegal instruction" on an older CPU. Build for any x86-64 CPU
from about 2009 on with:

```bash
cargo clean --release -p reed-solomon-erasure
RUST_REED_SOLOMON_ERASURE_ARCH=x86-64-v2 cargo build --release -p cyxcloud-gateway
```

The `cargo clean` step is needed because the crate's build script does not
rerun when the variable changes. Windows builds are not affected.

## Run the gateway

```powershell
$env:DATABASE_URL = 'postgres://cyxcloud:<password>@localhost:5433/cyxcloud'
$env:REDIS_URL    = 'redis://localhost:6380'
$env:JWT_SECRET   = '<the CyxWiz sign-in API secret>'
cargo run --release -p cyxcloud-gateway -- --http-addr 0.0.0.0:8180 --grpc-addr 0.0.0.0:50060
```

It applies its database migrations at start, and logs the database URL with
the password hidden. Check it: `GET http://localhost:8180/health` answers
200.

| Option / variable | Meaning |
| --- | --- |
| `--http-addr`, `--grpc-addr` | Listen addresses |
| `DATABASE_URL` / `--database-url` | Postgres; leave out, or `--memory-only`, for in-memory |
| `REDIS_URL` / `--redis-url` | The Redis cache |
| `JWT_SECRET` | Verifies users' tokens. Required when `CYXCLOUD_ENV=production`; otherwise a random one is made and no token verifies. |
| `ADMIN_JWT_SECRET` | Verifies administrator tokens on `/api/v1/admin`; unset or under 32 bytes, those routes answer 503 |
| `--grpc-auth` | Require a token on gRPC (on by default) |
| `--cors-permissive` | Allow every web origin (development only) |
| `CORS_ALLOWED_ORIGINS` | The allowed origins otherwise |
| `TLS_CERT`, `TLS_KEY`, `TLS_CA_CERT`, `TLS_REQUIRE_CLIENT_CERT` | TLS on the gateway's own ports |
| `TRUSTED_PROXY_HOPS` | How many of your proxies sit in front (1 behind the front door) |

## Run a storage node

The simplest way is the CLI: `cyxcloud login`, `cyxcloud node init`,
`cyxcloud node start` (see [Run a storage node](storage-node.md)). To start
one by hand instead:

```powershell
$env:CENTRAL_SERVER_ADDR = 'http://localhost:50060'   # the gateway's gRPC address
$env:CYXWIZ_API_URL      = 'http://localhost:3002'    # the CyxWiz sign-in API
$env:CYXWIZ_EMAIL        = '<account email>'
$env:CYXWIZ_PASSWORD     = '<account password>'
cargo run --release -p cyxcloud-node -- --data-dir ./node1-data --port 50061 --metrics-port 9091
```

Other settings: `NODE_ID`, `NODE_NAME`, `NODE_REGION`, `PUBLIC_ADDRESS`
(what the gateway dials back), `STORAGE_CAPACITY_GB`, `--config` (default
`config.toml`), `--logout` (removes the saved credentials). Without the
email and password the node asks for them on its console. A sign-in that
does not reach the API is retried 6 times, 5 seconds apart; a refused
credential stops the node.

Storage nodes on other machines point `CENTRAL_SERVER_ADDR` at the gateway
machine's address.

## Everything in Docker

`docker-compose.yml` runs Postgres, Redis, the gateway, storage nodes and
the CyxWiz web API. Secrets come from an untracked `.env`:

```powershell
Copy-Item .env.example .env    # then fill in the values
docker compose up -d --build
```

`.env.example` lists the names (`JWT_SECRET`, `POSTGRES_PASSWORD`, the
storage nodes' account password and the web API's database). Never commit
`.env`. `Dockerfile.node` builds a storage node image on its own; pass the
settings above as `-e` variables and mount the shard folder at `/data`.

## Next

- [The front door](front-door.md): one address, with TLS, for clients.
- [TLS](tls.md)
