---
title: Ports
description: Every port the Engine, a server node, the CyxWiz server side and CyxCloud open or dial, with who uses it and whether to open it.
type: reference
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - security.md
  - server-address.md
  - server-node/settings.md
  - cyxcloud/run-cyxcloud.md
---

The defaults below come from the code. "Open to the network" is what a
firewall should let in from other machines; everything else stays closed to
them. Read [Security notes](security.md) for why.

## Server node

Set in `server_config.yaml` (`network.*_address`, `network.http_api_port`)
or by the daemon's options.

| Port | Listens on | Service | Used by | Open to the network |
| --- | --- | --- | --- | --- |
| 50052 | `0.0.0.0` | P2P job service: training jobs and their data | The Engine of the user who reserved the node | **Yes**, to the Engines that train here |
| 50053 | `0.0.0.0` | Terminal sessions | Nothing in this release | **No**: it does not check who calls |
| 50054 | `localhost` | IPC for the GUI | `cyxwiz-server-gui` on this machine | No |
| 50055 | `0.0.0.0` | Node service: job assignments | The central server | Only to the central server |
| 50056 | `0.0.0.0` | Model deployment (gRPC) | **Deploy > Deploy to Server Node...** in the Engine | Only if you serve models; no sign-in is checked |
| 50057 | `0.0.0.0` | Inference (gRPC) | Clients of deployed models | Only if you serve models; no sign-in is checked |
| 8082 | `0.0.0.0` | [HTTP API](server-node/http-api.md) for deployed models | Clients of deployed models | Only if you serve models; no key is checked |

The node dials out to the central server, sign-in and, for CyxCloud
datasets, the gateway named in each job.

## Engine

The Engine dials out only: to the central server, the CyxCloud gateway and
sign-in (from the [server address](server-address.md)), and to a reserved
node's P2P port (50052). It opens a port only while **Deploy > Model
Serving** serves a model: 8080 by default, on every interface.

## The CyxWiz server side

The addresses clients use. With a front door, only the front door's port is
published.

| Port | Service | Used by |
| --- | --- | --- |
| 443 | Front door with TLS (`https://HOST`) | Every client |
| 8000 | Front door without TLS (`http://HOST:8000`, local or LAN) | Every client |
| 50051 | Central server (gRPC), no front door | The Engine, server nodes |
| 50060 | CyxCloud gateway (gRPC), no front door | The Engine, server nodes, the CLI, storage nodes |
| 8180 | CyxCloud gateway (HTTP: datasets REST, S3, `/health`), no front door | The CLI, the website |
| 3002 | Sign-in API, no front door | The Engine, server nodes, the CLI |

The gateway's own built-in gRPC default is 50052 (`--grpc-addr`), which
clashes with a server node on the same machine; the CyxWiz addresses assume
it runs on 50060 (`--grpc-addr 0.0.0.0:50060`).

## CyxCloud storage node

| Port | Listens on | Service | Open to the network |
| --- | --- | --- | --- |
| 50051 | `0.0.0.0` (`--port`, `[network] grpc_port`) | Shard traffic from the gateway and the rebalancer | Yes, to the gateway |
| 9090 | `--status-port`, `[metrics] port` | Status (`cyxcloud node status`) and metrics | No: this machine only |

Two storage nodes on one machine need different ports (for example
`--port 50061 --metrics-port 9091`).
