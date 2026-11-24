# Distributed training: a test lab

How to set up a small lab of machines to build and test distributed training:
the roles the machines play, what runs where, the ports, and the rules that
keep the lab reliable and safe. Your own machines (names, addresses, remote
logins, hardware) never go in the repository: they live in a local file that
git ignores.

## Your machines: `config/lab.local.json`

Copy [`config/lab.example.json`](../../config/lab.example.json) to
`config/lab.local.json` and list your machines there: a name, the operating
system, its fixed LAN address, its roles, and how you reach it (local, or an
ssh alias). Git ignores every `*.local.json`, so the file stays on your PC.

To make sure no private detail slips into a commit, list your machine names,
addresses and aliases as patterns in `config/private-patterns.local.txt`
(format: [`config/private-patterns.example.txt`](../../config/private-patterns.example.txt))
and run:

```bash
python scripts/check_private_details.py
```

## Roles

| Role | What it runs | Notes |
| --- | --- | --- |
| Development PC | The CyxWiz Engine, builds, tests | Can also run the whole server side locally (`deploy/local`) |
| Server side | The CyxWiz API (sign-in), the central server, the CyxCloud gateway, their databases | One machine the Engine points at |
| Server node (training) | `cyxwiz-server-node`: takes training jobs | Any OS with a build; GPU optional (CPU through ArrayFire) |
| CyxCloud storage node | Stores dataset shards | Offers part of a disk |

One machine may hold several roles, for example a storage node and a server
node.

## What runs where

```
                     Development PC: CyxWiz Engine, builds, tests
                              │  one address: the server side
                              ▼
   Server side: CyxWiz API, central server, CyxCloud gateway,
                databases (Postgres, Mongo, Redis), optionally a storage node
        │ reservations, jobs                 │ shards
        ▼                                    ▼
  server nodes (training)            CyxCloud storage nodes
```

- **Engine to server side.** The Engine and the `cyxcloud` CLI point at the
  server side once: `cyxcloud login --server <server-side address>`, and the
  Engine's server address. Without a front door, that machine serves each
  service on its own port: gateway HTTP 8180, gateway gRPC 50060, sign-in
  3002, central 50051.
- **Training.** The Engine reserves a server node through the central server
  and sends the job to it peer to peer. Datasets stream to the node from
  CyxCloud.
- **Storage.** CyxCloud spreads each chunk's 14 shards over the storage nodes.
  With three nodes each holds 4 or 5 shards of a chunk, so losing one node can
  exceed the 4 parity shards: four or more storage nodes are needed before
  the loss of any one node is survivable.

## Rules for the lab

- **Fixed addresses.** Give every lab machine a DHCP reservation on the
  router. Configs that point at an address break silently when it moves.
- **Access.** ssh with keys only; each machine gets its own key and alias.
  Passwords are used once to install a key and are not kept.
- **Secrets** stay on each machine, readable by its owner only (node keys,
  access secrets, sign-ins): never in the repository, never in logs.
- **Firewall.** Open only what a role needs: server nodes their P2P port
  (50052), storage nodes their shard port (50051 by default), the server side
  its service ports (one port, 443, once a front door exists).
- **Private details stay local.** Machine names, addresses, ssh aliases and
  hardware notes go in `config/lab.local.json`, never in a tracked file.
