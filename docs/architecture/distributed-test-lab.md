# Distributed training: the test lab

The machines the distributed-training work (TOFIX163) is built and tested on,
and the role each plays. Owner's setup, 2026-10-26. This page maps roles onto
the plan; it does not change the plan.

## Machines and roles

| Machine | OS | LAN address | Role | Reached by |
| --- | --- | --- | --- | --- |
| Dev PC | Windows 11 | 192.168.1.78 (was .116 until 2026-10-22) | Development and testing; builds everything; runs the Engine | local |
| Dell | Windows | 192.168.1.222 | Server node (training) | ssh `dell-pc` (set up) |
| Mac | macOS 15.7 (Intel) | 192.168.1.159 | Server node (training) | ssh `mac-intel` (set up) |
| HP (Fedora) | Fedora Linux | 192.168.1.216 | Central server (the server side), plus a CyxCloud storage node with 20 GB | ssh not set up yet |
| Lenovo | Windows | 192.168.1.246 | CyxCloud storage node and server node (training) | ssh not set up yet |
| HP (Arch) | Arch Linux | 192.168.1.180 | CyxCloud storage node (1 TB disk, 700 GB offered) and server node (training) | ssh `hp-linux` (set up) |

Hardware facts known so far: the Dev PC has an Intel Core i7-8750H (6 cores,
12 threads), 48 GB RAM and a GeForce GTX 1050 Ti (4 GB); the Dell has Intel
Iris Xe graphics (oneAPI / OpenCL); the Mac has Intel Iris Pro graphics. The
HP (Arch) has an Intel Core i7-2620M (2 cores, 4 threads), 7.7 GB RAM and
no discrete GPU (trains on the CPU through ArrayFire's CPU backend). The HP
(Fedora) and the Lenovo are recorded when they are first reached.

## What runs where

```
                     Dev PC (Windows) — CyxWiz Engine, builds, tests
                              │  one address: the HP (Fedora)
                              ▼
   HP (Fedora) — server side: CyxWiz API, central server, CyxCloud gateway
                 (+ the front door, TOFIX168), Postgres / Mongo / Redis,
                 and a 20 GB CyxCloud storage node
        │ reservations, jobs                 │ shards
        ▼                                    ▼
  server nodes (training)            CyxCloud storage nodes
  Dell · Mac · Lenovo · HP (Arch)    HP (Arch, 700 GB) · HP (Fedora, 20 GB) · Lenovo
```

- **Engine to server side.** The Engine and the `cyxcloud` CLI point at the
  HP (Fedora) once: `cyxcloud login --server 192.168.1.216`, and the Engine's
  server address. Until the front door exists (TOFIX168), that host serves the
  three services on their own ports (gateway HTTP 8180, gateway gRPC 50060,
  sign-in 3002, central 50051).
- **Training.** The Engine reserves a server node through the central server
  and sends the job to it peer to peer (Dell, Mac, Lenovo; several at once is
  TOFIX166; candidates Dell, Mac, Lenovo, HP Arch). Datasets stream to the
  node from CyxCloud (TOFIX164).
- **Storage.** CyxCloud spreads each chunk's 14 shards over the storage nodes
  (HP Arch, HP Fedora, Lenovo). With three nodes each holds 4 or 5 shards of
  a chunk, so losing one can exceed the 4 parity shards; four or more nodes
  are needed before any node loss is survivable (TOFIX176). The Dev PC's
  storage node makes the fourth while testing.
- **Dev PC.** Keeps running the whole stack locally for development
  (`deploy/local`), and its own storage node, until the HP (Fedora) takes the
  server side over.

## Rules for the lab

- **Fixed addresses.** Give every lab machine a DHCP reservation on the router.
  The Dev PC moved from .116 to .78 between 2026-10-22 and 2026-10-26, and
  configs that point at an address break silently when it moves.
- **Access.** ssh with keys only; each machine gets its own key and alias.
  Passwords are used once to install a key and are not kept.
- **Secrets** stay on each machine, owner-only (node keys, access secrets,
  sign-ins): never in the repo, never in logs.
- **Firewall.** Open only what a role needs: server nodes their P2P port
  (50052), storage nodes their shard port (50051 by default), the server side
  its service ports (one 443 once the front door exists).

## Status (2026-10-26)

- Dell and Mac: ssh keys and aliases exist from earlier work; neither answered
  on 2026-10-26 (switched off?).
- HP (Fedora) and Lenovo: new; ssh to be set up, then a machine inventory
  (CPU, GPU, RAM, disk).
- HP (Arch): storage node built and registered on 2026-10-22 (700 GB offer,
  node key since P2); confirmed in the lab by the owner on 2026-10-26 as a
  storage node and a server node. It did not answer on 2026-10-26 (off?);
  its storage node config still points at the Dev PC's old address (.116)
  and moves to the HP (Fedora) once the server side runs there. The server
  node (cyxwiz-server-node) needs a Linux build first (TOFIX163 D1).
