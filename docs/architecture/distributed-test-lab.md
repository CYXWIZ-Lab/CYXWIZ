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

Hardware facts known so far: the Dev PC has an Intel Core i7-8750H (6 cores,
12 threads), 48 GB RAM and a GeForce GTX 1050 Ti (4 GB); the Dell has Intel
Iris Xe graphics (oneAPI / OpenCL); the Mac has Intel Iris Pro graphics. The
HP (Fedora) and the Lenovo are recorded when they are first reached.

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
  Dell · Mac · Lenovo                HP (Fedora, 20 GB) · Lenovo
```

- **Engine to server side.** The Engine and the `cyxcloud` CLI point at the
  HP (Fedora) once: `cyxcloud login --server 192.168.1.216`, and the Engine's
  server address. Until the front door exists (TOFIX168), that host serves the
  three services on their own ports (gateway HTTP 8180, gateway gRPC 50060,
  sign-in 3002, central 50051).
- **Training.** The Engine reserves a server node through the central server
  and sends the job to it peer to peer (Dell, Mac, Lenovo; several at once is
  TOFIX166). Datasets stream to the node from CyxCloud (TOFIX164).
- **Storage.** CyxCloud spreads each chunk's 14 shards over the storage nodes
  (HP Fedora and Lenovo). With two storage nodes a lost node makes files
  unreadable (7 lost shards, 4 parity); four or more are needed before a node
  loss is survivable (TOFIX176).
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
- The earlier HP on Arch Linux (192.168.1.180, 1 TB disk, storage node set up
  2026-10-22) did not answer on 2026-10-26 either; its place in the lab is
  being confirmed with the owner.
