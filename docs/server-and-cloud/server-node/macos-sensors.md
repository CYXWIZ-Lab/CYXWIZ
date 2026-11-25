---
title: Temperature and power on macOS
description: What a server node reads on macOS, why temperatures and power show 0, and how to sign the programs with their entitlements.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - install.md
  - settings.md
redirect_from:
  - cyxwiz-server-node/MACOS_SMC_SETUP.md
---

On macOS a server node detects the GPUs and reads their memory and use,
the CPU (name, cores, frequency) and the system memory. **CPU and GPU
temperatures and power read 0 on macOS in this build**: the node has code
to read Apple's System Management Controller (SMC), but it does not call it
yet, so signing the programs does not change the readings. Everything else
a node does works without the readings.

The signing step below is still useful: it gives the programs the network
and library entitlements macOS asks for when you hand them to other Macs.

## Before you start

- The daemon and the GUI built in `build/macos-release/bin/` (see
  [Build and install a server node](install.md)).

## Sign with the script

1. In a terminal at the repository root, run:

   ```bash
   ./cyxwiz-server-node/sign_macos.sh
   ```

   It signs `cyxwiz-server-daemon` and, when present, `cyxwiz-server-gui`
   ad hoc (no Apple developer account needed) with
   `cyxwiz-server-node/macos_entitlements.plist`, then shows the
   entitlements of each.

## Sign by hand

```bash
cd build/macos-release/bin
codesign --force --deep --sign - \
  --entitlements ../../../cyxwiz-server-node/macos_entitlements.plist \
  --timestamp=none cyxwiz-server-daemon
codesign -d --entitlements - cyxwiz-server-daemon
```

Do the same for `cyxwiz-server-gui`. To hand the programs to other Macs,
sign with your Developer ID instead of `-`, add `--options runtime
--timestamp`, and notarize them with Apple (`xcrun notarytool`).

## What the entitlements allow

| Entitlement | Why |
| --- | --- |
| `com.apple.security.device.smc` | Access to the SMC (for temperature and power, once the node reads them) |
| `com.apple.security.network.server` | Open the node's gRPC and HTTP services |
| `com.apple.security.network.client` | Connect to the central server, CyxCloud and sign-in |
| `com.apple.security.files.user-selected.read-write` | Read and write the files you choose |
| `com.apple.security.cs.allow-dyld-environment-variables` | Load ArrayFire and other libraries from their paths |
| `com.apple.security.cs.disable-library-validation` | Load libraries not signed by the same team |

## If something goes wrong

**The program crashes at start after signing**: remove the signature
(`codesign --remove-signature cyxwiz-server-daemon`) and sign again with the
entitlements file.

Do not run the daemon with `sudo`: a node that runs as root runs every job
as root, and it does not make the readings appear.
