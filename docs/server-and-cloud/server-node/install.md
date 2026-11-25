---
title: Build and install a server node
description: Build the server node daemon and GUI from source on Windows, Linux or macOS, and prepare the machine to take jobs.
type: how-to
area: server-and-cloud
audience: operator
status: experimental
since: 1.0.8
owner: server-and-cloud
last_reviewed: 2026-11-25
related:
  - run-a-server-node.md
  - sign-in.md
  - ../ports.md
  - ../../install/gpu-packs.md
---

Use this to get the two server node programs, `cyxwiz-server-daemon` and
`cyxwiz-server-gui`, onto a machine you want to lend. The release packages
contain the Engine only; a server node is built from the source tree.

## Before you start

- A clone of the repository with its build requirements: a C++20 compiler,
  CMake, vcpkg and ArrayFire. See [INSTALL.md](../../../INSTALL.md) for the
  requirements on each operating system.
- The machine's devices verified, on this machine. A node refuses training
  jobs on a compute route that has not passed verification. Install the
  Engine on the same machine and run **Verify all** in
  **Preferences > Devices** (or verify with the installer); the node reads
  the same results file. See [GPU packs](../../install/gpu-packs.md).

## Build

1. In a terminal at the repository root, run the wrapper with
   `--server-node`. It configures the build and builds only the daemon and
   the GUI:

   ```powershell
   .\build.bat --server-node -j 8
   ```

   ```bash
   ./build.sh --server-node -j 8
   ```

2. The programs land in the build tree: `build/windows-release/bin/Release/`
   on Windows, `build/linux-release/bin/` or `build/macos-release/bin/` on
   Linux and macOS.
3. Optional: build and run the node's tests. On Windows:

   ```powershell
   cmake --build build/windows-release --config Release --target test_job_execution_service test_cloud_dataset_fetcher test_sign_in_presentation
   .\build\windows-release\bin\Release\test_job_execution_service.exe
   ```

## Prepare the machine

1. **Choose a data folder.** A remote job writes the dataset files it
   receives, and its resume checkpoints, under `CYXWIZ_NODE_DATA_DIR`, or
   under the system temp folder (`cyxwiz-node`) when that is not set. Set
   the variable to a drive with room for your largest datasets.
2. **Get the P2P secret** from whoever runs the central server you will
   register with, and save it in a file only your user can read (on Linux
   and macOS, `chmod 600`). The daemon does not start without it. See
   [Run a server node](run-a-server-node.md#what-the-node-needs).
3. **Open the P2P port** (50052) in the firewall for the machines whose
   Engines will train on this node. Keep every other node port closed to
   other machines; [Ports](../ports.md) says why for each one.
4. Optional, macOS: sign the programs with the entitlements file to read
   temperatures and power. See [Temperature and power on macOS](macos-sensors.md).

## Check it

From the folder that holds the programs:

```powershell
.\cyxwiz-server-daemon.exe --doctor
```

`--doctor` prints **OK**, **WARN** or **FAIL** per check and exits 0 when
the node can take training jobs:

| Check | FAIL or WARN means |
| --- | --- |
| **Verified routes** | No device route passed verification here: verify the devices first. |
| **Training benchmark** | No measurement yet; the node reports a compute score of 0. Run `--benchmark`. |
| **Compute preference** | The preferred device saved on this machine is not a verified route. |
| **Job data folder** | The folder is not writable, or has little free space. |
| **Central server** | The central server address does not answer. |

Then run `--benchmark` once. It measures training throughput on every
verified route and saves the result; the node reports it when it
registers, so nodes are ranked by measured speed.

## Next

- [Run a server node](run-a-server-node.md)
