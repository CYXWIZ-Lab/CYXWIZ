---
title: Add a GPU pack and verify it
description: Install the CUDA, OpenCL or oneAPI pack on Windows, then verify its routes in Preferences > Devices.
type: how-to
area: install
audience: user
status: stable
since: 1.0.7
owner: install
last_reviewed: 2026-11-24
related:
  - windows.md
  - update-and-uninstall.md
  - ../concepts/gpu-providers.md
redirect_from: []
---

Use this when: you want the Engine to train on your GPU.

The Engine always computes on the CPU through ArrayFire's CPU runtime. A
**GPU pack** adds one of ArrayFire's GPU runtimes beside the Engine, in the
install folder. Nothing is installed system-wide, and no driver is
installed: you bring the driver.

## Which pack

GPU packs are published for **Windows x64 only**. Linux and macOS installs
are CPU only until their packs can be verified on real hardware.

| Pack (installer name) | GPUs | Approximate download | Needs |
| --- | --- | --- | --- |
| **OpenCL** | NVIDIA, Intel (AMD untested) | 7 MB | The GPU vendor's OpenCL driver |
| **Intel oneAPI** | Intel GPUs and CPUs | 280 MB | A current Intel driver |
| **NVIDIA CUDA** | NVIDIA | 1.7 GB | A current NVIDIA display driver |

A pack is published only after its routes pass on real hardware. The Windows
packs are verified on NVIDIA GeForce (CUDA, OpenCL) and Intel UHD 630 and
Iris Xe (OpenCL, oneAPI). On Intel GPUs the Engine recommends OpenCL;
oneAPI is known to fail on Intel UHD 630.

## Before you start

- CyxWiz is installed on Windows. See [Install on Windows](windows.md).
- Install a current driver for your GPU from NVIDIA or Intel.
- Close training runs: verification and pack changes wait for them.

## Install a pack

1. Open **CyxWiz Installer** from the Start Menu. In the Engine you can also
   use **File > Preferences... > Devices**, then **Open CyxWiz Installer...**
   under **Installed packs**.
2. Select the **Individual components** tab. Each component shows its status
   on the right, its version, size and requirements, and **Recommended** when
   it suits this computer.
3. Tick the pack you want. The profile changes to **Custom**.

   A pack you cannot tick says why, for example **Not published** ("No signed
   ... package is published for this platform and release") or
   **Not compatible**. The CUDA card also shows whether an NVIDIA driver was
   found, for example "No NVIDIA display driver was detected on this
   system."
4. In the **Installation summary** column, select **Review & install**.
5. In **Review installation**, check **Acquire and locally qualify:** and the
   **Package data** size, then select **Apply changes**.
6. Wait for **Installation complete**. The installer verifies the pack's
   signature and qualifies each route on your hardware before it turns the
   pack on. A pack that fails stays inactive and says why.

Tip: the **Workloads** tab's **Recommended** profile ticks the recommended
packs for you. Its summary also lists **Optional for this computer** packs
you can add under **Individual components**.

## Verify the GPU in the Engine

Training uses only routes that passed verification on your machine.

1. Start the Engine and open **File > Preferences...** (Ctrl+,).
2. Select **Devices**. Each physical device is a card; each card lists its
   compute routes (for example CUDA, OpenCL, CPU) with a status such as
   **Not verified yet**, **Verified** or **Failed**.
3. Select **Verify all**. Each route runs a short qualification in a separate
   process on your hardware and records a measured speed. A full machine
   typically takes about two minutes. To stop, select
   **Cancel verification**.
4. Read **Latest verification result**. It names the fastest verified route
   for a fixed dense benchmark; select **Use fastest route** to take it.
   Actual model speed can differ.
5. Optional: to choose another route, select the circle in front of it, then
   **Use this device**. The Engine checks the route, then uses it.

To check one route again without touching the others, select
**Verify again** on its row. **Details** shows what was measured.

## Pack statuses

| Status | Meaning |
| --- | --- |
| **Verified after install** | Not installed yet; checked on your computer after install |
| **Compatible; verify** / **Installed; verify** | Requirements match; the route is not verified yet |
| **Verified and ready** / **Verified and measured** | The route passed every required operation |
| **Best verified** | The fastest comparable route in the latest verification |
| **Verification failed**, **timed out** or **crashed** | The route did not pass; update the driver and verify again, or use another route |
| **Verification outdated** | The saved results belong to another runtime; run **Verify all** again |
| **Update available** | A newer signed pack exists; see [Update or uninstall](update-and-uninstall.md) |
| **Not compatible** / **Unavailable** | This machine or release cannot use the pack |

## If something goes wrong

**A route shows "Failed" or "Needs driver update"**: install the newest
driver from your GPU vendor, restart, then select **Verify again** on the
route.

**"Verification did not complete"**: the saved results were not changed.
Fix the cause in the message and verify again.

**The Engine trains on the CPU although a pack is installed**: the GPU route
is not verified, or another route is selected. Verify, then pick the route
with **Use this device**.

Which layers run on CyxWiz's own GPU kernels instead of ArrayFire is
explained in [What runs where on a GPU](../concepts/gpu-providers.md). Known GPU
runtime problems are listed in
[KNOWN_ISSUES.md](https://github.com/CYXWIZ-Lab/CYXWIZ/blob/HEAD/KNOWN_ISSUES.md).

## Related

- [Update or uninstall](update-and-uninstall.md) (remove one pack)
- [What runs where on a GPU](../concepts/gpu-providers.md)
