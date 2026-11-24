---
title: Install on macOS
description: Run the CyxWiz setup on an Intel or Apple Silicon Mac, allow the unnotarized app, and start the Engine.
type: how-to
area: install
audience: user
status: stable
since: 1.0.7
owner: install
last_reviewed: 2026-11-24
related:
  - index.md
  - first-launch.md
  - update-and-uninstall.md
redirect_from: []
---

Use this when: you want the CyxWiz Engine on a Mac.

## Before you start

- macOS on Intel (x64) or Apple Silicon (arm64). Releases are built on
  macOS 15; older versions are untested.
- macOS installs are **CPU only**. No GPU pack is published for macOS yet,
  so the installer shows CUDA, OpenCL and oneAPI as **Not published**.
- The binaries are not notarized by Apple. macOS blocks them until you allow
  them (step 3).
- Optional, for scripts and the Python console: Python 3.12, from
  python.org or `brew install python@3.12`.

## Steps

1. On the [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases),
   download the setup for your Mac:
   - Intel: `cyxwiz-setup-macos-x64.tar.gz`
   - Apple Silicon: `cyxwiz-setup-macos-arm64.tar.gz`
2. In Terminal, extract it (replace `<platform>` with `macos-x64` or
   `macos-arm64`):

   ```bash
   mkdir cyxwiz-setup
   tar -xzf cyxwiz-setup-<platform>.tar.gz -C cyxwiz-setup
   ```

3. Allow the unnotarized setup. Either remove the quarantine flag:

   ```bash
   xattr -dr com.apple.quarantine cyxwiz-setup
   ```

   or run it once, then open **System Settings > Privacy & Security** and
   allow it there.
4. Run the setup:

   ```bash
   ./cyxwiz-setup/cyxwiz-setup
   ```

   The setup downloads and verifies the signed installer, then opens the
   **CyxWiz Installer** window.
5. On the **Workloads** tab, keep **CPU only** (or **Recommended**, which is
   the same on macOS today).
6. Optional: on the **Installation location** tab, change the folder.
   - **Install for me** (default) installs to `~/Applications/CyxWiz`.
   - **Install for all users** installs to `/Applications/CyxWiz` and needs
     administrator rights.
7. Select **Review & install**, check the list in **Review installation**,
   then select **Apply changes**.
8. Wait for **Installation complete**, then select **Launch CyxWiz**.

The install folder holds **CyxWiz.app** and **CyxWiz Installer.app**. Start
the Engine from `CyxWiz.app` next time. Continue with
[First launch](first-launch.md).

## Where things go

| What | Where |
| --- | --- |
| Engine, installer and runtimes | The install folder you chose |
| Engine settings | `~/.cyxwiz` |
| Device verification results | `~/Library/Application Support/CyxWiz` |
| Setup download cache | `~/Library/Caches/CyxWiz/Setup` |
| Your projects | Where you create them |

## If something goes wrong

**"cannot be opened because the developer cannot be verified"**: macOS
blocks unnotarized binaries. Remove the quarantine flag as in step 3, or
allow the app under **System Settings > Privacy & Security**.

**The setup prints an error and exits**: it could not download or verify the
installer. Check your internet connection and run it again.

**"Python not found"** on the start page: install Python 3.12 from
python.org or with `brew install python@3.12`, then select **Scan again** in
the **Python for scripting** dialog. See
[First launch](first-launch.md#python-for-scripting).

## Related

- [First launch](first-launch.md)
- [Update or uninstall](update-and-uninstall.md)
