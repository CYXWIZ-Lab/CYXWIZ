---
title: Install on Windows
description: Run the CyxWiz setup on Windows x64, choose a profile and folder, and start the Engine.
type: how-to
area: install
audience: user
status: stable
since: 1.0.7
owner: install
last_reviewed: 2026-11-24
related:
  - index.md
  - gpu-packs.md
  - first-launch.md
  - update-and-uninstall.md
redirect_from: []
---

Use this when: you want the CyxWiz Engine on a Windows x64 computer.

## Before you start

- Windows x64. Releases are built and tested on Windows Server 2022 runners;
  Windows 10 and 11 are the intended targets.
- Optional, for GPU training: a current driver for your NVIDIA or Intel GPU.
  The installer never installs drivers.
- Optional, for scripts and the Python console: Python 3.12 from
  python.org. You can add it later; see [First launch](first-launch.md).

The Microsoft Visual C++ runtime ships with CyxWiz. You do not need to
install it separately.

## Steps

1. On the [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases),
   download `cyxwiz-setup-windows-x64.zip` from the newest release.
2. Extract the zip, then run `cyxwiz-setup.exe`.

   The setup is not Authenticode-signed. If Windows SmartScreen shows
   **Windows protected your PC**, select **More info**, then **Run anyway**.
   The setup checks the signature of everything it downloads itself.
3. Wait while the setup downloads and verifies the installer. The
   **CyxWiz Installer** window opens, marked **Fresh installation**. The top
   right shows **Catalog verified** when the signed package list was read.
4. On the **Workloads** tab, choose a profile:
   - **Recommended**: the CPU Engine plus the GPU packs the catalog marks
     **Recommended** for this computer.
   - **CPU only**: the smallest install. It works on every supported machine.

   The **What will be installed** list shows the result. To pick packs one by
   one, use the **Individual components** tab instead; see
   [GPU packs](gpu-packs.md).
5. Optional: on the **Installation location** tab, change where CyxWiz goes.
   - **Install for me** (default) installs to
     `%LOCALAPPDATA%\Programs\CyxWiz`.
   - **Install for all users** installs to `%ProgramFiles%\CyxWiz` and needs
     administrator approval.

   To use another folder, type it and press Enter, or select **Browse...**.
   The installer uses an empty folder or a folder named `CyxWiz` as it is;
   for any other folder it creates a `CyxWiz` folder inside, because
   uninstalling deletes the whole install folder.
6. In the **Installation summary** column, check **Profile**, **Components**,
   **Required download** and **Install location**, then select
   **Review & install**.
7. In **Review installation**, read the list. Leave **Launch CyxWiz when
   installation completes** ticked if you want the Engine to open at the
   end, then select **Apply changes**.
8. Wait for **Installation complete**. The progress bar shows the package,
   the phase and the bytes downloaded. To stop, select
   **Cancel installation**; the installer cleans up before it closes.

The Engine opens, or select **Launch CyxWiz**. Continue with
[First launch](first-launch.md).

## Where things go

| What | Where |
| --- | --- |
| Engine, installer, runtimes and packs | The install folder you chose |
| Start Menu | **CyxWiz** and **CyxWiz Installer** |
| Uninstall entry | **Settings > Apps** (CyxWiz) |
| Engine settings | `%APPDATA%\CyxWiz` |
| Engine data (device verification results, run records, plugins) | `%LOCALAPPDATA%\CyxWiz` |
| Setup download cache | `%LOCALAPPDATA%\CyxWiz-Setup` |
| Your projects | Where you create them (default `Documents\CyxWiz Projects`) |

CyxWiz never changes your `PATH` or installs anything into Windows system
folders.

## Verify a manual download

Release metadata and packs are signed, and the setup checks them for you. If
you download files by hand, compare them with the release's
`SHA256SUMS.txt`:

```powershell
Get-FileHash .\cyxwiz-setup-windows-x64.zip -Algorithm SHA256
```

## If something goes wrong

**A message box titled "CyxWiz Setup"**: the setup could not download or
verify the installer. Read the message, check your internet connection, and
run the setup again.

**"Catalog unavailable"** in the installer header: the signed package list
could not be read. Select **Refresh catalog**. You can still install the CPU
Engine; GPU packs need the catalog.

**"Review & install" is greyed out**: the reason is shown under the button.
For example, "Finish the installation location first: press Enter after
typing it, or choose a folder with Browse...".

**Closing the window during an install** asks **Installation in progress**.
Select **Cancel installation and close** to stop safely, or
**Keep window open**.

**Alpha 1.0.3 or 1.0.4 is still installed**: those alphas have a Windows
uninstall defect. Back up `%LOCALAPPDATA%\CyxWiz` before you uninstall them.

Known GPU runtime problems are listed in
[Known issues](../known-issues.md),
which also ships in your install.

## Related

- [GPU packs](gpu-packs.md)
- [First launch](first-launch.md)
- [Update or uninstall](update-and-uninstall.md)
