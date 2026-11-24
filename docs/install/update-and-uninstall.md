---
title: Update or uninstall CyxWiz
description: Update the Engine and GPU packs, roll back a runtime, remove one pack, or uninstall CyxWiz.
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
  - windows.md
redirect_from: []
---

Use this when: a new release is out, a GPU pack misbehaves, or you want
CyxWiz off your computer.

All of this happens in the **CyxWiz Installer**. When CyxWiz is already
installed, its header says **Modify installation** and the location is fixed.

## Open the installer

- **Windows**: **CyxWiz Installer** in the Start Menu, or **Settings > Apps**,
  CyxWiz, then **Uninstall** (this opens the installer; nothing is removed
  until you confirm).
- **Linux**: **CyxWiz Installer** in your desktop's application menu.
- **macOS**: **CyxWiz Installer.app** in the install folder.
- **From the Engine**: **File > Preferences... > Devices**, then
  **Open CyxWiz Installer...** under **Installed packs**. This button works
  only in an installed Engine, not in a source build.

## Update

The Engine does not check for updates by itself yet: **Help > Check for
Updates...** is shown as planned and is disabled. Check from the installer.

1. Open the installer and select **Check for updates** in the header.
2. A component with a newer signed package shows **Update available**. The
   summary lists **CPU Engine (update available)** when the Engine itself has
   an update.
3. To update a GPU pack, select the **Individual components** tab and tick
   it ("Select this component to review and install its signed update.").
4. Select **Review & update**. In **Review installation**, check the list,
   then select **Apply changes**.
5. Wait for **Installation complete**, then select **Launch CyxWiz**.

An Engine update keeps the previous complete runtime, so you can roll back.
Close the Engine before you apply an update.

## Roll back to the previous runtime

1. In the Engine, open **File > Preferences... > Devices**.
2. Under **Installed packs**, select **Roll back runtime**. The button is
   disabled, with the reason as a tooltip, when there is nothing to roll
   back.
3. In **Confirm Backend Maintenance**, select **Queue for Exit**.
4. Select **Exit and Apply**. The Engine closes, the previous runtime is
   restored, and the next start uses it.

**Repair...** and **Remove...** in a pack's **Actions** menu work the same
way: they are queued and applied after the Engine exits.

## Remove one GPU pack

1. Open the installer and select **Individual components**.
2. On the installed pack, select **Uninstall component**.
3. In **Uninstall backend component**, select **Uninstall**. CyxWiz turns
   off the route, removes the pack's files and clears its download cache.
   Select **Keep component** to cancel.

The CPU Engine cannot be removed this way; it is **Required**.

## Uninstall CyxWiz

1. Close the Engine.
2. Open the installer and select **Uninstall CyxWiz** in the summary column.
3. In **Remove CyxWiz**, read the install folder under **Installation root**,
   tick **I understand that this cannot be undone**, then select
   **Uninstall all**.
4. Keep the window open until it says "CyxWiz was uninstalled successfully.
   You can install it again or close this window."

Uninstall removes the Engine, the installer, the CPU runtime and every GPU
pack in the install folder, and the Start Menu, desktop or app entries.

It does **not** remove:

- your projects and datasets stored outside the install folder;
- your Engine settings (`%APPDATA%\CyxWiz` on Windows, `~/.cyxwiz` on Linux
  and macOS);
- Engine data such as device verification results (`%LOCALAPPDATA%\CyxWiz`
  on Windows, `~/.local/state/cyxwiz` on Linux,
  `~/Library/Application Support/CyxWiz` on macOS).

Delete those folders yourself if you want a clean slate. A later install
picks the settings up again.

## If something goes wrong

**"Uninstall requires a manager running outside the installation"**: the
installer you opened runs from inside the install folder. Selecting
**Uninstall CyxWiz** reopens the installed manager; continue there.

**"Uninstall is blocked by a loaded product library"**: a CyxWiz program is
still running. Close the Engine and any Server Node, then try again.

**"Not uninstalled: CyxWiz is still installed."**: the message after it says
why. Fix that and select **Uninstall CyxWiz** again.

**"Recovery required"** in the header: an earlier change did not finish. The
installer must recover before it can modify the installation; follow its
message.

**Alpha 1.0.3 or 1.0.4 on Windows**: those alphas have an uninstall defect.
Back up `%LOCALAPPDATA%\CyxWiz` before you uninstall them.

## Related

- [Add a GPU pack and verify it](gpu-packs.md)
- [Install on Windows](windows.md)
