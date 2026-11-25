---
title: Install and first launch problems
description: Fix installer messages, macOS launch blocks and the Python for scripting messages the Engine shows at its first start.
type: troubleshooting
area: install
audience: user
status: stable
since: 1.0.7
owner: install
last_reviewed: 2026-11-25
related:
  - ../install/index.md
  - ../install/first-launch.md
  - ../install/update-and-uninstall.md
  - gpu-and-drivers.md
---

The installer messages are described with their install step on the
[Windows](../install/windows.md), [Linux](../install/linux.md) and
[macOS](../install/macos.md) pages. This page collects them by message,
with the Python messages the Engine shows at its first start.

## A message box titled "CyxWiz Setup"

**Cause:** the setup program could not download or verify the installer.

**Fix:**

1. Read the message in the box.
2. Check your internet connection, then run the setup again.

On Linux and macOS the setup prints the error in the terminal and exits;
run `./cyxwiz-setup/cyxwiz-setup --help` for its options.

## "Catalog unavailable"

**Cause:** the installer could not read the signed list of packages.

**Fix:** select **Refresh catalog**. You can still install the CPU Engine;
GPU packs need the catalog.

## "Review & install" is greyed out

**Cause:** something on the page is not finished. The reason is shown under
the button, for example "Finish the installation location first: press Enter
after typing it, or choose a folder with Browse...".

**Fix:** do what the reason says.

## "Uninstall requires a manager running outside the installation"

**Cause:** the installer you opened runs from inside the install folder.

**Fix:** select **Uninstall CyxWiz**; the installed manager reopens and you
continue there. See [Update or uninstall](../install/update-and-uninstall.md).

## "Uninstall is blocked by a loaded product library"

**Cause:** a CyxWiz program is still running.

**Fix:** close the Engine and any Server Node, then try again.

## "Recovery required"

**Cause:** an earlier install, update or uninstall did not finish.

**Fix:** follow the installer's message. It recovers the installation before
it lets you change it again.

## "cannot be opened because the developer cannot be verified"

**Cause:** macOS blocks binaries that are not notarized.

**Fix:** remove the quarantine flag as described in
[Install on macOS](../install/macos.md), or allow the app under
**System Settings > Privacy & Security**.

## "Python not found"

The Python chip on the start page reads **Python not found**, and the
**Python for scripting** dialog says "No Python 3.12 with the venv module was
found".

**Cause:** the Engine embeds Python 3.12. Scripts, the Python console and
project environments need a Python 3.12 installed on the computer, with its
`venv` module. Building graphs, training and Data Studio work without it.

**Fix:**

1. Install Python 3.12:
   - Windows: from python.org, with **pip** ticked in its installer.
   - macOS: from python.org, or `brew install python@3.12`.
   - Linux: `sudo apt install python3.12 python3.12-venv` on Debian and
     Ubuntu, the AUR package `python312` on Arch Linux, or a build from
     python.org. See [Python for scripting on Linux](../install/linux.md#python-for-scripting).
2. In **Python for scripting**, select **Scan again**.
3. Select the Python under **Usable**, then **Use this Python**.

To go on without scripting, select **Continue without scripting**. You can
set Python later in **File > Preferences... > Python**.

## "...; the interpreter saved in the settings no longer works"

**Cause:** the Python you chose earlier was moved, removed or upgraded.

**Fix:** select **Scan again**, or **Choose an interpreter...** and pick the
new `python` executable.

## "This Engine embeds Python 3.12 and needs a matching 3.12; found 3.x"

Shown under **Found but not usable** for each Python of another version.

**Cause:** an embedded Python accepts only its own minor version, so a
Python 3.11 or 3.13 cannot run the Engine's scripts.

**Fix:** install Python 3.12 beside the version you have, then select
**Scan again**. You do not need to remove the other version.

## "Python version too old" or "Python version unsupported"

**Cause:** the Python found is older than 3.12 or newer than 3.13 ("supported
versions: 3.12-3.13").

**Fix:** install Python 3.12, then select **Scan again**.

## "Python missing venv module (required for creating virtual environments)"

**Cause:** this Python has no `venv` module. On Debian and Ubuntu it is a
separate package.

**Fix:** install it (`sudo apt install python3.12-venv`), then select
**Scan again**.

## "One usable Python was found. Use it?" or "Several usable Pythons were found. Choose one."

Not an error. The chip reads **Choose a Python**.

**Fix:** select the Python you want under **Usable**, then **Use this
Python**.

## "Checking Python before opening the workspace..."

Not an error. You opened a project before the Python scan finished; the
project opens when it ends.

## The start page shows no task starter graphs

**Cause:** the Engine looks for `examples/cyxgraph` next to itself. Installed
releases include it; a moved or partial install may not.

**Fix:** reinstall with the installer, or copy the `examples` folder of the
release next to the Engine.

## Alpha 1.0.3 or 1.0.4 is still installed

**Cause:** those alphas have a Windows uninstall defect.

**Fix:** back up `%LOCALAPPDATA%\CyxWiz` before you uninstall them.

## Related

- [Start the Engine for the first time](../install/first-launch.md)
- [Update or uninstall](../install/update-and-uninstall.md)
- [GPU and drivers](gpu-and-drivers.md)
