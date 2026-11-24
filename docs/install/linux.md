---
title: Install on Linux
description: Run the CyxWiz setup on Linux x86_64, install the CPU Engine, and add Python 3.12 for scripting.
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

Use this when: you want the CyxWiz Engine on a Linux x86_64 computer.

## Before you start

- Linux x86_64 with a desktop session and OpenGL. Releases are built on
  Ubuntu 22.04; older distributions are untested.
- Linux installs are **CPU only**. No GPU pack is published for Linux yet,
  so the installer shows CUDA, OpenCL and oneAPI as **Not published**.
- Optional, for scripts and the Python console: Python 3.12 with its `venv`
  module. See [Python for scripting](#python-for-scripting).

## Steps

1. On the [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases),
   download `cyxwiz-setup-ubuntu-x64.tar.gz` from the newest release.
2. In a terminal, extract it and run the setup:

   ```bash
   mkdir cyxwiz-setup
   tar -xzf cyxwiz-setup-ubuntu-x64.tar.gz -C cyxwiz-setup
   ./cyxwiz-setup/cyxwiz-setup
   ```

   The setup downloads and verifies the signed installer, then opens the
   **CyxWiz Installer** window.
3. On the **Workloads** tab, keep **CPU only** (or **Recommended**, which is
   the same on Linux today).
4. Optional: on the **Installation location** tab, change the folder.
   - **Install for me** (default) installs to `~/.local/share/cyxwiz`, or
     `$XDG_DATA_HOME/cyxwiz` when that variable is set.
   - **Install for all users** installs to `/opt/cyxwiz` and needs
     administrator rights.

   Type a folder and press Enter, or select **Browse...**.
5. Select **Review & install**, check the list in **Review installation**,
   then select **Apply changes**.
6. Wait for **Installation complete**, then select **Launch CyxWiz**.

The installer adds **CyxWiz** and **CyxWiz Installer** to your desktop's
application menu (`cyxwiz.desktop` and `cyxwiz-installer.desktop`). Start the
Engine from there next time. Continue with [First launch](first-launch.md).

## Where things go

| What | Where |
| --- | --- |
| Engine, installer and runtimes | The install folder you chose |
| Engine settings | `~/.cyxwiz` |
| Device verification results | `~/.local/state/cyxwiz` (or `$XDG_STATE_HOME/cyxwiz`) |
| Setup download cache | `~/.cache/cyxwiz/setup` (or `$XDG_CACHE_HOME/cyxwiz/setup`) |
| Your projects | Where you create them |

## Python for scripting

The Engine embeds Python 3.12. Scripts, the Python console and project
environments also need a Python 3.12 on the system with its `venv` module.
Everything else in the Engine works without it.

At start the Engine looks for `python3.12` on `PATH` and in `/usr/bin`,
`/usr/local/bin` and `/opt/python3.12/bin`. It lists other versions it finds
as not usable. Install 3.12 next to your system Python; it does not replace
`python3`.

```bash
# Ubuntu 24.04 and Debian 13
sudo apt-get install -y python3.12 python3.12-venv
# Ubuntu 22.04: from the deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get install -y python3.12 python3.12-venv
# Arch Linux: the AUR package python312 (installs /usr/bin/python3.12)
sudo pacman -S --needed base-devel git tk
git clone https://aur.archlinux.org/python312.git && cd python312
makepkg -f && sudo pacman -U python312-3.12.*-x86_64.pkg.tar.zst
```

Where no package exists, build Python from python.org into `/usr/local`. It
takes about 15 minutes; `altinstall` adds `python3.12` without touching
`python3`. Install the development packages for OpenSSL, libffi, zlib, bzip2,
xz, SQLite, readline and gdbm first, or the matching Python modules are left
out.

```bash
curl -LO https://www.python.org/ftp/python/3.12.16/Python-3.12.16.tar.xz
tar xf Python-3.12.16.tar.xz && cd Python-3.12.16
./configure --prefix=/usr/local --enable-shared --enable-optimizations --with-ensurepip=install \
  LDFLAGS="-Wl,-rpath,/usr/local/lib"
make -j"$(nproc)" && sudo make altinstall
python3.12 -c "import venv, ssl, sqlite3, ctypes"   # no output means the modules the Engine uses are there
```

Then select **Scan again** in the Engine's **Python for scripting** dialog,
or restart the Engine. The dialog is described in
[First launch](first-launch.md#python-for-scripting).

## If something goes wrong

**The setup prints an error and exits**: it could not download or verify the
installer. Read the message, check your internet connection, and run it
again. Run `./cyxwiz-setup/cyxwiz-setup --help` for its options.

**"Python not found"** on the start page: no Python 3.12 with `venv` was
found. Install it as described in [Python for scripting](#python-for-scripting).

**No GPU in Preferences > Devices**: expected. Linux installs compute on the
CPU only.

## Related

- [First launch](first-launch.md)
- [Update or uninstall](update-and-uninstall.md)
- [GPU packs](gpu-packs.md)
