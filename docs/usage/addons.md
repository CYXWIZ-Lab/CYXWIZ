# Add-ons

Add-ons are optional features delivered as signed packages, the same way as
the GPU packs, but they add a feature instead of a compute device. The first
one is **Simulation (MuJoCo)**: MuJoCo environments, the 3D viewport and the
Environment Library.

An add-on needs no device check. It is ready when the Engine next starts.

## Installing one

In the CyxWiz Installer, open **Individual components**. The **Add-ons**
heading sits below the compute cards.

1. Tick **Simulation (MuJoCo)**. The profile becomes **Custom**, and the add-on
   appears in the installation summary. Packs the profile already included
   stay selected.
2. Choose **Review & install**, then **Apply changes**.

Neither **Recommended** nor **CPU only** installs an add-on: you choose it.
**CPU only** never removes an installed add-on.

From the Engine, **Preferences > Installed packs > Add-ons > Install add-on...**
opens the installer with the add-on ticked.

## What the card tells you

| Status | Meaning |
| --- | --- |
| Optional / Selected | Offered by the signed catalog; ticked or not |
| Installed | The Engine loads it at start. To turn it off for a while, use Apps > Plugin Manager |
| Update available | A newer signed package exists; tick the add-on to install it |
| Update blocked | The catalog offers an older or conflicting package, or one built for another CyxWiz Engine release; the installed one stays |
| Comes with the Engine update | The CPU Engine update in this installation reinstalls it for the new release; nothing to tick |
| CPU Engine update available | In the Engine: the Installer reinstalls it with the available CPU Engine update |
| After the Engine update | Not installed while a CPU Engine update is pending: install the update first, then add it |
| Needs Engine update | It is built for another CyxWiz Engine release: update the CPU Engine first, then add it |
| Needs reinstall | It was installed for another CyxWiz Engine release, so the Engine no longer loads it. Tick it to install the version for this release, or uninstall it |
| Unavailable | Its signed details could not be verified, or the release does not offer it |
| Not published | No signed package for this platform and release (macOS and Linux today) |

The card shows the version, the download size, and the third-party licences
the package includes (for MuJoCo: MuJoCo by Google DeepMind, Apache-2.0, and
the Gymnasium task models, MIT). The licence files are in the package's
`THIRD_PARTY_LICENSES` folder. In the Engine, **Details** lists the pack id,
the plugin id, the CPU Engine it is built for and the licence files.

## When the CPU Engine is updated

An add-on is built for one CPU Engine release. An update of the CPU Engine
also installs the new release's version of every installed add-on: the
installation summary lists it as reinstalled for the new CPU Engine, and its
card reads **Comes with the Engine update**. When the new release has no
version of an add-on yet, a warning at the top of the summary and on its card
says the Engine will stop loading it; after the update its card shows
**Needs reinstall** and keeps **Uninstall add-on**.

An add-on that is not installed cannot be added in the same run as a CPU
Engine update: its card reads **After the Engine update** and cannot be
ticked. Install the update, then open the installer again to add it. In
recovery the installer changes nothing, so no card shows these states.

## Uninstalling one

In the installer, **Uninstall add-on** on the card asks to confirm, then
removes the add-on's files and its downloaded package. Your projects are not
changed. In the Engine, **Uninstall add-on...** opens the installer at that
confirmation; the Engine stops loading it at its next start.

## Limits

- An add-on is built for one CyxWiz Engine/CPU base. A different base shows
  **Needs Engine update**.
- A package published for another CyxWiz release does not verify for this
  one. Its signed details cannot be read, so it is listed under the compute
  components as an unavailable package, not as an add-on.
