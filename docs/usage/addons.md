# Add-ons

Add-ons are optional features delivered as signed packages, the same way as
the GPU packs, but they add a feature instead of a compute device. The first
one is **Simulation (MuJoCo)**: MuJoCo environments, the 3D viewport and the
Environment Library.

An add-on needs no device check. It is ready when the Engine next starts.

Each add-on is shown under the name and description its plugin declares: the
signed package carries them.

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
| Installed | The Engine loads it at start, after checking every file against its signed package. To turn it off for a while, use Apps > Plugin Manager |
| Damaged | A file of the installed add-on differs from its signed package (changed, missing, or one more), so the Engine does not load it. Tick it in the installer (it reads **Repair selected**) to put the signed files back |
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

An installed add-on's card also checks its files against the signed package,
with the installation's own trust file (the one the Engine checks with when it
loads the add-on). In the Engine this runs in the background: the card says
*Checking its files against the signed package...* until it is done, then
**Details** lists **Files** and **Checked**. When the trust file cannot be read
the card says so and the add-on is not called damaged.

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

## When an add-on is damaged

Each time the Engine starts, it checks an installed add-on's files against
the package's signed file list before loading it. A file that was changed,
removed or added stops the load:

- the log says *Add-on Simulation (MuJoCo) was not loaded:* and names the
  file, for example `bin\mujoco.dll does not match its signed package`;
- **Apps > Plugin Manager** shows the add-on as **Not loaded** with
  *Signed add-on, check failed at start*, what differs under **Details**, and
  **Repair add-on...**;
- its card in **Preferences > Installed packs** and in the installer reads
  **Damaged**.

To repair it:

1. In the Engine, choose **Repair add-on...** on the Plugin Manager card or
   on the add-on's card. The CyxWiz Installer opens with the add-on ticked
   (**Repair selected**). In the installer alone, tick the **Damaged** card.
2. Close the CyxWiz Engine: the installer does not change an installation
   while its Engine is open (it says *CyxWiz is in use*).
3. Choose **Review & repair**, then **Apply changes**. The installer puts the
   signed files back from the signed package; your graphs and projects are
   not touched.
4. Start the Engine again. The Plugin Manager card reads *Signed add-on,
   verified at start*.

A verified add-on's Plugin Manager card shows *Signed add-on, verified at
start* and its package version; **Details** lists the pack, how many files
were checked and how long it took, and the CPU Engine it is built for. A
plugin from a plugin folder shows *Developer plugin, not signed*.

## Uninstalling one

In the installer, **Uninstall add-on** on the card asks to confirm, then
removes the add-on's files and its downloaded package. Your projects are not
changed. In the Engine, **Uninstall add-on...** opens the installer at that
confirmation; the Engine stops loading it at its next start.

## Limits

- An add-on is built for one CyxWiz Engine/CPU base. A different base shows
  **Needs Engine update**.
- The check proves the files match the signed package kept in the
  installation. It does not stop someone who can change your user folders
  from changing that package's keys too.
- A package published for another CyxWiz release does not verify for this
  one. Its signed details cannot be read, so it is listed under the compute
  components as an unavailable package, not as an add-on.
