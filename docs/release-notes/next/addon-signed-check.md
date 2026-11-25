kind: security
area: extending
page: usage/addons.md

An installed add-on is checked against its signed package before the Engine loads it: every file of its folder (none missing or added, each one's size and SHA-256) and the library copies the Engine loads, which are held against changes until loaded. A damaged add-on is not loaded: the log names the file, the Plugin Manager shows it as **Not loaded** with **Repair add-on...**, and its card in **Preferences > Installed packs** and in the CyxWiz Installer reads **Damaged**; ticking it repairs it from the signed package (**Review & repair**). A verified add-on's Plugin Manager card reads *Signed add-on, verified at start*; a plugin from a plugin folder, *Developer plugin, not signed*. The MuJoCo add-on is now called **Simulation (MuJoCo)** everywhere, in the Plugin Manager too.
