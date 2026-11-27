kind: fixed
area: extending
page: usage/addons.md

**An add-on package carries the add-on's own licence.** The `LICENSE` file of an add-on package is now the licence of its plugin, the one the plugin declares (for Simulation (MuJoCo): the Apache License 2.0). Before, the package carried the Engine's licence text in that place. The third-party licences in `THIRD_PARTY_LICENSES` are unchanged.
