# Example Panel

A plugin with a panel, built from the installed CyxWiz plugin SDK alone. It draws **Spring
Example**, a damped spring with three typed settings (mass in kg, stiffness in N/m, damping in N s/m),
a **Release** button and a live plot of the position in metres.

It shows what a plugin with a screen needs:

- `SetImGuiContext`: it draws in the Engine's ImGui context;
- `IPanelProvider`: the panel appears in the Engine's menus and is drawn every frame it is open;
- `PluginContext::GetUi()`: the Engine's own buttons, typed number fields and theme colours;
- `CyxWiz::PluginImGui` in `CMakeLists.txt`: the ImGui the Engine was built with. A plugin built
  with another ImGui is refused when it loads.

## Build it against the SDK

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<the CyxWiz SDK folder>
cmake --build build --config Release
```

The plugin folder is `build/panel_example` (`bin/` and `plugin.json`). Install it with **Plugins >
Plugin Manager > Install...**, then open **Spring Example** from the menu the Plugin Manager lists for
it, and press **Release**: the position swings and settles. With the defaults (1 kg, 40 N/m,
0.8 N s/m) it swings at about 1 Hz and halves its swing in under two seconds.

## Check it

```sh
cyxwiz-plugin-check build/panel_example
```

It loads the plugin as the Engine does and lists the panel (the check draws no panels).
