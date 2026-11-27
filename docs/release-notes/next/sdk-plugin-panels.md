kind: feature
area: extending
page: extending/plugin-api-reference.md

A plugin built from the installed plugin SDK can now draw panels. The SDK carries the Engine's own Dear ImGui (`include/imgui`, the static library in `lib`, its MIT licence in `licenses`) as the CMake target `CyxWiz::PluginImGui`; link it beside `CyxWiz::PluginSDK`. A plugin that links it is fingerprinted with its ImGui, and the Engine refuses one built with another ImGui, naming both, instead of loading it and crashing as it draws. A plugin that does not link it is not affected. To use ImGui's vector operators, set `IMGUI_DEFINE_MATH_OPERATORS` as a compile definition. A new example, `plugins/examples/panel_example` (a damped spring with typed fields and a plot), is built from the installed SDK by the SDK check. `cyxwiz-plugin-check --scratch` runs the check in a user profile of its own, so scripted checks leave your plugin cache and settings as they were.
