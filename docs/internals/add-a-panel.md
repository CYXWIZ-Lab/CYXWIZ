---
title: Add a panel to the Engine
description: Add a dockable panel to the Engine window, list it in the sidebar and View > Panels, and build it from the shared tokens and widgets.
type: how-to
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - architecture/overview.md
  - architecture/threading.md
  - coding-standards.md
  - testing.md
  - ../main-window/layouts-and-docking.md
---

Use this when: you add a built-in window to the Engine, such as a tool or
a monitor. A panel that ships in a plugin goes through the plugin panel
registry instead; see [Plugin API reference](../sdk/plugin-api-reference.md).

First decide whether it should be a panel at all. Anything that transforms
pipeline data during a run is a node; see [Add a built-in node](add-a-node.md).
A panel is right for a tool you open by hand, an inspector that reads data
without changing it, or a monitor of Engine state.

## Before you start

- A source build; see [Build from source](build-from-source.md).
- A screenshot of the screen the panel sits next to, and a mockup the
  maintainers approved for a new screen.
- The threading rules: long work never runs in `Render()`. See
  [Threading and async tasks](architecture/threading.md).

## Steps

1. **Write a presentation model first.** In `cyxwiz-engine/src/core/`, add
   `<name>_presentation.h` and `.cpp`: plain data in, plain data out, with
   no ImGui, no registries and no backend calls. It decides every word,
   status, enabled state and disabled reason the panel shows. More than 50
   screens follow this pattern, for example
   `compile_report_presentation.h`.
2. **Test the presentation model.** Add
   `cyxwiz-engine/tests/test_<name>_presentation.cpp` (a plain `main` with
   a `Check` helper, like `test_compile_report_presentation.cpp`), an
   `add_executable` target that compiles only the test and the sources it
   needs, and an `add_test` line in `cyxwiz-engine/CMakeLists.txt`.
3. **Write the panel.** In `cyxwiz-engine/src/gui/panels/`, add
   `<name>_panel.h` and `.cpp` with a class that derives from
   `cyxwiz::Panel` (`gui/panel.h`) and implements `Render()`. Give the
   constructor the window title, and the initial visibility (normally
   hidden). `Render()` returns at once when the panel is not visible, asks
   the presentation model what to show, and draws it.
4. **Draw with the shared kit only.**
   - Colours come from `cyxwiz::ui::CurrentTokens()` (`gui/ui_tokens.h`);
     never type a colour literal. Status words and colours come from
     `StatusStyleFor()`.
   - Buttons are the `cyxwiz::ui` buttons in `gui/ui_buttons.h`
     (`PrimaryButton`, `SecondaryButton`, `GhostButton`, `DangerButton`,
     `LinkButton`, chips and segmented controls), never a default
     `ImGui::Button`. A disabled button gets a reason, shown as a tooltip.
   - Numbers are typed fields with - / + steps, `ui::NumberField`
     (`gui/ui_widgets.h`), never a drag or slider.
   - Layout helpers are in `gui/ui_widgets.h`: `SectionHeader`,
     `BeginCard`/`EndCard`, `KeyValueTable`, `FactsTable`, `StatusChip`,
     `EmptyState`, `FlowRow` (wraps toolbar items instead of clipping
     them), `BeginDialog`/`EndDialog` and `ConfirmDialog`.
   - No outlines on fields and buttons; only the window has a border.
     Tables and editors sit on the window background.
   - Every option a user can change also gets a saved default in
     Preferences (the tabs are `gui/panels/toolbar_preferences_*.cpp`,
     the values `core/engine_config.h`).
5. **Add the sources to CMake.** List the `.cpp` and `.h` files in the
   Engine's source and header lists in `cyxwiz-engine/CMakeLists.txt`,
   next to the other panels.
6. **Own the panel in the main window.** In `gui/main_window.h`, forward
   declare the class and add a `std::unique_ptr` member. In
   `gui/main_window.cpp`:
   - create it in the constructor, next to the panels of its area, and
     wire its callbacks;
   - render it with `TimedRender("<name>", <name>_panel_)` in the render
     pass of its area (the timer feeds the frame metrics);
   - add it to the list of panels the default layout hides or shows;
   - reset it in the destructor in the same order as its neighbours.
7. **List it in the sidebar and View > Panels.** In
   `MainWindow::RegisterPanelsWithSidebar()`, add
   `add("<Title>", ICON_FA_..., <name>_panel_, "<Group>");`. The groups are
   Workspace, Data, Training, Network and Extensions; the order of the calls
   is the order on screen. **View > Panels** is built from the same list
   (`core/menu_presentation.cpp`), so the panel appears in both with one
   line.
8. **Optional: a menu command.** For an entry in another menu (for
   example **Tools**), add an action id to the menu table in
   `core/menu_presentation.cpp` and map it to a handler in
   `gui/panels/toolbar_menus.cpp`. The same table feeds the Command Palette,
   the shortcut handler and **Preferences > Shortcuts**, so a shortcut you
   print there is the one that runs.
9. **Move long work off the UI thread.** Capture what the work needs on the
   UI thread, then run it as an `AsyncTaskManager` task so it shows in
   Background Tasks and the window keeps drawing. Drain the result at the
   top of the next `Render()`.
10. **Check it on screen.** Build, open the panel in the default theme and
    at the smallest window size, and check that no button is clipped and
    nothing is hidden. Run the new presentation test and the other
    `test_*_presentation` executables.
11. **Document it in the same change.** Add a page in `docs/main-window/`
    from the reference template, link it from `docs/main-window/index.md`,
    and add a release-note fragment in `docs/release-notes/next/`. Run
    `python scripts/docs/check_docs.py`.

## If something goes wrong

**The panel never appears:** it is created but not rendered (step 6), or
not registered with the sidebar (step 7), so nothing toggles its
visibility.

**The window freezes while the panel works:** the work runs in `Render()`.
Move it to a task (step 9).

**A colour looks wrong in a light theme:** a literal colour slipped in.
Read it from the tokens.

## Related

- [Layouts and docking](../main-window/layouts-and-docking.md)
- [Threading and async tasks](architecture/threading.md)
- [Coding standards](coding-standards.md)
- [Tests](testing.md)
