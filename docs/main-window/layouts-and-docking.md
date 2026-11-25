---
title: Layouts and docking
description: Arrange the Engine's panels, move big windows out to a monitor of their own, and save or reset the layout.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-24
related:
  - index.md
  - menus.md
redirect_from:
  - ../usage/windows.md
---

Every panel lives in the dock area of the [main window](index.md). You can
rearrange the panels by dragging their tabs, move the big windows out to a
window of their own (on a second monitor, for example), and reset
everything to the default layout.

## Show and hide panels

- **View > Panels** lists every panel by group. Click one to show or hide
  it.
- The tool sidebar has one icon per panel: click to show or hide.
- A panel's tab close box hides it; the menu or the sidebar brings it back.

The default arrangement is described in
[The main window](index.md#the-panels-in-the-default-layout).

## Rearrange panels

- Drag a panel's tab to another dock area to make it a tab there.
- While you drag, drop targets appear. Drop on the middle target to join
  the area's tabs, or on an edge target to split the area.
- Drag the line between two areas to resize them.
- Drag a tab outside the dock area to make it float over the main window.
  For most panels it stops at the main window's edge; see
  [Windows of their own](#windows-of-their-own) to take it further.

## Save and reset the layout

- **View > Layout > Save Layout** writes the arrangement now.
- **View > Layout > Reset to Default** rebuilds the default layout: the
  default panels only, and every window of its own docked back.

The layout is kept in `imgui.ini`, in the folder the Engine starts in. It
is the same for every project: opening another project does not change
where panels are. Closing a project rebuilds the default layout.

## Windows of their own

The big windows can leave the main window and become windows of the
operating system: Dashboard, Plot, the plot view's own window, Plot
Output, Data Studio, Script Editor (and its notebooks), Training, Console,
Table Viewer, Variable Explorer and the CyxWiz Studio canvas.

Everything else stays in the main window by default: Properties, Asset
Browser, Nodes, the tool panels, dialogs, the sidebar and the status bar.

**Any window, if you want.** **Preferences > General > Windows > Any
window can have a window of its own** lets every panel leave the main
window. It applies at once and is saved. The sidebar, the status bar,
menus, popups and the canvas overlays (minimap, find bar) always stay.
Turning it off brings the panels it let out back inside; the big windows
stay where they are.

### Move a window out

Use any of these:

- Right-click the window's tab (or the title bar of a floating window) and
  select **Move to its own window**. Drag the new window anywhere, on any
  monitor.
- Drag the tab out past the main window's edge.
- Focus the window and select **View > Window > Move to its own window**.
  Greyed out, it says why: "No window has the focus.", or the window stays
  in the main window.

The new window is titled *<window name> - CyxWiz Engine*. It minimises
and maximises on its own, keeps the theme's colours, and the Engine's
shortcuts work in it (the window that has the focus gets them). A window of
its own stays separate even when it overlaps the main window: only docking
brings it back.

### Bring it back

- Right-click its tab or title bar and select **Dock back into the main
  window**. It returns to the place it left when that place still exists;
  otherwise it comes back as a tab beside CyxWiz Studio, from where you can
  drag it.
- Drag its tab into the main window's dock area.
- **View > Window > Dock back into the main window** for the focused
  window.
- **View > Layout > Reset to Default** brings every window back.

Closing a window of its own (its title bar's close box) hides that panel,
like closing its tab. **View > Panels** or the sidebar opens it again,
inside the main window.

### Between sessions

Where each window is (inside, or outside at which position and size) is
kept with the layout in `imgui.ini`. A window that was outside opens
outside again next time, at the same place.

## Which window comes to the front

Whatever you open last comes to the front, wherever it is: a docked tab
behind another tab, a floating window, or a window of its own. Opening a
node's dialog, a Plot or Dashboard node's window, a table in the Table
Viewer, a file in the Script Editor, or **Show on canvas** each brings its
window forward. When one action opens two windows (a double-click on a node
selects it and opens its dialog), the one opened last, the dialog, is in
front.

## Full screen

**View > Full Screen** (F11) shows the Engine window without its frame over
the whole monitor. F11 again restores it. A window of its own can still sit
on top of it.

## Related

- [The main window](index.md)
- [Preferences](preferences.md)
