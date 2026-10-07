# Windows of their own

The big windows can leave the Engine's main window and become windows of
their own on the desktop: on a second monitor, next to the Engine, or over
it. They come back with one command.

## Which windows

Dashboard, Plot, the plot view's own window, Plot Output, Data Studio, Script
Editor (and the notebooks in it), Training, Console, Table Viewer (Data
Viewer), Data Explorer, Variable Explorer and the CyxWiz Studio canvas.

Everything else stays in the main window: Properties, Asset Browser, Node
Browser, the tool panels, dialogs, the sidebar, the status bar. Dragging one
of them past the edge stops at the edge, as before.

## Moving a window out

- Right-click the window's tab (or the title bar of a floating window) and
  take **Move to its own window**. The window lifts off where it is, as a
  window of the operating system; drag it anywhere, on any monitor.
- Or drag the tab out of the main window and past its edge.
- Or **View > Window > Move to its own window** for the window that has the
  focus.

The new window is titled *<window name> - CyxWiz Engine* (the main window
is *CyxWiz Engine - <project>*). It minimises and maximises on its own,
keeps the theme's colours, and the Engine's shortcuts work in it as they do
inside (the window that has the focus gets them). Export PNG of a plot in
its own window captures that window.

A window that left stays its own window when it happens to overlap the main
window: only docking brings it back.

## Bringing it back

- Right-click its tab or title bar: **Dock back into the main window**. It
  returns to the dock node it left; when that node is gone, to the main
  dock space.
- Or drag its tab into the main window's dock space.
- Or **View > Window > Dock back into the main window**.
- **View > Layout > Reset to Default** brings every window back inside.

Closing a window of its own (the title bar's X) closes that panel, exactly
like closing its tab; View > Panels (or the sidebar) opens it again, inside.

## Between sessions

Where a window is (inside, or outside at which position and size) is kept
with the layout in `imgui.ini`, next to the dock layout. A window that was
outside opens outside again next time, at the same place.
