---
title: Theme Editor and node icons
description: The Theme Editor (every colour the Engine draws, the shape of fields and windows, saved themes) and View > Node Icon Pack.
type: reference
area: general
audience: user
status: experimental
since: 1.0.8
owner: main-window
last_reviewed: 2026-11-25
related:
  - preferences.md
  - menus.md
  - editing-values.md
redirect_from:
  - ../usage/theme_editor.md
---

The **Theme Editor** changes every colour the Engine draws, and the shape
of its fields, buttons and windows. Every row changes the screen at once.

Open it with **View > Theme Editor...**, or **File > Preferences >
Appearance > Theme Editor...** (Preferences closes as with OK, then the
editor opens in front).

## Base theme

The **Base theme** list at the top picks the preset the edits start from
(CyxWiz Dark, CyxWiz Light, CyxWiz Cream, Unreal Engine, Dracula, ...).
Picking one applies it to the whole Engine. **View > Theme** and
**File > Preferences > Appearance** offer the same themes.

Each preset brings its own **accent**: purple in CyxWiz Dark and CyxWiz
Cream, blue in CyxWiz Light and CyxWiz Launch, orange in Unreal Engine, and
so on.

## Tabs

| Tab | What it holds |
| --- | --- |
| **Engine colours** | The **Accent** (Train, OK and every primary button, selection, check marks, links, the selected tab's line, group frames on the canvas); the 24 **Node families** (the fill of each kind of node on the canvas); the **Data types** (pins and links by what flows through them, and the plot series colours in that order); the **Status words** (Ready, warnings, errors, running ...). A status colour that is hard to read on the window shows its contrast and the 4.5:1 target under it. **Reset the Engine colours to the base theme's** takes the preset's again. |
| **Surfaces** | Window, text, fields, buttons, bars, tabs, tables, lists, menus and popups, dialogs, scrollbars. A filter box narrows the rows. What follows the accent (check marks, selection, the tab line) has no row here; splitters and resize grips follow the window colour. |
| **Canvas** | The grid (background, lines, major lines). Nodes, pins and links use the Engine colours. |
| **Shape** | Rounding, borders, padding, spacing and sizes, as typed numbers with - / + steps. Fields and buttons have no outline (a fill marks them); windows keep a line. |
| **Saved themes** | Your custom themes, with **Apply**, and where they are stored. |

## Colour fields

Each colour row has a swatch (click it for a picker), a **hex** field
(`#RRGGBB`, or `#RRGGBBAA` where the colour has opacity; anything that is
not six or eight hex digits is ignored) and **R**, **G**, **B** (and **A**)
typed from 0 to 255, with - / + steps.

## Keeping your changes

Edits apply live but last only for the session until they are saved. While
there are unsaved changes the editor says so, with **Save as custom
theme...** and **Discard changes**. A custom theme keeps the base theme,
every surface colour, the shape, and the Engine colours (accent, node
families, data types, status words). The saved theme is applied again at
the next start.

## Node Icon Pack

**View > Node Icon Pack** picks the icon set drawn on the nodes of the
canvas. It changes only the icons, not the colours. The packs:

| Pack | Style |
| --- | --- |
| Font Awesome (default) | Solid icons |
| Tabler Icons | Outline icons |
| Remix Icon | Filled icons |
| Lucide Icons | Thin outline icons |
| Iconoir | Simple outline icons |
| Phosphor Icons | Outline icons |

The choice applies at once, to every project, and is kept for the next
start. The same list is **Node icons** in **File > Preferences >
Appearance**.

## Related

- [Preferences](preferences.md)
- [Menus](menus.md)
- [Editing values](editing-values.md)
