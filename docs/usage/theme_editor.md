# Theme Editor

The Theme Editor changes every colour the Engine draws, and the shape of its
fields, buttons and windows. Every row changes the screen at once.

Open it with **View > Theme Editor...**, or **File > Preferences > Appearance >
Theme Editor...** (Preferences closes as with OK, then the editor opens in
front).

## Base theme

The **Base theme** list at the top picks the preset the edits start from
(CyxWiz Dark, CyxWiz Light, CyxWiz Cream, Unreal Engine, Dracula, ...). Picking
one applies it to the whole Engine. File > Preferences > Appearance shows the
same themes as cards.

Each preset brings its own **accent**: purple in CyxWiz Dark and CyxWiz Cream,
blue in CyxWiz Light and CyxWiz Launch, orange in Unreal Engine, and so on.

## Tabs

| Tab | What it holds |
| --- | --- |
| **Engine colours** | The **Accent** (Train, OK and every primary button, selection, check marks, links, the selected tab's line, the canvas box selector); the 24 **Node families** (the fill of each kind of node on the canvas); the **Data types** (pins and links by what flows through them, and the plot series colours in that order); the **Status words** (Ready, warnings, errors, running ...). A status colour that is hard to read on the window shows its contrast and the 4.5:1 target under it. **Reset the Engine colours to the base theme's** takes the preset's again. |
| **Surfaces** | Window, text, fields, buttons, bars, tabs, tables, lists, menus and popups, dialogs, scrollbars. A filter box narrows the rows. What follows the accent (check marks, selection, the tab line) has no row here; splitters and resize grips follow the window colour. |
| **Canvas** | The grid (background, lines, major lines). Nodes, pins and links use the Engine colours. |
| **Shape** | Rounding, borders, padding, spacing and sizes, as typed numbers with - / + steps. Fields and buttons have no outline (a fill marks them); windows keep a line. |
| **Saved themes** | Your custom themes, with **Apply**, and where they are stored. |

## Colour fields

Each colour row has a swatch (click it for a picker), a **hex** field
(`#RRGGBB`, or `#RRGGBBAA` where the colour has opacity; anything that is not
six or eight hex digits is ignored) and **R**, **G**, **B** (and **A**) typed
from 0 to 255, with - / + steps.

## Keeping your changes

Edits apply live but last only for the session until they are saved. While
there are unsaved changes the editor says so, with **Save as custom theme...**
and **Discard changes**. A custom theme keeps the base theme, every surface
colour, the shape, and the Engine colours (accent, node families, data types,
status words). The saved theme is applied again at the next start.
