---
title: Editing values
description: How number fields, - and + steps, text fields, dropdowns, checkboxes, file, column and colour pickers work.
type: reference
area: general
audience: user
status: experimental
since: 1.0.8
owner: main-window
last_reviewed: 2026-11-24
related:
  - ../user-guide/graph-editor/node-properties.md
  - ./properties.md
  - ./node-dialogs.md
  - ./preferences.md
redirect_from: []
---

The Engine uses the same few controls everywhere you set a value: in the
**Properties** panel, in node dialogs, in **Preferences** and in the tool
panels. This page says how each control behaves and when your change takes
effect.

## When a change takes effect

Where you edit decides when the value is used:

| Where | When the change applies | How to take it back |
| --- | --- | --- |
| **Properties** panel | At once, while you type or pick | The row's reset arrow, or **Actions > Reset all settings to defaults** |
| A node dialog (Data Input, Data Loader, Data Split, ...) | When you select **Apply** or **OK** | **Cancel** restores what the dialog had when you opened it |
| **Preferences** | At once, and saved at once | **Restore this page's defaults** |
| Tool panels (Data Studio views, Utilities, Theme Editor, ...) | At once, for that panel | Change it back |

In a node dialog, **Apply** stays greyed until you change something; hover it
to see why. **Apply** and **OK** are also greyed while the dialog is busy
(for example while a file loads). **Cancel** always works.

Setting changes are not separate undo steps on the canvas. **Ctrl+Z** goes
back to the last change to nodes or links, and every setting you changed
after that change goes back with it. Use the reset arrow to undo one setting.

## Number fields

Numbers are typed, never dragged: the Engine has no sliders or drag fields
for numbers, so you always see and set the exact value.

- Click the field and type the value.
- Select **-** or **+** next to the field to step the value down or up.
  Hold **Ctrl** while you select **-** or **+** to take a bigger step
  (10 steps for decimal numbers, 100 for whole numbers in the
  **Properties** panel).
- In **Preferences** and the tool panels, hover the field to see its
  allowed range, for example `1 to 50`. In the **Properties** panel, hover
  the setting's name to see its description, its default and whether it is
  required.
- A value outside the range is set to the nearest end of the range.

In a narrow **Properties** panel the **-** and **+** buttons drop out to
leave room for the value. Widen the panel to get them back.

The step in the **Properties** panel follows the setting's range: 0.01 for a
range up to 1, 0.1 up to 10, and 1 beyond. A value smaller than the step,
such as `0.00001`, steps by its own size, so one press never jumps it by
orders of magnitude. Small values keep their digits when saved (`1e-09`
stays `1e-09`).

Two kinds of number field exist today, and they apply at different times:

| Field | Where | Applies |
| --- | --- | --- |
| Setting rows | **Properties** panel | As you type |
| Other number fields | **Preferences**, tool panels, the Theme Editor | When you press **Enter** or leave the field. **Esc** leaves the field without applying. |

## Text fields

- One-line fields take the text as you type it.
- Long-text settings show a box four lines high.
- Secret settings (passwords and keys) show dots instead of the text.

## Checkboxes

Select the box to switch the setting on or off. In the saved graph the value
is `true` or `false`.

## Dropdowns

Select the field to open the list, then select one entry. The list shows the
words for each choice when the setting has them (for example a Normalize
node's **Same for every channel**, **Per channel** and **ImageNet**), otherwise the value saved in the file. The
**Node Info** panel shows the saved values, so the same setting can read
differently there.

## File and folder pickers

A file or folder setting has a path field and a folder button.

1. Type or paste a path into the field, or select the folder button
   (tooltip **Choose a file** or **Choose a folder**).
2. In the system dialog, select the file or folder.
3. Hover the path to read all of it when it is wider than the field.

A setting that writes a file, such as a fitted-state file, opens a **Save**
dialog instead of an **Open** dialog. In the **Data Input** dialog the
buttons are labelled **Browse file...** and **Browse folder...**.

## Column pickers

A column setting is a dropdown of the columns that reach the node or exist
in the file.

- In the **Properties** panel the first entry is
  **(the Data Input's label)**: it keeps the label column chosen in the
  Data Input.
- In the **Data Input** dialog the first entry says what an empty choice
  means, for example **(none)** or **(none: the file's order)**.
- Before the columns are known, the picker is a text field: type the column
  name. In the **Properties** panel, select **Execute Pipeline** on the
  Studio bar to fill the list with the columns the steps before the node make.

## Colour fields

Colour fields (in the Theme Editor and some views) have three parts:

1. A swatch: select it to open a colour picker.
2. A hex field: type `#RRGGBB`, or `#RRGGBBAA` where the field has an alpha
   part, then press **Enter** or leave the field. A value that is not a
   valid hex colour is ignored and the colour stays as it was.
3. **R**, **G**, **B** (and **A**) number fields from 0 to 255, with **-**
   and **+** steps.

## Values that are wrong

A setting the Engine can't use shows a short note under its row in the
**Properties** panel:

| Note | Meaning |
| --- | --- |
| **Required** | The setting is empty but the node needs it |
| **Invalid integer** / **Invalid number** | The saved text is not a number of that kind |
| **Value must be >= N** / **Value must be <= N** | The value is outside the range |
| **Invalid option** | The saved value is not one of the choices |
| **One number per channel, separated by commas** | A per-channel setting has the wrong form |
| **Each value must be > 0** | A per-channel value is zero or negative |
| **Unsupported value; expected X** | A value set by the Engine was changed in the file |

These notes come mostly from graph files edited by hand or saved by an older
Engine, because the controls keep typed values in range. Fix the value in
the row, or reset it with the row's reset arrow.

## Read-only values

Some rows show a value you can't edit, for example a format version set by
the Engine. Hover the value to see why (**Set by the engine; not editable**).
A node whose settings live in its dialog shows **This node is set up in its
dialog.** and an **Open Dialog...** button instead of rows.

## Related

- [Set node properties](../user-guide/graph-editor/node-properties.md)
- [Properties panel](./properties.md)
- [Node dialogs](./node-dialogs.md)
- [Preferences](./preferences.md)
