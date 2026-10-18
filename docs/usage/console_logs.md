# Console and logs

The **Console** (bottom dock) lists what the Engine logs, newest last. The
same messages go to `engine_log.txt` beside `cyxwiz-engine.exe`. Each start
keeps the four before it as `engine_log.1.txt` (the previous launch) to
`engine_log.4.txt`, so the log of a launch that crashed is still there after
the restart.

## Levels

The chips above the list count and show each level: **Trace**, **Debug**,
**Info**, **Warn**, **Error**, **Critical**. Click a chip to hide or show
that level. **Search messages** filters by text; the structured filter
takes expressions such as `level >= warn and category = "device"`. **Pause**
stops the list from moving, **Auto-scroll** follows new messages.

Each line has a **category** naming the part of the Engine it comes from:
`engine`, `compute` (backends, ArrayFire, the saved compute choice),
`device`, `ui` (window, fonts, panels), `plugin`, `python`, `project`,
`account`, `network`, `training` (training runs, checkpoints, generation
previews), `data` and `graph`. A plugin's own lines carry its logger's
name; `system` is left for lines no rule names yet. Filter by it, for example
`category = "training" and level >= warn`.

At start the Engine writes one Info line per step (the Engine and its folder,
the compute backend, the devices, the window, the plugins, Python, the project
and its files, the account and the network); the detail behind each step
(fonts, panels, each device and plugin path, Python's paths) is at Debug.

- **Info**: what happened, once per action: a run started and finished,
  each node's outcome (a check passed with its count, an SQL step's rows in
  and out, a Data Input's rows), a node completed.
- **Debug**: the detail behind it: every node starting, each dataset
  registered or added to the catalog, which operator ran a node, the
  pipeline sent to the executor (one line of JSON), canvas selection.
- **Warn** and **Error**: something needs a look.

The list shows **Time**, **Level**, **Category** and **Message**. **Source**,
**Code**, **Run** and **Device** appear only when a shown row has a value for
them; the status line says which are hidden. A message wraps to two lines in
its column and ends with "..." when it goes on. Click a row for its details:
beside the list when the Console is wide (the list keeps its height), under
it otherwise. They show the whole message in a field you can select from, a
JSON ending formatted (tick **Show raw** for it as written), every field
(click a value to copy it), and **Copy message**, **Copy row** and **Copy as
JSON**. Esc or **Close** closes them.

For the Berean tier1 recipe (a Data Input, an SQL step and four Row Count
Checks) a run writes 17 Info lines and 45 Debug lines; nothing is logged
while the Engine is idle.

## Searching from Commands

Open a **Commands** session with **+** and type `show logs ...` (for
example `show logs grep generation`, `show logs warnings`, `show logs where
category=training and level>=warn`; `help` lists every form). Each match is a
row:

- the event number, the local time (hover it for the full UTC time), the
  level and the category in fixed columns;
- the message in its own column. **Wrap** (the default, top right) wraps a
  long message inside that column; **One line** keeps each to one line, cut
  with "...";
- under it, the event's other fields on one faint line (a code, a run, a
  backend, a device, its details), when it has any;
- a message that ends in JSON (a generation sample, a provenance record)
  shows its text and a **Show JSON** link with the size and number of
  entries; the JSON opens formatted, arrays of numbers on one line, in a
  field you can select and copy from;
- click a message to open it in a field you can select from; right-click
  for **Copy message** and **Copy row**.

**Open in Logs** on the command's line shows the same events in the Logs
session, with the command's filter in its structured filter.

## Recording Debug

The Engine records Debug messages in every build, since the people using it
are developers. **Preferences > General > Logs > Record Debug messages**
turns them off (Info, warnings and errors only) at once and for the next
starts; it applies to every project and is saved in the Engine settings
(`engine_config.json`, `observability.record_debug_logs`).
