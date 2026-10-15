# Console and logs

The **Console** (bottom dock) lists what the Engine logs, newest last. The
same messages go to `engine_log.txt` beside `cyxwiz-engine.exe` (a new file
each start).

## Levels

The chips above the list count and show each level: **Trace**, **Debug**,
**Info**, **Warn**, **Error**, **Critical**. Click a chip to hide or show
that level. **Search messages** filters by text; the structured filter
takes expressions such as `level >= warn and category = "device"`. **Pause**
stops the list from moving, **Auto-scroll** follows new messages.

- **Info**: what happened, once per action: a run started and finished,
  each node's outcome (a check passed with its count, an SQL step's rows in
  and out, a Data Input's rows), a node completed.
- **Debug**: the detail behind it: every node starting, each dataset
  registered or added to the catalog, which operator ran a node, the
  pipeline sent to the executor (one line of JSON), canvas selection.
- **Warn** and **Error**: something needs a look.

For the Berean tier1 recipe (a Data Input, an SQL step and four Row Count
Checks) a run writes 17 Info lines and 45 Debug lines; nothing is logged
while the Engine is idle.

## Recording Debug

The Engine records Debug messages in every build, since the people using it
are developers. **Preferences > General > Logs > Record Debug messages**
turns them off (Info, warnings and errors only) at once and for the next
starts; it applies to every project and is saved in the Engine settings
(`engine_config.json`, `observability.record_debug_logs`).
